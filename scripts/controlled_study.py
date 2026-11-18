"""Matched-input, fixed-origin comparison and native quarter-hour study."""
from pathlib import Path
from datetime import timedelta
import os,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache'/'huggingface');os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np,pandas as pd,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from transformers import PatchTSTConfig,PatchTSTForPrediction
from chronos import ChronosBoltPipeline
from day_ahead_study import load_area
OUT=ROOT/'results'/'controlled';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)

def source(area,mode):
    if mode=='hourly':return load_area(area)[0]
    f=pd.read_csv(ROOT/'data'/f'dayahead_{area}.csv')
    return pd.Series(f.DayAheadPriceEUR.values,index=pd.to_datetime(f.TimeUTC,utc=True)).sort_index()

def samples(s,mode):
    steps=1 if mode=='hourly' else 4;c=168*steps;h=25*steps
    dates=sorted(set(s.index.tz_convert('Europe/Copenhagen').date));out=[]
    for d in dates:
        start=pd.Timestamp(d).tz_localize('Europe/Copenhagen').tz_convert('UTC');i=s.index.get_indexer([start])[0]
        end=(pd.Timestamp(d).tz_localize('Europe/Copenhagen')+pd.DateOffset(days=1)).tz_convert('UTC')
        n=int((end-start).total_seconds()/3600*steps)
        if i<c or i+n>len(s):continue
        x=s.iloc[i-c:i].to_numpy(dtype='float32')
        y=np.full(h,np.nan,dtype='float32');v=s.iloc[i:i+h].values;y[:len(v)]=v
        out.append((d,start,x,y,n))
    return out

def fitpatch(x,y,epochs,seed,config,val=None):
    torch.manual_seed(seed);net=PatchTSTForPrediction(PatchTSTConfig(**config))
    opt=torch.optim.AdamW(net.parameters(),lr=.001);best=(float('inf'),1);log=[]
    X=torch.from_numpy(x).unsqueeze(-1);Y=torch.from_numpy(y).unsqueeze(-1)
    for epoch in range(1,epochs+1):
        net.train()
        for ix in torch.randperm(len(X)).split(32):
            opt.zero_grad();loss=net(past_values=X[ix],future_values=Y[ix]).loss;loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if val:
            vx,vy=val;pred=predictpatch(net,vx);mae=float(np.nanmean(abs(pred-vy)));log.append(mae)
            if mae<best[0]:best=(mae,epoch)
    return net,best[1],log

def predictpatch(net,x):
    net.eval()
    with torch.no_grad():return torch.cat([net(past_values=b.unsqueeze(-1)).prediction_outputs[:,:,0] for b in torch.from_numpy(x).split(32)]).numpy()

def run():
    bolt=ChronosBoltPipeline.from_pretrained('amazon/chronos-bolt-tiny',device_map='cpu',dtype=torch.float32,revision='a0e552de83495b5c28c14c71c374f3e33280b340',local_files_only=os.environ.get('DENMARK_ALLOW_MODEL_DOWNLOAD') != '1')
    metadata={};metrics=[];allpred=[]
    for mode in ['hourly','quarter_hour']:
      for area in ['DK1','DK2']:
        s=source(area,mode);items=samples(s,mode);dates=[r[0] for r in items]
        nt=180 if mode=='hourly' else 90;nv=90 if mode=='hourly' else 60
        test=dates[-nt];val=dates[-nt-nv]
        def arrays(rows):return np.stack([r[2] for r in rows]),np.stack([r[3] for r in rows])
        train=[r for r in items if val-timedelta(days=730)<=r[0]<val-timedelta(days=2)]
        valid=[r for r in items if val<=r[0]<test-timedelta(days=2)]
        final=[r for r in items if test-timedelta(days=730)<=r[0]<test-timedelta(days=2)]
        evaluation=[r for r in items if r[0]>=test]
        x,y=arrays(train);vx,vy=arrays(valid);fx,fy=arrays(final);ex,ey=arrays(evaluation)
        print(mode,area,'arrays',x.shape,flush=True)
        assert np.isfinite(y).all() and np.isfinite(fy).all()
        scores={}
        for alpha in [1,100,10000]:
            r=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd'));r.fit(x,y)
            scores[alpha]=float(np.nanmean(abs(r.predict(vx)-vy)))
        alpha=min(scores,key=scores.get)
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd'));ridge.fit(fx,fy)
        pred={'ridge':ridge.predict(ex)}
        import subprocess,sys
        temp=OUT/f'{mode}_{area}_fit.npz';dest=OUT/f'{mode}_{area}_lgb.npy'
        np.savez(temp,x=fx,y=fy,ex=ex)
        subprocess.run([sys.executable,str(ROOT/'scripts'/'fit_lgb_worker.py'),str(temp),str(dest)],check=True)
        pred['lightgbm']=np.load(dest)
        print('lightgbm done',flush=True)
        steps=1 if mode=='hourly' else 4;c=168*steps;h=25*steps
        conf=dict(context_length=c,prediction_length=h,patch_length=16*steps,patch_stride=8*steps,d_model=32,num_hidden_layers=2,num_attention_heads=4,ffn_dim=64,num_input_channels=1,scaling='std',loss='mse',head_dropout=.1)
        _,epochs,history=fitpatch(x,y,12,42,conf,(vx,vy))
        seeds=[]
        for seed in [42,43,44]:
            net,_,_=fitpatch(fx,fy,epochs,seed,conf);pr=predictpatch(net,ex);seeds.append(pr)
            pred[f'patchtst_seed{seed}']=pr
        pred['patchtst_ensemble']=np.mean(seeds,axis=0)
        with torch.no_grad():
            q=torch.cat([bolt.predict(b,prediction_length=h) for b in torch.from_numpy(ex).split(16)]).numpy()
        qi=list(bolt.quantiles);pred['chronos_bolt_tiny']=q[:,qi.index(.5)]
        pred['chronos_q10']=q[:,qi.index(.1)];pred['chronos_q90']=q[:,qi.index(.9)]
        frames=[]
        for j,(d,start,context,target,n) in enumerate(evaluation):
            idx=pd.date_range(start,periods=n,freq='h' if steps==1 else '15min')
            local=idx.tz_convert('Europe/Copenhagen')
            def lagcurve(lag):
                ld=d-timedelta(days=lag);hist=s[s.index.tz_convert('Europe/Copenhagen').date==ld]
                hi=hist.index.tz_convert('Europe/Copenhagen');slots=hi.hour*steps+hi.minute//(60//steps)
                curve=pd.Series(hist.values,index=slots).groupby(level=0).mean().reindex(range(24*steps)).interpolate(limit_direction='both')
                return curve.reindex(local.hour*steps+local.minute//(60//steps)).values
            f=pd.DataFrame({'mode':mode,'area':area,'date':str(d),'time_utc':idx,'actual':target[:n],
               'previous_day':lagcurve(1),'previous_week':lagcurve(7)})
            for name,values in pred.items():f[name]=values[j,:n]
            frames.append(f)
        p=pd.concat(frames);assert np.isfinite(p.select_dtypes('number').values).all()
        for model in ['previous_day','previous_week',*pred]:
            if model.startswith('chronos_q'):continue
            e=p[model]-p.actual;b=(p.previous_day-p.actual).abs().mean()
            metrics.append(dict(mode=mode,area=area,model=model,n=len(p),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean()),improvement_pct=100*(1-e.abs().mean()/b)))
        key=f'{mode}_{area}';p.to_csv(OUT/f'{key}_predictions.csv',index=False)
        metadata[key]={'validation_start':str(val),'test_start':str(test),'end':str(dates[-1]),'train_validation_days':len(x),'train_final_days':len(fx),'evaluation_days':len(ex),'ridge_alpha':alpha,'ridge_validation_scores':scores,'patchtst_epochs':epochs,'patchtst_validation_mae':history,'patchtst_config':conf,'seeds':[42,43,44], 'retraining':'all locally trained models fixed before first test day','inputs':'same univariate 7-day context for every model; no calendar covariates'}
        (OUT/'configuration.json').write_text(json.dumps(metadata,indent=2))
        pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
        print(key,'complete',metrics[-3:],flush=True)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')}
    (OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2))

if __name__=='__main__':run()
