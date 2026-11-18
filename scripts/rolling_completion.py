"""Locked weekly rolling refits, daily price-only LEAR and sequential CQR."""
from pathlib import Path
from datetime import timedelta
import os,json,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np,pandas as pd,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,LassoLarsIC
from controlled_study import source,samples,fitpatch,predictpatch
from chronos import ChronosBoltPipeline
O=ROOT/'results/completion';O.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'completion_protocol.json').read_text())
torch.set_num_threads(2)

def extended(a,mode):
 s=source(a,mode)
 raw=json.loads((ROOT/'data/later_locked'/f'{a}.json').read_text())['records'];f=pd.DataFrame(raw)
 t=pd.Series(f.DayAheadPriceEUR.values,index=pd.to_datetime(f.TimeUTC,utc=True)).sort_index()
 assert not t.index.duplicated().any()
 if mode=='hourly':
  assert (t.resample('h').count()==4).all();t=t.resample('h').mean()
 out=pd.concat([s,t[t.index>s.index.max()]])
 step=pd.Timedelta(hours=1 if mode=='hourly' else .25)
 assert (out.index.to_series().diff().dropna()==step).all()
 return out

def lear_rows(s):
 local=s.index.tz_convert('Europe/Copenhagen');f=pd.DataFrame({'d':local.date,'h':local.hour,'p':s.values})
 curve=f.pivot_table(index='d',columns='h',values='p',aggfunc='mean').reindex(columns=range(24)).interpolate(axis=1,limit_direction='both')
 rows={}
 for d in curve.index:
  if all(d-timedelta(days=l) in curve.index for l in [1,2,3,7]):
   x=np.array([curve.loc[d-timedelta(days=l),h] for h in range(24) for l in [1,2,3,7]]+[float(d.weekday()==w) for w in range(7)])
   rows[d]=(x,curve.loc[d].values)
 return rows

def invariant(a):
 med=np.median(a,axis=0);scale=np.median(abs(a-med),axis=0)/.6744897501960817;scale=np.where(scale>1e-8,scale,1.)
 return np.arcsinh((a-med)/scale),med,scale

def lear_predict(rows,d):
 days=[k for k in rows if d-timedelta(days=730)<=k<d-timedelta(days=1)]
 x=np.stack([rows[k][0] for k in days]);y=np.stack([rows[k][1] for k in days])
 xs,med,scale=invariant(x[:,:-7]);ys,ym,yscale=invariant(y)
 xx=np.c_[xs,x[:,-7:]];test=np.r_[np.arcsinh((rows[d][0][:-7]-med)/scale),rows[d][0][-7:]][None,:]
 pred=[]
 for h in range(24):
  alpha=LassoLarsIC(criterion='aic',max_iter=2500).fit(xx,ys[:,h]).alpha_
  fit=Lasso(alpha=max(alpha,1e-10),max_iter=2500).fit(xx,ys[:,h]);pred.append(fit.predict(test)[0])
 return np.sinh(np.array(pred))*yscale+ym

def run():
 bolt=ChronosBoltPipeline.from_pretrained('amazon/chronos-bolt-tiny',revision='a0e552de83495b5c28c14c71c374f3e33280b340',local_files_only=os.environ.get('DENMARK_ALLOW_MODEL_DOWNLOAD') != '1',device_map='cpu',dtype=torch.float32)
 logs=[]
 for mode in ['hourly','quarter_hour']:
  for a in ['DK1','DK2']:
   s=extended(a,mode);rows=samples(s,mode);bydate={r[0]:r for r in rows}
   old=json.loads((ROOT/'results/controlled/configuration.json').read_text())[f'{mode}_{a}'];start=pd.Timestamp(old['test_start']).date()
   dates=[d for d in bydate if d>=start];steps=1 if mode=='hourly' else 4;h=25*steps
   # Generate quantiles for calibration before first evaluation day and all later origins.
   qdates=[d for d in bydate if d>=start-timedelta(days=62)]
   contexts=np.stack([bydate[d][2] for d in qdates]);qs=[]
   with torch.no_grad():
    for b in torch.from_numpy(contexts).split(16):qs.append(bolt.predict(b,prediction_length=h).numpy())
   quant=np.concatenate(qs);qi=list(bolt.quantiles);qmap=dict(zip(qdates,quant))
   scores={}
   for d in qdates:
    r=bydate[d];n=r[4];q=qmap[d];scores[d]=np.maximum.reduce([q[qi.index(.1),:n]-r[3][:n],r[3][:n]-q[qi.index(.9),:n],np.zeros(n)])
   conf=old['patchtst_config'];lr=lear_rows(s) if mode=='hourly' else None
   result=[]
   for offset in range(0,len(dates),7):
    block=dates[offset:offset+7];d=block[0]
    train=[r for r in rows if d-timedelta(days=730)<=r[0]<d-timedelta(days=2)]
    x=np.stack([r[2] for r in train]);y=np.stack([r[3] for r in train]);ex=np.stack([bydate[k][2] for k in block]);assert np.isfinite(y).all()
    issue=pd.Timestamp(str(d-timedelta(days=1))+' 10:00',tz='Europe/Copenhagen').tz_convert('UTC')
    step=pd.Timedelta(minutes=60//steps)
    latest=max(r[1]+(h-1)*step for r in train);assert latest<issue
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=CFG['ridge_alpha'][mode],solver='svd'));ridge.fit(x,y);rp=ridge.predict(ex)
    tmp=O/f'{mode}_{a}_worker.npz';dest=O/f'{mode}_{a}_worker.npy';np.savez(tmp,x=x,y=y,ex=ex)
    subprocess.run([sys.executable,str(ROOT/'scripts'/'fit_lgb_worker.py'),str(tmp),str(dest)],check=True);lp=np.load(dest)
    net,_,_=fitpatch(x,y,10,42,conf);pp=predictpatch(net,ex)
    # Persist each fitted checkpoint for genuine reuse and forecasting.
    import joblib
    modeldir=O/'models'/f'{mode}_{a}_{d}';modeldir.mkdir(parents=True,exist_ok=True)
    joblib.dump(ridge,modeldir/'ridge.joblib');torch.save(net.state_dict(),modeldir/'patchtst.pt')
    logs.append(dict(mode=mode,area=a,origin=str(d),training_days=len(train),latest_label_utc=str(latest),issue_utc=str(issue)))
    for j,k in enumerate(block):
     row=bydate[k];n=row[4];q=qmap[k];idx=pd.date_range(row[1],periods=n,freq=step);local=idx.tz_convert('Europe/Copenhagen')
     def lag(l):
      hist=s[s.index.tz_convert('Europe/Copenhagen').date==k-timedelta(days=l)];loc=hist.index.tz_convert('Europe/Copenhagen')
      curve=pd.Series(hist.values,index=loc.hour*steps+loc.minute//(60//steps)).groupby(level=0).mean().reindex(range(24*steps)).interpolate(limit_direction='both')
      return curve.reindex(local.hour*steps+local.minute//(60//steps)).values
     prior=[v for dd,v in scores.items() if k-timedelta(days=61)<=dd<k-timedelta(days=1)]
     cal=np.concatenate(prior);rank=min(len(cal),int(np.ceil((len(cal)+1)*.8)));expansion=float(np.partition(cal,rank-1)[rank-1])
     f=pd.DataFrame(dict(mode=mode,area=a,date=str(k),time_utc=idx,actual=row[3][:n],previous_day=lag(1),previous_week=lag(7),ridge=rp[j,:n],lightgbm=lp[j,:n],patchtst=pp[j,:n],chronos=q[qi.index(.5),:n],raw_lower=q[qi.index(.1),:n],raw_upper=q[qi.index(.9),:n],cal_lower=q[qi.index(.1),:n]-expansion,cal_upper=q[qi.index(.9),:n]+expansion,cal_expansion=expansion,fit_origin=str(d),split='later_locked' if k>=pd.Timestamp(CFG['later_start']).date() else 'historical'))
     if lr is not None:f['lear']=lear_predict(lr,k)[local.hour]
     result.append(f)
    pd.concat(result).to_csv(O/f'{mode}_{a}_predictions.csv',index=False)
    (O/'training_log.json').write_text(json.dumps(logs,indent=2))
    print(mode,a,d,'rolling block complete',flush=True)

if __name__=='__main__':run()
