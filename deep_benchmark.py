"""Fixed-origin PatchTST and Chronos-Bolt-tiny comparisons. No test tuning."""
from pathlib import Path
from datetime import timedelta
import os,json
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'.cache'/'huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np
import pandas as pd
import torch
from transformers import PatchTSTConfig,PatchTSTForPrediction
from chronos import ChronosBoltPipeline
from day_ahead_study import load_area

OUT=ROOT/'results'/'benchmark_v2'
torch.set_num_threads(2)
torch.manual_seed(42);np.random.seed(42)
CONFIG=dict(context_length=168,prediction_length=25,patch_length=16,patch_stride=8,
 d_model=32,num_hidden_layers=2,num_attention_heads=4,ffn_dim=64,num_input_channels=1,
 scaling='std',loss='mse',head_dropout=0.1)

def windows(s,dates):
    x,y,kept=[],[],[]
    for d in dates:
        start=pd.Timestamp(d).tz_localize('Europe/Copenhagen').tz_convert('UTC')
        i=s.index.get_indexer([start])[0]
        if i<168 or i+25>len(s):continue
        x.append(s.iloc[i-168:i].values);y.append(s.iloc[i:i+25].values);kept.append(d)
    return torch.tensor(np.array(x),dtype=torch.float32).unsqueeze(-1),torch.tensor(np.array(y),dtype=torch.float32).unsqueeze(-1),kept

def train(x,y,epochs,validation=None):
    torch.manual_seed(42)
    model=PatchTSTForPrediction(PatchTSTConfig(**CONFIG))
    opt=torch.optim.AdamW(model.parameters(),lr=0.001)
    best=float('inf');best_epoch=1;log=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for inds in torch.randperm(len(x)).split(32):
            opt.zero_grad();loss=model(past_values=x[inds],future_values=y[inds]).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
        row={'epoch':epoch,'train_mse':float(np.mean(losses))}
        if validation is not None:
            vx,vy=validation;model.eval()
            with torch.no_grad():v=model(past_values=vx).prediction_outputs
            mae=(v-vy).abs().mean().item();row['validation_mae']=mae
            if mae<best:best=mae;best_epoch=epoch
        log.append(row)
        print('PatchTST',row,flush=True)
    return model,best_epoch,log

def run():
    cfg=json.loads((OUT/'config.json').read_text())
    val=pd.Timestamp(cfg['validation_start']).date();test=pd.Timestamp(cfg['exploratory_test_start']).date()
    base=pd.read_csv(OUT/'predictions.csv',parse_dates=['time_utc','issue_utc'])
    frames=[];logs={}
    pipeline=ChronosBoltPipeline.from_pretrained('amazon/chronos-bolt-tiny',device_map='cpu',dtype=torch.float32)
    for area in ['DK1','DK2']:
        s,_=load_area(area)
        dates=sorted(set(s.index.tz_convert('Europe/Copenhagen').date))
        # For 25-step training labels, exclude two delivery days before cutoff.
        trdates=[d for d in dates if val-timedelta(days=732)<=d<val-timedelta(days=2)]
        vadates=[d for d in dates if val<=d<test-timedelta(days=2)]
        x,y,_=windows(s,trdates);vx,vy,_=windows(s,vadates)
        _,epochs,history=train(x,y,15,(vx,vy))
        finaldates=[d for d in dates if test-timedelta(days=732)<=d<test-timedelta(days=2)]
        fx,fy,_=windows(s,finaldates)
        model,_,_=train(fx,fy,epochs)
        torch.save(model.state_dict(),OUT/f'patchtst_{area}.pt')
        evalbase=base[(base.area==area)&(base.split=='exploratory_test')].copy()
        edates=sorted(pd.to_datetime(evalbase.date).dt.date.unique())
        # Prediction windows need only past; final delivery day need not have 25 future labels.
        contexts=[]
        for d in edates:
            start=pd.Timestamp(d).tz_localize('Europe/Copenhagen').tz_convert('UTC')
            i=s.index.get_indexer([start])[0]
            contexts.append(s.iloc[i-168:i].values)
        ex=torch.tensor(np.array(contexts),dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            patch=torch.cat([model(past_values=b.unsqueeze(-1)).prediction_outputs[:,:,0] for b in ex.split(32)]).numpy()
            bolt=torch.cat([pipeline.predict(b,prediction_length=25) for b in ex.split(32)]).numpy()
        median_idx=list(pipeline.quantiles).index(0.5)
        blocks=[]
        for j,d in enumerate(edates):
            g=evalbase[pd.to_datetime(evalbase.date).dt.date==d].copy();n=len(g)
            g['patchtst']=patch[j,:n];g['chronos_bolt_tiny']=bolt[j,median_idx,:n]
            blocks.append(g)
        frames.append(pd.concat(blocks))
        logs[area]={'selected_epochs':epochs,'history':history,'final_training_days':len(fx)}
        print(area,'deep predictions complete',flush=True)
    p=pd.concat(frames);p.to_csv(OUT/'all_model_predictions.csv',index=False)
    metrics=[]
    for area,g in p.groupby('area'):
        b=(g.previous_day-g.actual).abs().mean()
        for name in ['previous_day','previous_week','ridge','lightgbm','patchtst','chronos_bolt_tiny']:
            e=g[name]-g.actual
            assert e.notna().all()
            metrics.append(dict(area=area,model=name,n=len(g),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean()),improvement_pct=100*(1-e.abs().mean()/b)))
    m=pd.DataFrame(metrics);m.to_csv(OUT/'all_model_metrics.csv',index=False)
    (OUT/'deep_config.json').write_text(json.dumps({'patchtst':CONFIG,'training':logs,'chronos':'amazon/chronos-bolt-tiny','refit':'PatchTST fixed at test origin; ridge/LightGBM monthly. Different refit schedules are a comparison limitation.'},indent=2))
    print(m.to_string(index=False),flush=True)

if __name__=='__main__':run()
