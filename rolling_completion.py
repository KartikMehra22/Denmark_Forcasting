"""Locked weekly rolling refits, daily price-only LEAR and sequential CQR."""
from pathlib import Path
from datetime import timedelta
import os,json,sys,subprocess
ROOT=Path(__file__).resolve().parent
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

