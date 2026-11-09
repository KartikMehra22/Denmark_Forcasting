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

