"""Matched-input, fixed-origin comparison and native quarter-hour study."""
from pathlib import Path
from datetime import timedelta
import os,json,hashlib
ROOT=Path(__file__).resolve().parent
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

