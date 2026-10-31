"""Daily price-curve models with local-calendar lags and chronological validation.
Run .venv/bin/python benchmark.py. Last 180 days are exploratory (previously inspected).
"""
from datetime import timedelta
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import holidays
from day_ahead_study import load_area

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'benchmark_v2'
OUT.mkdir(parents=True,exist_ok=True)
HOLIDAYS=holidays.Denmark(years=range(2021,2028))

def build():
    raw={a:load_area(a)[0] for a in ['DK1','DK2']}
    curves={}
    for a,s in raw.items():
        local=s.index.tz_convert('Europe/Copenhagen')
        f=pd.DataFrame({'date':local.date,'hour':local.hour,'price':s.values})
        # Fall-back repeated hour averaged ONLY as a historical lag feature.
        curves[a]=f.pivot_table(index='date',columns='hour',values='price',aggfunc='mean').reindex(columns=range(24))
        # Spring-forward missing hour filled ONLY within an already-published historical curve.
        curves[a]=curves[a].interpolate(axis=1,limit_direction='both')
    frames=[]
    for area,s in raw.items():
        rows=[]
        for t,y in s.items():
            local=t.tz_convert('Europe/Copenhagen'); d=local.date(); h=local.hour
            issue=pd.Timestamp(str(d-timedelta(days=1))+' 10:00',tz='Europe/Copenhagen').tz_convert('UTC')
            row={'time_utc':t,'date':d,'issue_utc':issue,'area':area,'actual':y,
                 'hour':h,'weekday':local.dayofweek,'month':local.month,'holiday':int(d in HOLIDAYS)}
            good=True
            for a in raw:
                for lag in [1,2,3,7]:
                    ld=d-timedelta(days=lag)
                    if ld not in curves[a].index:
                        good=False;break
                    vals=curves[a].loc[ld]
                    # Whole D-1 curve was published on D-2 (assumption documented).
                    for hh in range(24):row[f'{a}_d{lag}_h{hh}']=vals[hh]
                    row[f'{a}_d{lag}_same_hour']=vals[h]
                if not good:break
            if good:
                row['previous_day']=row[f'{area}_d1_same_hour']
                row['previous_week']=row[f'{area}_d7_same_hour']
                rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames,ignore_index=True)


