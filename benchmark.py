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


def run():
    df=build()
    # Require complete real delivery days; DST days naturally contain 23/25 records.
    good=[]
    for (a,d),g in df.groupby(['area','date']):
        start=pd.Timestamp(d).tz_localize('Europe/Copenhagen')
        n=int(((start+pd.DateOffset(days=1))-start).total_seconds()/3600)
        if len(g)==n:good.append((a,d))
    keys=set(good)
    df=df[[ (a,d) in keys for a,d in zip(df.area,df.date)]].copy()
    dates=sorted(df.date.unique())
    test_start=dates[-180]; val_start=dates[-270]
    config={'issue_time':'10:00 Europe/Copenhagen D-1','validation_start':str(val_start),
       'exploratory_test_start':str(test_start),'end':str(dates[-1]),'train_days':730,
       'refit':'monthly; all training delivery dates strictly before issue date',
       'ridge_alpha_candidates':[1,100,10000],
       'lightgbm':{'objective':'regression_l1','n_estimators':150,'learning_rate':0.05,'num_leaves':15,'min_child_samples':100,'random_state':42,'n_jobs':2},
       'note':'Test period already inspected in v1; not pristine holdout. No further tuning on its results.'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    excluded={'time_utc','date','issue_utc','area','actual','previous_day','previous_week'}
    cols=[c for c in df if c not in excluded]
    preds=[]; choices={}
    for area,g in df.groupby('area'):
        g=g.sort_values('time_utc').copy()
        for split,start,end in [('validation',val_start,test_start),('exploratory_test',test_start,dates[-1]+timedelta(days=1))]:
            eval_g=g[(g.date>=start)&(g.date<end)]
            blocks=eval_g.groupby(eval_g.time_utc.dt.tz_convert('Europe/Copenhagen').dt.strftime('%Y-%m'))
            for month,b in blocks:
                first=b.date.min(); cutoff=first-timedelta(days=1)
                tr=g[(g.date<cutoff)&(g.date>=first-timedelta(days=730))]
                assert tr.date.max()<cutoff
                out=b[['time_utc','date','issue_utc','area','actual','previous_day','previous_week']].copy()
                out['split']=split
                for alpha in config['ridge_alpha_candidates']:
                    if split=='exploratory_test' and alpha!=choices[area]:continue
                    ridge=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
                    ridge.fit(tr[cols],tr.actual)
                    out[f'ridge_{alpha}']=ridge.predict(b[cols])
                lgb=LGBMRegressor(**config['lightgbm'],verbosity=-1)
                lgb.fit(tr[cols],tr.actual)
                out['lightgbm']=lgb.predict(b[cols])
                preds.append(out)
                print(area,split,month,'complete',flush=True)
            if split=='validation':
                v=pd.concat(preds); v=v[(v.area==area)&(v.split=='validation')]
                choices[area]=min(config['ridge_alpha_candidates'],key=lambda a:np.mean(abs(v[f'ridge_{a}']-v.actual)))
    p=pd.concat(preds,ignore_index=True)
    p['ridge']=np.nan
    for a,alpha in choices.items():
        mask=p.area==a;p.loc[mask,'ridge']=p.loc[mask,f'ridge_{alpha}']
    p.to_csv(OUT/'predictions.csv',index=False)
    metrics=[]
    for (area,split),g in p.groupby(['area','split']):
        base=(g.previous_day-g.actual).abs().mean()
        for model in ['previous_day','previous_week','ridge','lightgbm']:
            e=g[model]-g.actual
            metrics.append({'area':area,'split':split,'model':model,'n':len(g),'MAE':e.abs().mean(),
              'RMSE':np.sqrt((e**2).mean()),'improvement_over_previous_day_pct':100*(1-e.abs().mean()/base)})
    m=pd.DataFrame(metrics);m.to_csv(OUT/'metrics.csv',index=False)
    (OUT/'selected_ridge.json').write_text(json.dumps(choices,indent=2))
    # Sanity checks: rows align, forecast issue precedes target, all models complete.
    assert not p[['actual','ridge','lightgbm','previous_day','previous_week']].isna().any().any()
    assert not p.duplicated(['area','time_utc']).any()
    assert (p.issue_utc<p.time_utc).all()
    daily=[]
    for (a,split,d),g in p.groupby(['area','split','date']):
        daily.append({'area':a,'split':split,'date':d,**{model:float((g[model]-g.actual).abs().mean()) for model in ['previous_day','previous_week','ridge','lightgbm']}})
    pd.DataFrame(daily).to_csv(OUT/'daily_mae.csv',index=False)
    print(m.to_string(index=False),flush=True)

if __name__=='__main__':run()
