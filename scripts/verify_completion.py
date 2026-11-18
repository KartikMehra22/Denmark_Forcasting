from pathlib import Path
from datetime import timedelta
import json,numpy as np,pandas as pd,hashlib
from rolling_completion import extended,CFG,O,ROOT
log=json.loads((O/'training_log.json').read_text())
assert all(pd.Timestamp(r['latest_label_utc'])<pd.Timestamp(r['issue_utc']) for r in log)
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
    s=extended(area,mode);p=pd.read_csv(O/f'{mode}_{area}_predictions.csv',parse_dates=['time_utc'])
    step=pd.Timedelta(minutes=60 if mode=='hourly' else 15)
    assert not p.time_utc.duplicated().any();assert np.isfinite(p.select_dtypes('number').values).all()
    for d,g in p.groupby('date'):
        start=pd.Timestamp(d,tz='Europe/Copenhagen');end=start+pd.DateOffset(days=1)
        assert set(g.time_utc)==set(pd.date_range(start,end,freq=step,inclusive='left').tz_convert('UTC'))
        np.testing.assert_allclose(g.actual,s.reindex(g.time_utc),atol=1e-4,rtol=1e-6)
        assert (g.cal_lower<=g.raw_lower).all() and (g.cal_upper>=g.raw_upper).all()
        assert (pd.to_datetime(g.fit_origin).dt.date<=pd.Timestamp(d).date()).all()
    # Independently reconstruct every later-day expansion from earlier stored forecasts.
    day_scores={}
    for date,g in p.groupby('date'):
        day_scores[pd.Timestamp(date).date()]=np.maximum.reduce([g.raw_lower.to_numpy()-g.actual.to_numpy(),g.actual.to_numpy()-g.raw_upper.to_numpy(),np.zeros(len(g))])
    for date,g in p[p.split=='later_locked'].groupby('date'):
        d=pd.Timestamp(date).date()
        old=[v for dd,v in day_scores.items() if d-timedelta(days=61)<=dd<d-timedelta(days=1)]
        values=np.concatenate(old);rank=min(len(values),int(np.ceil((len(values)+1)*.8)))
        expansion=np.partition(values,rank-1)[rank-1]
        np.testing.assert_allclose(g.cal_expansion,expansion,rtol=1e-5,atol=1e-4)
    assert p[p.split=='historical'].date.nunique()==(180 if mode=='hourly' else 90)
    assert (p.raw_lower<=p.raw_upper).all()
    fits=[r for r in log if r['mode']==mode and r['area']==area]
    assert len(fits)==(29 if mode=='hourly' else 17)
    origins=pd.to_datetime([r['origin'] for r in fits]);assert (origins.to_series().diff().dropna()==pd.Timedelta(days=7)).all()
    later=p[p.split=='later_locked'];assert later.date.nunique()==23
    expected=23*24*(1 if mode=='hourly' else 4)+(1 if mode=='hourly' else 4)
    assert len(later)==expected
    # DST fall-back adds a real hour rather than dropping/repeating target labels.
    assert len(p[p.date=='2026-10-25'])==(25 if mode=='hourly' else 100)
    print(mode,area,'PASS all days, actuals, finite outputs, rolling origins, widening and DST')
# Recompute metrics independently from persisted predictions.
m=pd.read_csv(O/'metrics.csv')
for r in m.itertuples():
 p=pd.read_csv(O/f'{r.mode}_{r.area}_predictions.csv');g=p[p.split==r.split];e=g[r.model]-g.actual
 np.testing.assert_allclose([e.abs().mean(),np.sqrt((e**2).mean())],[r.MAE,r.RMSE],rtol=1e-6,atol=1e-5)
(O/'verification.json').write_text(json.dumps({'status':'PASS','checks':['training label embargo','raw target alignment','complete local delivery days','23 later days','autumn DST day','finite predictions','calibrated intervals widen raw','independent metrics']},indent=2))
print('Independent metric verification PASS')
