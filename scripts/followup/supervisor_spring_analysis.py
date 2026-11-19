"""Verify spring forecasts and calculate adjusted daily-loss comparisons."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import json
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from rolling_completion import extended
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';metrics=[];tests=[]
for area in ['DK1','DK2']:
 mode='quarter_hour';p=pd.read_csv(O/f'spring_{mode}_{area}.csv',parse_dates=['time_utc']);cfg=json.loads((O/f'spring_{mode}_{area}_settings.json').read_text());s=extended(area,mode)
 assert p.date.min()=='2026-03-01' and p.date.max()=='2026-05-31' and p.date.nunique()==92 and not p.time_utc.duplicated().any()
 assert cfg['validation_end']<'2026-03-01' and cfg['validation_training_end']<cfg['validation_start']
 np.testing.assert_allclose(p.actual,s.reindex(pd.DatetimeIndex(p.time_utc)),atol=1e-4,rtol=1e-6)
 assert np.isfinite(p.select_dtypes('number').to_numpy()).all()
 for day,g in p.groupby('date'):
  start=pd.Timestamp(day,tz='Europe/Copenhagen');expect=pd.date_range(start,start+pd.DateOffset(days=1),freq='15min',inclusive='left').tz_convert('UTC');assert pd.DatetimeIndex(g.time_utc).equals(expect)
  origin=pd.Timestamp(g.fit_origin.iloc[0]);assert origin<=pd.Timestamp(day)<origin+pd.Timedelta(days=7)
 for origin in p.fit_origin.unique():
  d=O/'spring_models'/f'{mode}_{area}_{origin}'
  assert all((d/f).is_file() for f in ['ridge.joblib','patchtst.pt','lightgbm.json.gz'])
 for m in ['previous_day','ridge','lightgbm','patchtst','chronos']:
  e=p[m]-p.actual;metrics.append(dict(mode=mode,area=area,period='spring_independently_tuned',model=m,days=p.date.nunique(),n=len(e),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean())))
  if m=='previous_day':continue
  delta=p.assign(v=(p.previous_day-p.actual).abs()-(p[m]-p.actual).abs()).groupby('date').v.mean().to_numpy();fit=sm.OLS(delta,np.ones(len(delta))).fit(cov_type='HAC',cov_kwds={'maxlags':6,'use_correction':False});z=float(fit.tvalues[0]);tests.append(dict(mode=mode,area=area,period='spring_independently_tuned',model=m,days=len(delta),mean_daily_mae_improvement=float(delta.mean()),hac_lag=6,z=z,p_raw=2*norm.sf(abs(z))))
print('Spring chronology, dates, finite predictions, raw targets and all checkpoints PASS')
pd.DataFrame(metrics).to_csv(O/'spring_metrics.csv',index=False)
old=pd.read_csv(O/'dm_tests.csv');new=pd.DataFrame(tests);combined=pd.concat([old,new],ignore_index=True);mask=combined.p_raw.notna();combined['p_holm_all_periods_including_spring']=np.nan;combined.loc[mask,'p_holm_all_periods_including_spring']=multipletests(combined.loc[mask,'p_raw'],method='holm')[1];combined.to_csv(O/'dm_tests_with_spring.csv',index=False)
seasonal=pd.read_csv(O/'seasonal_dm_tests.csv');spring_seasonal=new.copy();spring_seasonal['experiment']='spring_independently_tuned';spring_seasonal['season']='spring';all_seasonal=pd.concat([seasonal,spring_seasonal],ignore_index=True);valid=all_seasonal.p_raw.notna();all_seasonal['p_holm_with_spring']=np.nan;all_seasonal.loc[valid,'p_holm_with_spring']=multipletests(all_seasonal.loc[valid,'p_raw'],method='holm')[1];all_seasonal.to_csv(O/'seasonal_dm_tests_with_spring.csv',index=False)
print(pd.DataFrame(metrics).to_string(index=False));print(combined[combined.period=='spring_independently_tuned'][['area','model','mean_daily_mae_improvement','p_raw','p_holm_all_periods_including_spring']].to_string(index=False))
