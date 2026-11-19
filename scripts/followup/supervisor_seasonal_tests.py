"""Exploratory seasonal DM-style tests using daily MAE and HAC(6); Holm family-wide."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import norm
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';rows=[]
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
  original=pd.read_csv(R/f'results/completion/{mode}_{area}_predictions.csv',parse_dates=['time_utc']);original['experiment']='original'
  if mode=='hourly':original=original.merge(pd.read_csv(O/f'lear_weekly_{area}.csv',parse_dates=['time_utc'])[['time_utc','lear_weekly']],on='time_utc',validate='one_to_one')
  winter=pd.read_csv(O/f'winter_{mode}_{area}.csv',parse_dates=['time_utc']);winter['experiment']='winter_independently_tuned'
  for f in [original,winter]:
   month=f.time_utc.dt.tz_convert('Europe/Copenhagen').dt.month
   f['season']=month.map({12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'})
   for season,g in f.groupby('season'):
    for model in ['ridge','lightgbm','patchtst','chronos']+(['lear_weekly'] if mode=='hourly' else []):
     x=g.assign(delta=(g.previous_day-g.actual).abs()-(g[model]-g.actual).abs()).groupby('date')['delta'].mean().to_numpy();n=len(x)
     if n<14:continue
     fit=sm.OLS(x,np.ones(n)).fit(cov_type='HAC',cov_kwds={'maxlags':min(6,n-1),'use_correction':False});z=float(fit.tvalues[0]);rows.append(dict(mode=mode,area=area,experiment=g.experiment.iloc[0],season=season,model=model,days=n,mean_daily_mae_improvement=x.mean(),hac_lag=min(6,n-1),z=z,p_raw=2*norm.sf(abs(z))))
D=pd.DataFrame(rows);ix=D.p_raw.dropna().sort_values().index;n=len(ix);adjusted=[];v=0
for j,i in enumerate(ix):v=max(v,min(1,(n-j)*D.loc[i,'p_raw']));adjusted.append(v)
D['p_holm']=np.nan;D.loc[ix,'p_holm']=adjusted;D.to_csv(O/'seasonal_dm_tests.csv',index=False)
print('seasonal tests',len(D),'Holm family',n);print(D[D.model=='chronos'].to_string(index=False))
