"""Reconstruct LEAR scalers and coefficients and verify against saved forecasts."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import warnings,json,numpy as np,pandas as pd
from sklearn.linear_model import Lasso,LassoLarsIC
from sklearn.exceptions import ConvergenceWarning
from rolling_completion import extended,lear_rows,invariant
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';D=O/'reconstructed_lear';D.mkdir(exist_ok=True)
warnings.filterwarnings('ignore',category=ConvergenceWarning);records=[]
for area in ['DK1','DK2']:
 s=extended(area,'hourly');rows=lear_rows(s)
 orig=pd.read_csv(R/f'results/completion/hourly_{area}_predictions.csv',parse_dates=['time_utc']);week=pd.read_csv(O/f'lear_weekly_{area}.csv',parse_dates=['time_utc']);winter=pd.read_csv(O/f'winter_hourly_{area}.csv',parse_dates=['time_utc'])
 jobs=[('daily',orig,'date','lear',1),('weekly',week,'fit_origin','lear_weekly',2),('winter_weekly',winter,'fit_origin','lear_weekly',2)]
 for typ,frame,col,model,gap in jobs:
  for key,g in frame.groupby(col,sort=True):
   d=pd.Timestamp(key).date();days=[k for k in rows if d-timedelta(days=730)<=k<d-timedelta(days=gap)]
   x=np.stack([rows[k][0] for k in days]);y=np.stack([rows[k][1] for k in days]);xs,xm,xd=invariant(x[:,:-7]);ys,ym,yd=invariant(y);xx=np.c_[xs,x[:,-7:]]
   coef=[];intercept=[];alphas=[]
   for h in range(24):
    alpha=LassoLarsIC(criterion='aic',max_iter=2500).fit(xx,ys[:,h]).alpha_;m=Lasso(alpha=max(alpha,1e-10),max_iter=2500).fit(xx,ys[:,h]);coef.append(m.coef_);intercept.append(m.intercept_);alphas.append(alpha)
   coef=np.stack(coef);intercept=np.array(intercept);maxdiff=0.
   for day,grp in g.groupby('date'):
    k=pd.Timestamp(day).date();v=rows[k][0];test=np.r_[np.arcsinh((v[:-7]-xm)/xd),v[-7:]];curve=np.sinh(coef@test+intercept)*yd+ym;v=curve[grp.time_utc.dt.tz_convert('Europe/Copenhagen').dt.hour];maxdiff=max(maxdiff,float(np.max(abs(v-grp[model]))))
   if maxdiff>1e-5:raise ValueError(f'{typ} {area} {d}: {maxdiff}')
   path=D/f'{typ}_{area}_{d}.npz';np.savez_compressed(path,coef=coef,intercept=intercept,alpha=alphas,x_median=xm,x_scale=xd,y_median=ym,y_scale=yd)
   records.append(dict(kind=typ,area=area,fit_origin=str(d),training_days=len(days),max_abs_prediction_difference=maxdiff,file=path.name))
   if len(records)%10==0:pd.DataFrame(records).to_csv(O/'reconstructed_lear_verification.csv',index=False);print('LEAR',typ,area,d,'checkpoints',len(records),flush=True)
pd.DataFrame(records).to_csv(O/'reconstructed_lear_verification.csv',index=False)
