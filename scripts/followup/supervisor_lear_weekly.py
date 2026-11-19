"""Weekly LEAR with exactly the shared weekly training date window."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import json,warnings
import numpy as np,pandas as pd
from sklearn.linear_model import Lasso,LassoLarsIC
from sklearn.exceptions import ConvergenceWarning
from rolling_completion import extended,lear_rows,invariant
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';O.mkdir(parents=True,exist_ok=True)
warnings.filterwarnings('ignore',category=ConvergenceWarning)
for area in ['DK1','DK2']:
 s=extended(area,'hourly');rows=lear_rows(s)
 logs=[r for r in json.loads((R/'results/completion/training_log.json').read_text()) if r['mode']=='hourly' and r['area']==area]
 prev=pd.read_csv(R/f'results/completion/hourly_{area}_predictions.csv',parse_dates=['time_utc']);output=[]
 for item in logs:
  d=pd.Timestamp(item['origin']).date();days=[k for k in rows if d-timedelta(days=730)<=k<d-timedelta(days=2)]
  x=np.stack([rows[k][0] for k in days]);y=np.stack([rows[k][1] for k in days]);xs,xm,xd=invariant(x[:,:-7]);ys,ym,yd=invariant(y);xx=np.c_[xs,x[:,-7:]]
  models=[]
  for h in range(24):
   a=LassoLarsIC(criterion='aic',max_iter=2500).fit(xx,ys[:,h]).alpha_
   models.append(Lasso(alpha=max(a,1e-10),max_iter=2500).fit(xx,ys[:,h]))
  for k in range(7):
   day=d+timedelta(days=k);g=prev[prev.date==str(day)].copy()
   if g.empty:continue
   v=rows[day][0];xx_test=np.r_[np.arcsinh((v[:-7]-xm)/xd),v[-7:]][None,:]
   curve=np.sinh(np.array([m.predict(xx_test)[0] for m in models]))*yd+ym
   g['lear_weekly']=curve[g.time_utc.dt.tz_convert('Europe/Copenhagen').dt.hour];output.append(g[['area','date','time_utc','actual','previous_day','lear','lear_weekly','split','fit_origin']])
  pd.concat(output).to_csv(O/f'lear_weekly_{area}.csv',index=False)
  print(area,d,flush=True)
