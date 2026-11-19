"""Rebuild winter model checkpoints and check against stored winter forecasts."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import json,sys,subprocess,joblib,gzip
import numpy as np,pandas as pd,torch
from controlled_study import samples,fitpatch,predictpatch
from rolling_completion import extended
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';D=O/'reconstructed_winter';D.mkdir(exist_ok=True);records=[]
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
  s=extended(area,mode);items=samples(s,mode);by={r[0]:r for r in items}
  p=pd.read_csv(O/f'winter_{mode}_{area}.csv',parse_dates=['time_utc']);cfg=json.loads((O/f'winter_{mode}_{area}_settings.json').read_text());conf=json.loads((R/'results/controlled/configuration.json').read_text())[f'{mode}_{area}']['patchtst_config']
  for origin,g in p.groupby('fit_origin',sort=True):
   d=pd.Timestamp(origin).date();tr=[r for r in items if d-timedelta(days=730)<=r[0]<d-timedelta(days=2)];dates=sorted(pd.to_datetime(g.date).dt.date.unique());x=np.stack([r[2] for r in tr]);y=np.stack([r[3] for r in tr]);ex=np.stack([by[k][2] for k in dates]);ntrain=len(tr)
   temp=O/'winter_rebuild_worker.npz';predpath=O/'winter_rebuild_worker.npy';dest=D/f'{mode}_{area}_{d}_lgb.json.gz';np.savez(temp,x=x,y=y,ex=ex);subprocess.run([sys.executable,str(R/'scripts/followup/supervisor_lgb_worker.py'),str(temp),str(predpath),str(dest)],check=True);lp=np.load(predpath)
   net,_,_=fitpatch(x,y,cfg['patch_epochs'],42,conf);pp=predictpatch(net,ex);torch.save(net.state_dict(),D/f'{mode}_{area}_{d}_patchtst.pt')
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import Ridge
   ridge=make_pipeline(StandardScaler(),Ridge(alpha=cfg['alpha'],solver='svd'));ridge.fit(x,y);rp=ridge.predict(ex);joblib.dump(ridge,D/f'{mode}_{area}_{d}_ridge.joblib')
   diffs={m:0. for m in ['ridge','patchtst','lightgbm']}
   for j,k in enumerate(dates):
    z=g[g.date==str(k)].sort_values('time_utc');n=len(z)
    for m,a in [('ridge',rp),('patchtst',pp),('lightgbm',lp)]:diffs[m]=max(diffs[m],float(np.max(abs(z[m].values-a[j,:n]))))
   if max(diffs.values())>1e-4:raise ValueError(f'{mode} {area} {d}: {diffs}')
   records.append(dict(mode=mode,area=area,fit_origin=str(d),training_days=ntrain,**{f'{m}_max_abs_difference':v for m,v in diffs.items()}))
   pd.DataFrame(records).to_csv(O/'reconstructed_winter_verification.csv',index=False);print('winter checkpoint',mode,area,d,diffs,flush=True)
