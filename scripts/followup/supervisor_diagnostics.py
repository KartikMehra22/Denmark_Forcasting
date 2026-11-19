"""Inspect inputs, label range shift, early/later errors and Chronos lead times."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import json,numpy as np,pandas as pd
from controlled_study import samples
from rolling_completion import extended
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';logs=json.loads((R/'results/completion/training_log.json').read_text());details=[];leads=[]
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
  s=extended(area,mode);items=samples(s,mode);by={r[0]:r for r in items};p=pd.read_csv(R/f'results/completion/{mode}_{area}_predictions.csv',parse_dates=['time_utc'])
  for item in [x for x in logs if x['mode']==mode and x['area']==area]:
   d=pd.Timestamp(item['origin']).date();train=[r for r in items if d-timedelta(days=730)<=r[0]<d-timedelta(days=2)];x=np.stack([r[2] for r in train]);y=np.stack([r[3] for r in train]);g=p[p.fit_origin==str(d)].copy()
   for ds,z in g.groupby('date'):
    t=pd.Timestamp(ds).date();features=by[t][2];values=by[t][3][:by[t][4]];h=len(values)
    low=y[:,:h].min(axis=0);high=y[:,:h].max(axis=0)
    details.append(dict(mode=mode,area=area,date=ds,split=z.split.iloc[0],fit_origin=str(d),training_days=len(train),target_min=values.min(),target_max=values.max(),train_target_min=low.min(),train_target_max=high.max(),target_outside_per_lead=int(((values<low)|(values>high)).sum()),n=h,input_fraction_outside_feature_range=float(((features<x.min(axis=0))|(features>x.max(axis=0))).mean()),actual_mean=values.mean(),lear_mae=float((z.lear-z.actual).abs().mean()) if mode=='hourly' else np.nan,lgb_mae=(z.lightgbm-z.actual).abs().mean(),baseline_mae=(z.previous_day-z.actual).abs().mean()))
   if mode=='quarter_hour':
    g=g.sort_values('time_utc');g['lead']=g.groupby('date').cumcount()+1
    for h,z in g.groupby('lead'):leads.append(dict(area=area,split=z.split.iloc[0],lead=h,n=len(z),chronos_abs_error_sum=(z.chronos-z.actual).abs().sum(),prev_abs_error_sum=(z.previous_day-z.actual).abs().sum()))
D=pd.DataFrame(details);D.to_csv(O/'distribution_diagnostics_daily.csv',index=False)
print(D.groupby(['mode','area','split']).agg(days=('date','count'),outside_targets=('target_outside_per_lead','sum'),n=('n','sum'),feature_outside=('input_fraction_outside_feature_range','mean'),lgb_MAE=('lgb_mae','mean'),baseline=('baseline_mae','mean')).to_string())
