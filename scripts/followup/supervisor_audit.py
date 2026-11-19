"""Produce exact period, refit and calibration tables from preserved experiment artifacts."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import hashlib,json,sys
import numpy as np,pandas as pd
from rolling_completion import extended,lear_rows
from controlled_study import samples
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';O.mkdir(parents=True,exist_ok=True)
logs=json.loads((R/'results/completion/training_log.json').read_text());cfg=json.loads((R/'results/controlled/configuration.json').read_text())
period=[];windows=[];cal=[];timeline=[];sources=[]
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
  key=f'{mode}_{area}';s=extended(area,mode);items=samples(s,mode);lookup={r[0]:r for r in items};old=cfg[key]
  p=pd.read_csv(R/f'results/completion/{key}_predictions.csv',parse_dates=['time_utc']);l=[x for x in logs if x['mode']==mode and x['area']==area]
  val=pd.Timestamp(old['validation_start']).date();test=pd.Timestamp(old['test_start']).date()
  def selected(d):return [r for r in items if d-timedelta(days=730)<=r[0]<d-timedelta(days=2)]
  initial=selected(val);final=selected(test)
  valid=[r for r in items if val<=r[0]<test-timedelta(days=2)]
  for row in l:
   d=pd.Timestamp(row['origin']).date();train=selected(d);assert len(train)==row['training_days']
   windows.append(dict(mode=mode,area=area,refit_origin=d,first_training_delivery=train[0][0],last_training_delivery=train[-1][0],training_days=len(train),latest_label_utc=row['latest_label_utc'],issue_utc=row['issue_utc']))
  for split in ['historical','later_locked']:
   g=p[p.split==split]
   for model in ['previous_day','previous_week','ridge','lightgbm','patchtst','chronos']+(['lear'] if mode=='hourly' else []):
    e=g[model]-g.actual;sources.append(dict(mode=mode,area=area,split=split,model=model,first=g.date.min(),last=g.date.max(),n=len(g),days=g.date.nunique(),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean())))
  period.append(dict(mode=mode,area=area,raw_first_utc=s.index.min(),raw_last_utc=s.index.max(),initial_train_first=initial[0][0],initial_train_last=initial[-1][0],initial_train_n=len(initial),selection_start=valid[0][0],selection_end=valid[-1][0],selection_n=len(valid),final_train_first=final[0][0],final_train_last=final[-1][0],final_train_n=len(final),earlier_start=p[p.split=='historical'].date.min(),earlier_end=p[p.split=='historical'].date.max(),later_start=p[p.split=='later_locked'].date.min(),later_end=p[p.split=='later_locked'].date.max(),ridge_alpha=old['ridge_alpha'],patchtst_epochs=old['patchtst_epochs'],lightgbm_trees='80 fixed, not tuned',context=168 if mode=='hourly' else 672))
  # Exact dates of forecast errors used by the implementation; the fixed pretrained model is unchanged.
  for day in [test,pd.Timestamp('2026-10-25').date()]:
   first=day-timedelta(days=61);last=day-timedelta(days=2)
   cal.append(dict(mode=mode,area=area,forecast_day=day,calibration_first=first,calibration_last=last,days=60,selection_start=valid[0][0],selection_end=valid[-1][0],overlap_days=max(0,(min(last,valid[-1][0])-max(first,valid[0][0])).days+1)))
  for ds in ['2026-03-29','2026-10-25','2026-10-26']:
   d=pd.Timestamp(ds).date()
   if d in lookup:
    r=lookup[d];end=r[1];step=pd.Timedelta(minutes=60 if mode=='hourly' else 15);last=end-step
    issue=pd.Timestamp(str(d-timedelta(days=1))+' 10:00',tz='Europe/Copenhagen').tz_convert('UTC')
    timeline.append(dict(mode=mode,area=area,delivery_day=d,issue_utc=issue,last_input_utc=last,last_input_local=last.tz_convert('Europe/Copenhagen'),target_steps=r[4],input_observations=len(r[2]),last_input_is_future_delivery_at_issue=bool(last>issue)))
for name,data in [('periods',period),('training_windows',windows),('calibration_windows',cal),('cutoff_examples',timeline),('earlier_later_metrics',sources)]:pd.DataFrame(data).to_csv(O/f'{name}.csv',index=False)
print('PERIODS');print(pd.DataFrame(period).to_string(index=False));print('CALIBRATION');print(pd.DataFrame(cal).to_string(index=False));print('TRAINING RANGES');w=pd.DataFrame(windows);print(w.groupby(['mode','area']).training_days.agg(['min','max']).to_string())
