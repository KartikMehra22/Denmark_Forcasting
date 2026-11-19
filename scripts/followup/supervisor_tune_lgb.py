"""Post-hoc LightGBM sensitivity analysis: select only using pre-test validation."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import json,subprocess,sys
import numpy as np,pandas as pd
from rolling_completion import extended
from controlled_study import samples
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';D=O/'tuned_lightgbm_models';D.mkdir(exist_ok=True)
base=json.loads((R/'results/controlled/configuration.json').read_text())
grid=[{'trees':80,'leaves':7},{'trees':160,'leaves':7},{'trees':160,'leaves':15}]
summary=[]
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
  cfg=base[f'{mode}_{area}'];val=pd.Timestamp(cfg['validation_start']).date();test=pd.Timestamp(cfg['test_start']).date();series=extended(area,mode);items=samples(series,mode);by={r[0]:r for r in items}
  train=[r for r in items if val-timedelta(days=730)<=r[0]<val-timedelta(days=2)];valid=[r for r in items if val<=r[0]<test-timedelta(days=2)]
  X=np.stack([r[2] for r in train]);Y=np.stack([r[3] for r in train]);V=np.stack([r[2] for r in valid]);actual=[r[3][:r[4]] for r in valid];inp=O/f'tune_{mode}_{area}_worker.npz';out=O/f'tune_{mode}_{area}_worker.npy';np.savez(inp,x=X,y=Y,ex=V)
  scored=[]
  for candidate in grid:
   subprocess.run([sys.executable,str(R/'scripts/followup/supervisor_tune_lgb_worker.py'),str(inp),str(out),json.dumps(candidate)],check=True);pred=np.load(out);errors=np.concatenate([np.abs(pred[j,:len(y)]-y) for j,y in enumerate(actual)]);scored.append({'settings':candidate,'validation_mae':float(errors.mean())});print(mode,area,'validation',candidate,scored[-1]['validation_mae'],flush=True)
  choice=min(scored,key=lambda a:a['validation_mae'])['settings'];summary.append({'mode':mode,'area':area,'train_start':str(train[0][0]),'train_end':str(train[-1][0]),'validation_start':str(valid[0][0]),'validation_end':str(valid[-1][0]),'candidates':scored,'chosen':choice,'caveat':'Post-hoc exploratory analysis designed after inspecting original test results; do not claim prospective selection.'});(O/'tuned_lgb_settings.json').write_text(json.dumps(summary,indent=2))
  p=pd.read_csv(R/f'results/completion/{mode}_{area}_predictions.csv',parse_dates=['time_utc']);p=p[p.split=='later_locked'];frames=[]
  for origin,g in p.groupby('fit_origin',sort=True):
   d=pd.Timestamp(origin).date();tr=[r for r in items if d-timedelta(days=730)<=r[0]<d-timedelta(days=2)];days=sorted(pd.to_datetime(g.date).dt.date.unique());ex=np.stack([by[k][2] for k in days]);np.savez(inp,x=np.stack([r[2] for r in tr]),y=np.stack([r[3] for r in tr]),ex=ex)
   checkpoint=D/f'{mode}_{area}_{origin}.json.gz';subprocess.run([sys.executable,str(R/'scripts/followup/supervisor_tune_lgb_worker.py'),str(inp),str(out),json.dumps(choice),str(checkpoint)],check=True);pred=np.load(out)
   for j,day in enumerate(days):
    f=g[g.date==str(day)].sort_values('time_utc').copy();f['lightgbm_tuned']=pred[j,:len(f)];frames.append(f[['mode','area','date','time_utc','actual','previous_day','lightgbm','lightgbm_tuned','fit_origin']])
   pd.concat(frames).to_csv(O/f'tuned_lgb_{mode}_{area}.csv',index=False);print(mode,area,'test refit',origin,flush=True)
