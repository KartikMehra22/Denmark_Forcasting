"""Check exploratory LightGBM tuning chronology and summarize held-out predictions."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import json,numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';settings=json.loads((O/'tuned_lgb_settings.json').read_text());rows=[];tests=[]
assert len(settings)==4
for entry in settings:
 mode,area=entry['mode'],entry['area'];assert entry['train_end']<entry['validation_start']<=entry['validation_end'];p=pd.read_csv(O/f'tuned_lgb_{mode}_{area}.csv',parse_dates=['time_utc']);original=pd.read_csv(R/f'results/completion/{mode}_{area}_predictions.csv',parse_dates=['time_utc']);original=original[original.split=='later_locked'];assert len(p)==len(original) and not p.time_utc.duplicated().any();p=p.sort_values('time_utc');original=original.sort_values('time_utc');assert np.array_equal(p.time_utc,original.time_utc);np.testing.assert_allclose(p.actual,original.actual);np.testing.assert_allclose(p.lightgbm,original.lightgbm)
 assert p.date.min()=='2026-10-25' and p.date.max()=='2026-11-16' and p.date.nunique()==23 and entry['validation_end']<p.date.min()
 for origin in p.fit_origin.unique():assert (O/'tuned_lightgbm_models'/f'{mode}_{area}_{origin}.json.gz').exists()
 for model in ['previous_day','lightgbm','lightgbm_tuned']:
  e=p[model]-p.actual;rows.append(dict(mode=mode,area=area,model=model,days=p.date.nunique(),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean()),trees=entry['chosen']['trees'],leaves=entry['chosen']['leaves']))
 for reference in ['previous_day','lightgbm']:
  x=p.assign(delta=(p[reference]-p.actual).abs()-(p.lightgbm_tuned-p.actual).abs()).groupby('date').delta.mean().to_numpy();m=sm.OLS(x,np.ones(len(x))).fit(cov_type='HAC',cov_kwds={'maxlags':6,'use_correction':False});z=float(m.tvalues[0]);tests.append(dict(mode=mode,area=area,reference=reference,days=len(x),mean_daily_mae_improvement=x.mean(),z=z,p_raw=2*norm.sf(abs(z))))
pd.DataFrame(rows).to_csv(O/'tuned_lgb_metrics.csv',index=False);t=pd.DataFrame(tests);t['p_holm_exploratory_family']=multipletests(t.p_raw,method='holm')[1];t.to_csv(O/'tuned_lgb_tests.csv',index=False);print('Exploratory candidate tuning occurred before held-out test dates; analysis was designed after seeing original results.');print(pd.DataFrame(rows).to_string(index=False));print(t.to_string(index=False))
