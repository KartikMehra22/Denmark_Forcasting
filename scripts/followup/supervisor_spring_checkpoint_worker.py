"""Verify saved spring LightGBM boosters in isolation from PyTorch."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import sys,json,gzip
from pathlib import Path
import numpy as np,lightgbm as lgb
z=np.load(sys.argv[1]);jobs=json.loads(Path(sys.argv[2]).read_text());records=[]
for i,j in enumerate(jobs):
 x=z[f'x{i}'];expected=z[f'expected{i}'];counts=z[f'counts{i}'];
 with gzip.open(j['model'],'rt') as f:models=json.load(f)['models']
 pred=np.stack([lgb.Booster(model_str=m).predict(x,num_threads=1) for m in models],axis=1);delta=max(float(np.max(np.abs(pred[k,:n]-expected[k,:n]))) for k,n in enumerate(counts));assert delta<1e-4,(j,delta);records.append({**({'mode':j['mode']} if 'mode' in j else {}),'area':j['area'],'fit_origin':j['fit_origin'],'lightgbm_max_abs_difference':delta});print(j['area'],j['fit_origin'],delta,flush=True)
Path(sys.argv[3]).write_text(json.dumps(records,indent=2))
