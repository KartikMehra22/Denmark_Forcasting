"""Compare original forecasts with a complete rerun in an extracted clean directory."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];original=R/'results/completion';out=R/'results/supervisor_response/clean_run_verification.json'
parser=argparse.ArgumentParser();parser.add_argument('clean_project_root',type=Path);args=parser.parse_args();clean=args.clean_project_root/'results/completion';results=[]
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
  name=f'{mode}_{area}_predictions.csv';a=pd.read_csv(original/name);b=pd.read_csv(clean/name);assert len(a)==len(b) and list(a.columns)==list(b.columns)
  for col in ['date','time_utc','split','fit_origin','mode','area']:assert a[col].equals(b[col]),(name,col)
  diff={}
  for col in a.select_dtypes(include='number').columns:
   np.testing.assert_allclose(a[col],b[col],rtol=1e-6,atol=1e-4,err_msg=name+' '+col);diff[col]=float(np.max(np.abs(a[col]-b[col])))
  results.append({'mode':mode,'area':area,'rows':len(a),'original_sha256':hashlib.sha256((original/name).read_bytes()).hexdigest(),'clean_sha256':hashlib.sha256((clean/name).read_bytes()).hexdigest(),'max_absolute_difference_by_numeric_column':diff})
verification=json.loads((clean/'verification.json').read_text());assert verification['status']=='PASS'
out.write_text(json.dumps({'environment':'fresh Python 3.12 virtual environment; pinned requirements-lock and supervisor requirements','rerun':'scripts/run_completion.py in isolated extracted copy','tolerances':{'absolute':1e-4,'relative':1e-6},'original_pipeline_verification':verification,'comparisons':results},indent=2));print(json.dumps(results,indent=2))
