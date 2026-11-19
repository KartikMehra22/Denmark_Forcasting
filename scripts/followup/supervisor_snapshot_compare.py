import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import json,pandas as pd,numpy as np
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';out=[]
for ds,timecol,price in [('Elspotprices','HourUTC','SpotPriceEUR'),('DayAheadPrices','TimeUTC','DayAheadPriceEUR')]:
 f=pd.DataFrame(json.loads((O/'api_snapshots'/f'{ds}.json').read_text())['records']);f[timecol]=pd.to_datetime(f[timecol],utc=True)
 for a in ['DK1','DK2']:
  name=f'elspot_{a}_history.csv' if ds=='Elspotprices' else f'dayahead_{a}.csv';old=pd.read_csv(R/'data'/name);old[timecol]=pd.to_datetime(old[timecol],utc=True)
  new=f[f.PriceArea==a];assert not old[timecol].duplicated().any() and not new[timecol].duplicated().any()
  x=old[[timecol,price]].merge(new[[timecol,price]],on=timecol,how='outer',suffixes=('_original','_fresh'),indicator=True)
  both=x[x._merge=='both'];diff=(both[f'{price}_original']-both[f'{price}_fresh']).abs()
  out.append(dict(dataset=ds,area=a,original_rows=len(old),fresh_rows=len(new),both=len(both),only_original=(x._merge=='left_only').sum(),only_fresh=(x._merge=='right_only').sum(),price_values_changed=(diff>1e-9).sum(),max_absolute_change=float(diff.max()),fresh_snapshot_is_original_raw_response=False))
O.joinpath('snapshot_comparison.csv').write_text(pd.DataFrame(out).to_csv(index=False));print(pd.DataFrame(out).to_string(index=False))
