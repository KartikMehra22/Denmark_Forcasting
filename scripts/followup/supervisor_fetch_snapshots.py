"""Capture current official API responses, independently of original CSV snapshots."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import hashlib,json,time,requests
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response/api_snapshots';O.mkdir(parents=True,exist_ok=True)
job=[('Elspotprices','2022-01-01','2025-10-01'),('DayAheadPrices','2025-10-01','2026-10-25')]
manifest={}
for name,start,end in job:
 p={'start':start,'end':end,'filter':json.dumps({'PriceArea':['DK1','DK2']}),'limit':0,'sort':'TimeUTC asc' if name=='DayAheadPrices' else 'HourUTC asc'}
 for attempt in range(6):
  r=requests.get(f'https://api.energidataservice.dk/dataset/{name}',params=p,timeout=90)
  if r.status_code!=429:break
  t=max(60,int(r.headers.get('Retry-After','60')));print('rate limited',t,flush=True);time.sleep(t)
 r.raise_for_status();d=r.json();rows=d.get('records',[])
 if not rows:raise RuntimeError(f'{name}: empty response')
 path=O/f'{name}.json';path.write_bytes(r.content)
 manifest[name]=dict(retrieved_at_utc=datetime.now(timezone.utc).isoformat(),start_danish_local=start,end_exclusive_danish_local=end,rows=len(rows),sha256=hashlib.sha256(r.content).hexdigest(),url=r.url,note='Fresh API response, not the original historical download.')
 print(name,len(rows),flush=True);time.sleep(3)
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
