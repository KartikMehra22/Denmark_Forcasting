from pathlib import Path
import requests,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/later_locked';D.mkdir(exist_ok=True)
p=json.loads((ROOT/'completion_protocol.json').read_text());manifest={}
for a in ['DK1','DK2']:
 params={'start':p['later_start'],'end':p['later_end_exclusive'],'filter':json.dumps({'PriceArea':[a]}),'limit':10000,'sort':'TimeUTC asc'}
 path=D/f'{a}.json'
 if path.exists():
  raw=json.loads(path.read_text());url=requests.Request('GET','https://api.energidataservice.dk/dataset/DayAheadPrices',params=params).prepare().url
 else:
  for attempt in range(4):
   r=requests.get('https://api.energidataservice.dk/dataset/DayAheadPrices',params=params,timeout=35)
   if r.status_code!=429:break
   delay=max(60,int(r.headers.get('Retry-After','60')));print('Rate limit; waiting',delay,'seconds',flush=True);time.sleep(delay)
  r.raise_for_status();raw=r.json();url=r.url
 rows=raw.get('records',[])
 if not rows:raise RuntimeError('No later data returned')
 path=D/f'{a}.json';path.write_text(json.dumps(raw))
 manifest[a]={'url':url,'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 print(a,len(rows),flush=True);time.sleep(2)
(D/'manifest.json').write_text(json.dumps(manifest,indent=2))
