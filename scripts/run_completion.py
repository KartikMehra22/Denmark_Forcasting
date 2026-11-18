"""Run the locked completion experiment and checks without altering the protocol."""
from pathlib import Path
import subprocess,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
protocol=ROOT/'completion_protocol.json';before=hashlib.sha256(protocol.read_bytes()).hexdigest()
for script in ['fetch_later.py','rolling_completion.py','completion_summary.py','verify_completion.py']:
 subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
assert hashlib.sha256(protocol.read_bytes()).hexdigest()==before
(ROOT/'results/completion/SUCCESS.json').write_text(json.dumps({'status':'complete','protocol_sha256':before,'checks':'verify_completion.py passed'},indent=2))
