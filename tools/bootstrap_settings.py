"""Install code-only settings on a fresh checkout without overwriting results."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
source=root/'config/controlled_settings.json'
target=root/'results/controlled/configuration.json'
if target.exists():
    raise SystemExit('Existing experiment configuration preserved; no file overwritten.')
settings=json.loads(source.read_text())
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(settings,indent=2))
print(target)
