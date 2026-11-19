"""Reload the tuned LightGBM checkpoints and verify all saved later predictions."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import json
import sys
import subprocess
import numpy as np
import pandas as pd
from controlled_study import samples
from rolling_completion import extended

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response"
arrays = {}
jobs = []
for mode in ["hourly", "quarter_hour"]:
    for area in ["DK1", "DK2"]:
        by = {r[0]: r for r in samples(extended(area, mode), mode)}
        p = pd.read_csv(O / f"tuned_lgb_{mode}_{area}.csv")
        for origin, g in p.groupby("fit_origin", sort=True):
            days = sorted(pd.to_datetime(g.date).dt.date.unique())
            x = np.stack([by[d][2] for d in days])
            h = 25 if mode == "hourly" else 100
            expected = np.zeros((len(days), h))
            counts = []
            for j, d in enumerate(days):
                day = g[g.date == str(d)].sort_values("time_utc")
                n = len(day)
                counts.append(n)
                expected[j, :n] = day.lightgbm_tuned.to_numpy()
            i = len(jobs)
            arrays[f"x{i}"] = x
            arrays[f"expected{i}"] = expected
            arrays[f"counts{i}"] = np.array(counts)
            jobs.append(
                {
                    "area": area,
                    "mode": mode,
                    "fit_origin": origin,
                    "model": str(
                        O / "tuned_lightgbm_models" / f"{mode}_{area}_{origin}.json.gz"
                    ),
                }
            )
npz = O / "tuned_verify_worker.npz"
manifest = O / "tuned_verify_worker.json"
output = O / "tuned_verify_worker_results.json"
np.savez(npz, **arrays)
manifest.write_text(json.dumps(jobs))
subprocess.run(
    [
        sys.executable,
        str(R / "scripts/followup/supervisor_spring_checkpoint_worker.py"),
        str(npz),
        str(manifest),
        str(output),
    ],
    check=True,
)
records = json.loads(output.read_text())
assert len(records) == len(jobs)
pd.DataFrame(records).to_csv(O / "tuned_lgb_checkpoint_verification.csv", index=False)
print("PASS all", len(records), "tuned LightGBM checkpoints")
