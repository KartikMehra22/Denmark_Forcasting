"""Reload every spring checkpoint and compare its predictions with saved forecasts."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import json
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
import joblib
from transformers import PatchTSTConfig, PatchTSTForPrediction
from controlled_study import samples, predictpatch
from rolling_completion import extended

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response"
cfg = json.loads((R / "results/controlled/configuration.json").read_text())
arrays = {}
jobs = []
records = []
for area in ["DK1", "DK2"]:
    by = {r[0]: r for r in samples(extended(area, "quarter_hour"), "quarter_hour")}
    p = pd.read_csv(O / f"spring_quarter_hour_{area}.csv")
    for origin, g in p.groupby("fit_origin", sort=True):
        days = sorted(pd.to_datetime(g.date).dt.date.unique())
        x = np.stack([by[d][2] for d in days])
        folder = O / "spring_models" / f"quarter_hour_{area}_{origin}"
        ridge = joblib.load(folder / "ridge.joblib").predict(x)
        net = PatchTSTForPrediction(
            PatchTSTConfig(**cfg[f"quarter_hour_{area}"]["patchtst_config"])
        )
        net.load_state_dict(
            torch.load(folder / "patchtst.pt", map_location="cpu", weights_only=True)
        )
        patch = predictpatch(net, x)
        expected = np.zeros((len(days), 100))
        counts = []
        rerr = perr = 0.0
        for j, d in enumerate(days):
            day = g[g.date == str(d)].sort_values("time_utc")
            n = len(day)
            counts.append(n)
            expected[j, :n] = day.lightgbm.to_numpy()
            rerr = max(rerr, float(np.max(np.abs(ridge[j, :n] - day.ridge))))
            perr = max(perr, float(np.max(np.abs(patch[j, :n] - day.patchtst))))
        assert max(rerr, perr) < 1e-4, (area, origin, rerr, perr)
        i = len(jobs)
        arrays[f"x{i}"] = x
        arrays[f"expected{i}"] = expected
        arrays[f"counts{i}"] = np.array(counts)
        jobs.append(
            {
                "area": area,
                "fit_origin": origin,
                "model": str(folder / "lightgbm.json.gz"),
            }
        )
        records.append(
            {
                "area": area,
                "fit_origin": origin,
                "ridge_max_abs_difference": rerr,
                "patchtst_max_abs_difference": perr,
            }
        )
        print("ridge and patch", area, origin, flush=True)
npz = O / "spring_verify_worker.npz"
manifest = O / "spring_verify_worker.json"
output = O / "spring_verify_worker_results.json"
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
a = pd.DataFrame(records)
b = pd.DataFrame(json.loads(output.read_text()))
result = a.merge(b, on=["area", "fit_origin"], validate="one_to_one")
assert len(result) == 28
result.to_csv(O / "spring_checkpoint_verification.csv", index=False)
print("PASS all", len(result), "spring refit checkpoints")
