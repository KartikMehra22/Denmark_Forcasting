"""Reconstruct historical LGB checkpoints and verify every saved delivery prediction."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import json
import sys
import subprocess
import numpy as np
import pandas as pd
from controlled_study import samples
from rolling_completion import extended

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response"
D = O / "reconstructed_lightgbm"
D.mkdir(exist_ok=True)
logs = json.loads((R / "results/completion/training_log.json").read_text())
verified = []
for mode in ["hourly", "quarter_hour"]:
    for area in ["DK1", "DK2"]:
        s = extended(area, mode)
        items = samples(s, mode)
        by = {r[0]: r for r in items}
        p = pd.read_csv(
            R / f"results/completion/{mode}_{area}_predictions.csv",
            parse_dates=["time_utc"],
        )
        steps = 1 if mode == "hourly" else 4
        for item in [r for r in logs if r["mode"] == mode and r["area"] == area]:
            d = pd.Timestamp(item["origin"]).date()
            train = [
                r
                for r in items
                if d - timedelta(days=730) <= r[0] < d - timedelta(days=2)
            ]
            g = p[p.fit_origin == str(d)].copy()
            dates = sorted(pd.to_datetime(g.date).dt.date.unique())
            x = np.stack([r[2] for r in train])
            y = np.stack([r[3] for r in train])
            ex = np.stack([by[k][2] for k in dates])
            temp = O / "lgb_rebuild_worker.npz"
            predpath = O / "lgb_rebuild_worker.npy"
            dest = D / f"{mode}_{area}_{d}.json.gz"
            np.savez(temp, x=x, y=y, ex=ex)
            subprocess.run(
                [
                    sys.executable,
                    str(R / "scripts/followup/supervisor_lgb_worker.py"),
                    str(temp),
                    str(predpath),
                    str(dest),
                ],
                check=True,
            )
            v = np.load(predpath)
            maxdiff = 0.0
            for j, k in enumerate(dates):
                z = g[g.date == str(k)].sort_values("time_utc")
                n = len(z)
                maxdiff = max(
                    maxdiff, float(np.max(abs(z.lightgbm.to_numpy() - v[j, :n])))
                )
            if maxdiff > 1e-5:
                raise ValueError(
                    f"{mode} {area} {d}: checkpoint prediction mismatch {maxdiff}"
                )
            verified.append(
                dict(
                    mode=mode,
                    area=area,
                    fit_origin=str(d),
                    max_abs_prediction_difference=maxdiff,
                    model_file=dest.name,
                    horizons=v.shape[1],
                )
            )
            pd.DataFrame(verified).to_csv(
                O / "reconstructed_lightgbm_verification.csv", index=False
            )
            print("LGB", mode, area, d, "maxdiff", maxdiff, flush=True)
