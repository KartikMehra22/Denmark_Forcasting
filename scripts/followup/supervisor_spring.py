"""Retrospective quarter-hour spring comparison; tune in February before test."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import json
import subprocess
import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from controlled_study import samples, fitpatch, predictpatch
from rolling_completion import extended, lear_rows, invariant
from chronos import ChronosBoltPipeline
from sklearn.linear_model import Lasso, LassoLarsIC

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "results/supervisor_response"
O.mkdir(exist_ok=True)
torch.set_num_threads(2)
bolt = ChronosBoltPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",
    revision="a0e552de83495b5c28c14c71c374f3e33280b340",
    local_files_only=True,
    device_map="cpu",
    dtype=torch.float32,
)
VALID = pd.Timestamp("2026-02-01").date()
START = pd.Timestamp("2026-03-01").date()
END = pd.Timestamp("2026-06-01").date()
for mode in ["quarter_hour"]:
    for area in ["DK1", "DK2"]:
        s = extended(area, mode)
        items = samples(s, mode)
        by = {x[0]: x for x in items}
        conf = json.loads((ROOT / "results/controlled/configuration.json").read_text())[
            f"{mode}_{area}"
        ]["patchtst_config"]

        def trainrows(d):
            return [
                r
                for r in items
                if d - timedelta(days=730) <= r[0] < d - timedelta(days=2)
            ]

        def arr(rows):
            return np.stack([r[2] for r in rows]), np.stack([r[3] for r in rows])

        tr = trainrows(VALID)
        va = [r for r in items if VALID <= r[0] < START - timedelta(days=2)]
        x, y = arr(tr)
        vx, vy = arr(va)
        scores = {}
        for alpha in [1, 100, 10000]:
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
            m.fit(x, y)
            scores[alpha] = float(np.mean(abs(m.predict(vx) - vy)))
        alpha = min(scores, key=scores.get)
        _, epochs, history = fitpatch(x, y, 12, 42, conf, (vx, vy))
        info = {
            "validation_training_start": str(tr[0][0]),
            "validation_training_end": str(tr[-1][0]),
            "validation_training_days": len(tr),
            "validation_start": str(va[0][0]),
            "validation_end": str(va[-1][0]),
            "alpha": alpha,
            "validation_mae": scores,
            "patch_epochs": epochs,
            "patch_validation_mae": history,
            "lightgbm_trees": 80,
            "lightgbm_trees_selected": False,
        }
        (O / f"spring_{mode}_{area}_settings.json").write_text(
            json.dumps(info, indent=2)
        )
        dates = [d for d in by if START <= d < END]
        contexts = np.stack([by[d][2] for d in dates])
        qs = []
        with torch.no_grad():
            for b in torch.from_numpy(contexts).split(16):
                qs.append(
                    bolt.predict(b, prediction_length=conf["prediction_length"]).numpy()
                )
        q = np.concatenate(qs)
        qi = list(bolt.quantiles).index(0.5)
        records = []
        lr = lear_rows(s) if mode == "hourly" else None
        for off in range(0, len(dates), 7):
            block = dates[off : off + 7]
            d = block[0]
            train = trainrows(d)
            tx, ty = arr(train)
            ex = np.stack([by[k][2] for k in block])
            issue = pd.Timestamp(
                str(d - timedelta(days=1)) + " 10:00", tz="Europe/Copenhagen"
            ).tz_convert("UTC")
            step = pd.Timedelta(minutes=60 if mode == "hourly" else 15)
            assert max(t[1] + (len(t[3]) - 1) * step for t in train) < issue
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
            ridge.fit(tx, ty)
            rp = ridge.predict(ex)
            tmp = O / f"spring_{mode}_{area}_worker.npz"
            dest = O / f"spring_{mode}_{area}_worker.npy"
            np.savez(tmp, x=tx, y=ty, ex=ex)
            modeldir = O / "spring_models" / f"{mode}_{area}_{d}"
            modeldir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/followup/supervisor_lgb_worker.py"),
                    str(tmp),
                    str(dest),
                    str(modeldir / "lightgbm.json.gz"),
                ],
                check=True,
            )
            lp = np.load(dest)
            net, _, _ = fitpatch(tx, ty, epochs, 42, conf)
            pp = predictpatch(net, ex)
            import joblib

            joblib.dump(ridge, modeldir / "ridge.joblib")
            torch.save(net.state_dict(), modeldir / "patchtst.pt")
            # LEAR receives same fit origin and training date cutoff as the other models.
            models = None
            if lr is not None:
                days = [
                    k
                    for k in lr
                    if d - timedelta(days=730) <= k < d - timedelta(days=2)
                ]
                xx = np.stack([lr[k][0] for k in days])
                yy = np.stack([lr[k][1] for k in days])
                xxs, xm, xd = invariant(xx[:, :-7])
                yys, ym, yd = invariant(yy)
                xxs = np.c_[xxs, xx[:, -7:]]
                models = []
                for h in range(24):
                    a = (
                        LassoLarsIC(criterion="aic", max_iter=2500)
                        .fit(xxs, yys[:, h])
                        .alpha_
                    )
                    models.append(
                        Lasso(alpha=max(a, 1e-10), max_iter=2500).fit(xxs, yys[:, h])
                    )
            for j, k in enumerate(block):
                row = by[k]
                n = row[4]
                idx = pd.date_range(row[1], periods=n, freq=step)
                loc = idx.tz_convert("Europe/Copenhagen")
                steps = 1 if mode == "hourly" else 4

                def lag(days):
                    past = s[
                        s.index.tz_convert("Europe/Copenhagen").date
                        == k - timedelta(days=days)
                    ]
                    local = past.index.tz_convert("Europe/Copenhagen")
                    slots = local.hour * steps + local.minute // (60 // steps)
                    v = (
                        pd.Series(past.values, index=slots)
                        .groupby(level=0)
                        .mean()
                        .reindex(range(24 * steps))
                        .interpolate(limit_direction="both")
                    )
                    return v.reindex(
                        loc.hour * steps + loc.minute // (60 // steps)
                    ).values

                f = pd.DataFrame(
                    dict(
                        mode=mode,
                        area=area,
                        date=str(k),
                        time_utc=idx,
                        actual=row[3][:n],
                        previous_day=lag(1),
                        previous_week=lag(7),
                        ridge=rp[j, :n],
                        lightgbm=lp[j, :n],
                        patchtst=pp[j, :n],
                        chronos=q[off + j, qi, :n],
                        fit_origin=str(d),
                        training_days=len(train),
                    )
                )
                if models is not None:
                    v = lr[k][0]
                    v = np.r_[np.arcsinh((v[:-7] - xm) / xd), v[-7:]][None, :]
                    pred = (
                        np.sinh(np.array([m.predict(v)[0] for m in models])) * yd + ym
                    )
                    f["lear_weekly"] = pred[loc.hour]
                records.append(f)
            pd.concat(records).to_csv(O / f"spring_{mode}_{area}.csv", index=False)
            print("spring", mode, area, d, flush=True)
