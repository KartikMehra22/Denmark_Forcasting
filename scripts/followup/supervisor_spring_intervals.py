"""Chronos raw and sequential CQR intervals for the independently tuned spring test."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
from controlled_study import samples
from rolling_completion import extended
from chronos import ChronosBoltPipeline

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response"
torch.set_num_threads(2)
model = ChronosBoltPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",
    revision="a0e552de83495b5c28c14c71c374f3e33280b340",
    local_files_only=True,
    device_map="cpu",
    dtype=torch.float32,
)
qind = list(model.quantiles)
start = pd.Timestamp("2026-03-01").date()
end = pd.Timestamp("2026-06-01").date()
for area in ["DK1", "DK2"]:
    rows = samples(extended(area, "quarter_hour"), "quarter_hour")
    qrows = [r for r in rows if start - timedelta(days=62) <= r[0] < end]
    contexts = np.stack([r[2] for r in qrows])
    qs = []
    with torch.no_grad():
        for batch in torch.from_numpy(contexts).split(16):
            qs.append(model.predict(batch, prediction_length=100).numpy())
    qm = {r[0]: q for r, q in zip(qrows, np.concatenate(qs))}
    scores = {
        r[0]: np.maximum.reduce(
            [
                qm[r[0]][qind.index(0.1), : r[4]] - r[3][: r[4]],
                r[3][: r[4]] - qm[r[0]][qind.index(0.9), : r[4]],
                np.zeros(r[4]),
            ]
        )
        for r in qrows
    }
    f = pd.read_csv(O / f"spring_quarter_hour_{area}.csv", parse_dates=["time_utc"])
    out = []
    for r in qrows:
        d = r[0]
        if d < start:
            continue
        g = f[f.date == str(d)].copy()
        n = r[4]
        q = qm[d]
        assert len(g) == n
        np.testing.assert_allclose(
            g.chronos, q[qind.index(0.5), :n], atol=1e-4, rtol=1e-6
        )
        relevant = [
            v
            for date, v in scores.items()
            if d - timedelta(days=61) <= date < d - timedelta(days=1)
        ]
        assert len(relevant) == 60
        z = np.concatenate(relevant)
        rank = min(len(z), int(np.ceil((len(z) + 1) * 0.8)))
        exp = float(np.partition(z, rank - 1)[rank - 1])
        lo = q[qind.index(0.1), :n]
        hi = q[qind.index(0.9), :n]
        g = g[["mode", "area", "date", "time_utc", "actual", "chronos"]]
        g["raw_lower"] = lo
        g["raw_upper"] = hi
        g["cal_lower"] = lo - exp
        g["cal_upper"] = hi + exp
        g["cal_expansion"] = exp
        g["cal_first_error_day"] = str(d - timedelta(days=61))
        g["cal_last_error_day"] = str(d - timedelta(days=2))
        out.append(g)
    out = pd.concat(out)
    assert (out.cal_lower <= out.raw_lower).all() and (
        out.cal_upper >= out.raw_upper
    ).all()
    out.to_csv(O / f"spring_quarter_hour_{area}_intervals.csv", index=False)
    for typ in ["raw", "cal"]:
        print(
            area,
            typ,
            "coverage",
            (
                (out.actual >= out[f"{typ}_lower"])
                & (out.actual <= out[f"{typ}_upper"])
            ).mean(),
            "mean_width",
            (out[f"{typ}_upper"] - out[f"{typ}_lower"]).mean(),
            flush=True,
        )
# Save interval-level coverage and disclose overlap with spring setting-selection dates.
import json

summary = []
for area in ["DK1", "DK2"]:
    data = pd.read_csv(O / f"spring_quarter_hour_{area}_intervals.csv")
    cfg = json.loads((O / f"spring_quarter_hour_{area}_settings.json").read_text())
    first = data.iloc[0]
    v1 = pd.Timestamp(cfg["validation_start"])
    v2 = pd.Timestamp(cfg["validation_end"])
    c1 = pd.Timestamp(first.cal_first_error_day)
    c2 = pd.Timestamp(first.cal_last_error_day)
    overlap = max(0, (min(c2, v2) - max(c1, v1)).days + 1)
    for name in ["raw", "cal"]:
        summary.append(
            dict(
                area=area,
                days=data.date.nunique(),
                interval=name,
                coverage=(
                    (data.actual >= data[f"{name}_lower"])
                    & (data.actual <= data[f"{name}_upper"])
                ).mean(),
                mean_width=(data[f"{name}_upper"] - data[f"{name}_lower"]).mean(),
                first_calibration_start=str(c1.date()),
                first_calibration_end=str(c2.date()),
                first_calibration_overlap_selection_days=overlap,
            )
        )
pd.DataFrame(summary).to_csv(O / "spring_interval_summary.csv", index=False)
