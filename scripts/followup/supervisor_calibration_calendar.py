"""Calendar of each forecast's historical error pool, including validation overlap."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import pandas as pd
import json

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response"
cfg = json.loads((R / "results/controlled/configuration.json").read_text())
out = []
for mode in ["hourly", "quarter_hour"]:
    for a in ["DK1", "DK2"]:
        p = pd.read_csv(R / f"results/completion/{mode}_{a}_predictions.csv")
        c = cfg[f"{mode}_{a}"]
        start = pd.Timestamp(c["validation_start"]).date()
        end = pd.Timestamp(c["test_start"]).date() - timedelta(days=3)
        for date, g in p.groupby("date", sort=True):
            d = pd.Timestamp(date).date()
            first = d - timedelta(days=61)
            last = d - timedelta(days=2)
            n = max(0, (min(last, end) - max(first, start)).days + 1)
            out.append(
                dict(
                    mode=mode,
                    area=a,
                    forecast_day=d,
                    first_error_day=first,
                    last_error_day=last,
                    error_days=60,
                    selection_overlap_days=n,
                    cal_expansion=g.cal_expansion.iloc[0],
                    split=g.split.iloc[0],
                )
            )
pd.DataFrame(out).to_csv(O / "all_calibration_windows.csv", index=False)
