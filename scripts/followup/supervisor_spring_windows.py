"""Verify and record each spring training window and forecast cutoff."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import pandas as pd
from rolling_completion import extended
from controlled_study import samples

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response"
rows = []
for area in ["DK1", "DK2"]:
    allrows = samples(extended(area, "quarter_hour"), "quarter_hour")
    p = pd.read_csv(O / f"spring_quarter_hour_{area}.csv")
    for origin, g in p.groupby("fit_origin", sort=True):
        d = pd.Timestamp(origin).date()
        tr = [
            r
            for r in allrows
            if d - timedelta(days=730) <= r[0] < d - timedelta(days=2)
        ]
        latest = max(r[1] + pd.Timedelta(minutes=15 * (len(r[3]) - 1)) for r in tr)
        issue = pd.Timestamp(
            str(d - timedelta(days=1)) + " 10:00", tz="Europe/Copenhagen"
        ).tz_convert("UTC")
        assert (
            latest < issue
            and g.training_days.nunique() == 1
            and g.training_days.iloc[0] == len(tr)
        )
        rows.append(
            dict(
                mode="quarter_hour",
                area=area,
                fit_origin=d,
                first_training_delivery=tr[0][0],
                last_training_delivery=tr[-1][0],
                training_days=len(tr),
                latest_label_utc=latest,
                issue_utc=issue,
                forecast_days=g.date.nunique(),
            )
        )
out = pd.DataFrame(rows)
out.to_csv(O / "spring_training_windows.csv", index=False)
print(out.groupby("area").training_days.agg(["min", "max", "count"]).to_string())
