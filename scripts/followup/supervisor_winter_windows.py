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
allrows = []
for mode in ["hourly", "quarter_hour"]:
    for area in ["DK1", "DK2"]:
        items = samples(extended(area, mode), mode)
        p = pd.read_csv(O / f"winter_{mode}_{area}.csv")
        step = pd.Timedelta(minutes=60 if mode == "hourly" else 15)
        for origin, g in p.groupby("fit_origin"):
            d = pd.Timestamp(origin).date()
            train = [
                r
                for r in items
                if d - timedelta(days=730) <= r[0] < d - timedelta(days=2)
            ]
            issue = pd.Timestamp(
                str(d - timedelta(days=1)) + " 10:00", tz="Europe/Copenhagen"
            ).tz_convert("UTC")
            last = max(r[1] + (len(r[3]) - 1) * step for r in train)
            assert last < issue and len(train) == g.training_days.iloc[0]
            allrows.append(
                dict(
                    mode=mode,
                    area=area,
                    refit_origin=d,
                    first_training_delivery=train[0][0],
                    last_training_delivery=train[-1][0],
                    training_days=len(train),
                    latest_label_utc=last,
                    issue_utc=issue,
                )
            )
pd.DataFrame(allrows).to_csv(O / "winter_training_windows.csv", index=False)
