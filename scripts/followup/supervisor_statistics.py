"""Paired daily-loss DM-style tests, HAC Bartlett(6), Holm adjustment, seasonal errors."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response"
comparisons = []
seasons = []
horizons = []
for mode in ["hourly", "quarter_hour"]:
    for area in ["DK1", "DK2"]:
        p = pd.read_csv(
            R / f"results/completion/{mode}_{area}_predictions.csv",
            parse_dates=["time_utc"],
        )
        p["period"] = p["split"].map({"historical": "earlier", "later_locked": "later"})
        if mode == "hourly":
            w = pd.read_csv(O / f"lear_weekly_{area}.csv", parse_dates=["time_utc"])
            p = p.merge(
                w[["time_utc", "lear_weekly"]], on="time_utc", validate="one_to_one"
            )
        win = pd.read_csv(O / f"winter_{mode}_{area}.csv", parse_dates=["time_utc"])
        win["period"] = "winter_independently_tuned"
        datasets = [(p, False), (win, True)]
        for frame, iswinter in datasets:
            for period, g in frame.groupby("period"):
                models = ["ridge", "lightgbm", "patchtst", "chronos"] + (
                    ["lear_weekly", "lear"]
                    if mode == "hourly" and not iswinter
                    else ["lear_weekly"] if mode == "hourly" else []
                )
                for model in models:
                    daily = (
                        g.assign(
                            diff=(g.previous_day - g.actual).abs()
                            - (g[model] - g.actual).abs()
                        )
                        .groupby("date")["diff"]
                        .mean()
                        .to_numpy()
                    )
                    n = len(daily)
                    center = daily - daily.mean()
                    L = min(6, n - 1)
                    var = np.dot(center, center) / n
                    for lag in range(1, L + 1):
                        var += (
                            2
                            * (1 - lag / (L + 1))
                            * np.dot(center[lag:], center[:-lag])
                            / n
                        )
                    z = daily.mean() / np.sqrt(var / n) if var > 0 else np.nan
                    pval = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
                    comparisons.append(
                        dict(
                            mode=mode,
                            area=area,
                            period=period,
                            model=model,
                            days=n,
                            mean_daily_mae_improvement=daily.mean(),
                            hac_lag=L,
                            z=z,
                            p_raw=pval,
                            earlier_features="absolute loss; positive means model improves on previous day",
                        )
                    )
            f = frame.copy()
            local = f.time_utc.dt.tz_convert("Europe/Copenhagen")
            f["month"] = local.dt.month
            f["season"] = f.month.map(
                {
                    12: "winter",
                    1: "winter",
                    2: "winter",
                    3: "spring",
                    4: "spring",
                    5: "spring",
                    6: "summer",
                    7: "summer",
                    8: "summer",
                    9: "autumn",
                    10: "autumn",
                    11: "autumn",
                }
            )
            # The independently tuned winter run is not mixed with the original fixed-setting evaluation.
            for season, g in f.groupby("season"):
                for model in [
                    "previous_day",
                    "ridge",
                    "lightgbm",
                    "patchtst",
                    "chronos",
                ] + (["lear_weekly"] if mode == "hourly" else []):
                    e = (g[model] - g.actual).abs()
                    seasons.append(
                        dict(
                            mode=mode,
                            area=area,
                            experiment=(
                                "winter_independently_tuned" if iswinter else "original"
                            ),
                            season=season,
                            model=model,
                            days=g.date.nunique(),
                            n=len(g),
                            MAE=e.mean(),
                        )
                    )
            # Actual elapsed forecast lead; aggregate repeated/missing local clock hours without forcing 96 labels.
            if mode == "quarter_hour":
                f = f.sort_values("time_utc")
                f["lead"] = f.groupby("date").cumcount() + 1
                for lead, g in f.groupby("lead"):
                    horizons.append(
                        dict(
                            area=area,
                            experiment=(
                                "winter_independently_tuned" if iswinter else "original"
                            ),
                            lead=lead,
                            n=len(g),
                            chronos_MAE=(g.chronos - g.actual).abs().mean(),
                            previous_day_MAE=(g.previous_day - g.actual).abs().mean(),
                        )
                    )
D = pd.DataFrame(comparisons)
# Holm controls the whole family of reported comparisons, zones, resolutions and evaluation periods.
idx = D.p_raw.dropna().sort_values().index
n = len(idx)
adj = []
running = 0
for rank, i in enumerate(idx):
    running = max(running, min(1, (n - rank) * D.loc[i, "p_raw"]))
    adj.append(running)
D["p_holm"] = np.nan
D.loc[idx, "p_holm"] = adj
D.to_csv(O / "dm_tests.csv", index=False)
pd.DataFrame(seasons).to_csv(O / "seasonal_metrics.csv", index=False)
pd.DataFrame(horizons).to_csv(O / "chronos_horizon.csv", index=False)
print("DM tests", len(D), "Holm family", n)
