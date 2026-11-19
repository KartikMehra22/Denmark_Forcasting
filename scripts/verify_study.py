"""Independent checks of raw integrity, evaluation completeness and DST alignment."""

from pathlib import Path
import pandas as pd
from day_ahead_study import load_area

ROOT = Path(__file__).resolve().parents[1]
for area in ["DK1", "DK2"]:
    s, audit = load_area(area)
    assert all(a["missing_intervals"] == 0 and a["missing_prices"] == 0 for a in audit)
p = pd.read_csv(
    ROOT / "results/benchmark_v2/predictions.csv", parse_dates=["time_utc", "issue_utc"]
)
assert not p.duplicated(["area", "time_utc"]).any()
assert (p.issue_utc < p.time_utc).all()
for (area, split), g in p.groupby(["area", "split"]):
    assert (
        not g[["actual", "previous_day", "previous_week", "ridge", "lightgbm"]]
        .isna()
        .any()
        .any()
    )
    for day, h in g.groupby("date"):
        start = pd.Timestamp(day).tz_localize("Europe/Copenhagen")
        expected = pd.date_range(
            start, start + pd.DateOffset(days=1), freq="h", inclusive="left"
        ).tz_convert("UTC")
        assert set(h.time_utc) == set(expected), (area, day)
    issue = g.issue_utc.dt.tz_convert("Europe/Copenhagen")
    assert (issue.dt.hour == 10).all()
    assert (
        issue.dt.date == (pd.to_datetime(g.date) - pd.Timedelta(days=1)).dt.date
    ).all()
# Spring-forward day is present, complete, and contains 23 genuine delivery hours.
spring = p[(p.area == "DK1") & (p.date == "2026-03-29")]
assert len(spring) == 23
for area, g in p.groupby("area"):
    assert (
        g[g.split == "validation"].time_utc.max()
        < g[g.split == "exploratory_test"].time_utc.min()
    )
print(
    "PASS: raw continuity, complete forecasts, issue times, chronological splits, 23-hour DST day."
)
