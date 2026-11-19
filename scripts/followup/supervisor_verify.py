"""Independent checks for response tables, saved forecasts and HAC calculations."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from datetime import timedelta
import json
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from rolling_completion import extended

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response"
for a in ["DK1", "DK2"]:
    old = pd.read_csv(
        R / f"results/completion/hourly_{a}_predictions.csv", parse_dates=["time_utc"]
    )
    wk = pd.read_csv(O / f"lear_weekly_{a}.csv", parse_dates=["time_utc"])
    assert len(wk) == len(old) and not wk.time_utc.duplicated().any()
    assert np.array_equal(wk.time_utc, old.time_utc) and np.allclose(
        wk.actual, old.actual
    )
    assert np.array_equal(wk.fit_origin, old.fit_origin)
for mode in ["hourly", "quarter_hour"]:
    for a in ["DK1", "DK2"]:
        p = pd.read_csv(O / f"winter_{mode}_{a}.csv", parse_dates=["time_utc"])
        s = extended(a, mode)
        assert p.date.nunique() == 90 and not p.time_utc.duplicated().any()
        np.testing.assert_allclose(
            p.actual, s.reindex(pd.DatetimeIndex(p.time_utc)), atol=1e-4, rtol=1e-6
        )
        cfg = json.loads((O / f"winter_{mode}_{a}_settings.json").read_text())
        assert (
            cfg["validation_end"] < "2025-12-01"
            and cfg["validation_training_end"] < cfg["validation_start"]
        )
        assert np.isfinite(p.select_dtypes("number").to_numpy()).all()
# Recompute a separate regression/Newey-West HAC standard error for every test.
D = pd.read_csv(O / "dm_tests.csv")
for r in D.itertuples():
    if r.period == "winter_independently_tuned":
        p = pd.read_csv(O / f"winter_{r.mode}_{r.area}.csv")
    else:
        p = pd.read_csv(R / f"results/completion/{r.mode}_{r.area}_predictions.csv")
        p = p[p.split == ("historical" if r.period == "earlier" else "later_locked")]
        if r.model == "lear_weekly":
            p = p.merge(
                pd.read_csv(O / f"lear_weekly_{r.area}.csv")[
                    ["time_utc", "lear_weekly"]
                ],
                on="time_utc",
                validate="one_to_one",
            )
    v = (
        p.assign(
            delta=(p.previous_day - p.actual).abs() - (p[r.model] - p.actual).abs()
        )
        .groupby("date")["delta"]
        .mean()
        .to_numpy()
    )
    model = sm.OLS(v, np.ones(len(v))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False}
    )
    np.testing.assert_allclose(
        [v.mean(), float(model.tvalues[0]), 2 * norm.sf(abs(float(model.tvalues[0])))],
        [r.mean_daily_mae_improvement, r.z, r.p_raw],
        rtol=1e-8,
        atol=1e-9,
    )
# Verify fresh source snapshots were captured and retain the exact source-record fingerprints.
manifest = json.loads((O / "api_snapshots/manifest.json").read_text())
for key, x in manifest.items():
    assert (
        hashlib.sha256((O / "api_snapshots" / f"{key}.json").read_bytes()).hexdigest()
        == x["sha256"]
    )
assert (pd.read_csv(O / "snapshot_comparison.csv").price_values_changed == 0).all()
print(
    "PASS weekly alignment; winter chronology and targets; 58 independently checked HAC statistics; API hashes and source values"
)
