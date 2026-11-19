"""Independent numerical, provenance and information-cutoff audit."""

from pathlib import Path
from datetime import timedelta
import hashlib
import json
import numpy as np
import pandas as pd
from controlled_study import source, samples, OUT, ROOT

findings = []


def record(check, status, detail):
    findings.append(dict(check=check, status=status, detail=detail))


metrics = pd.read_csv(OUT / "metrics.csv")
cfg = json.loads((OUT / "configuration.json").read_text())
hashes = json.loads((OUT / "input_hashes.json").read_text())
for name, expected in hashes.items():
    assert (
        hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == expected
    ), name
record(
    "Raw data provenance", "PASS", "All six input hashes match the saved experiment."
)
for mode in ["hourly", "quarter_hour"]:
    for area in ["DK1", "DK2"]:
        key = f"{mode}_{area}"
        c = cfg[key]
        s = source(area, mode)
        rows = samples(s, mode)
        p = pd.read_csv(OUT / f"{key}_predictions.csv")
        base = np.mean(abs(p.previous_day - p.actual))
        for r in metrics[
            (metrics["mode"] == mode) & (metrics.area == area)
        ].itertuples():
            err = p[r.model] - p.actual
            np.testing.assert_allclose(
                [
                    err.abs().mean(),
                    np.sqrt(np.mean(err**2)),
                    100 * (1 - err.abs().mean() / base),
                ],
                [r.MAE, r.RMSE, r.improvement_pct],
                rtol=2e-6,
                atol=1e-5,
            )
            assert r.n == len(p)
        # Assert exact latest label timestamp, not merely delivery date ordering.
        for origin, lower in [(c["validation_start"], False), (c["test_start"], True)]:
            d = pd.Timestamp(origin).date()
            issue = pd.Timestamp(
                str(d - timedelta(days=1)) + " 10:00", tz="Europe/Copenhagen"
            ).tz_convert("UTC")
            selected = [
                r
                for r in rows
                if d - timedelta(days=730) <= r[0] < d - timedelta(days=2)
            ]
            step = (
                pd.Timedelta(hours=1) if mode == "hourly" else pd.Timedelta(minutes=15)
            )
            assert all(r[1] + (len(r[3]) - 1) * step < issue for r in selected)
        # Verify selection can be recovered without looking at evaluation errors.
        scores = c["ridge_validation_scores"]
        assert int(min(scores, key=scores.get)) == c["ridge_alpha"]
        assert np.argmin(c["patchtst_validation_mae"]) + 1 == c["patchtst_epochs"]
record(
    "Saved metrics",
    "PASS",
    "Recomputed MAE, RMSE, improvement and row count for every model in all four runs.",
)
record(
    "Training cutoff",
    "PASS",
    "All training labels precede issue time for validation and evaluation fits.",
)
record(
    "Hyperparameter selection",
    "PASS",
    "Saved ridge alpha and PatchTST epoch reproduce earlier-validation selections.",
)
record(
    "Original rolling-origin plan",
    "PARTIAL",
    "Earlier ridge/LightGBM run has monthly refits; controlled comparison fixes all trained models. No matched rolling-refit comparison yet.",
)
record(
    "Untouched evaluation",
    "OPEN",
    "Hourly evaluation has been inspected; no locked newly collected evaluation exists. Current price data end 24 October 2026.",
)
record(
    "Quarter-hour uncertainty",
    "OPEN",
    "Raw nominal 80% coverage is about 59–60%; calibration is not implemented.",
)
record(
    "LEAR specification",
    "PARTIAL",
    "Regularised ridge autoregression was implemented, not a canonical LASSO-estimated LEAR model.",
)
record(
    "Information availability",
    "ASSUMPTION",
    "Standard prior-day auction publication is assumed; exceptional historical delays have not been audited.",
)
record(
    "Model comparison",
    "LIMITATION",
    "Matched context/refit schedule does not match parameter count, pretraining or optimisation budget. PatchTST is a compact initial model.",
)
record(
    "Later datasets",
    "OUT OF SCOPE",
    "Load, generation, aFRR and mFRR are deferred stages, not completed experiments.",
)
(OUT / "audit_2026-11-17.json").write_text(json.dumps(findings, indent=2))
lines = [
    "# Research audit 17 November 2026",
    "",
    "The initial benchmarks are implemented and their saved numbers check out. The full original evaluation plan is not complete.",
    "",
]
for f in findings:
    lines += ["## " + f["check"], "", f["status"] + ": " + f["detail"], ""]
(OUT / "audit_2026-11-17.md").write_text("\n".join(lines))
print("\n".join(f["status"] + " " + f["check"] for f in findings))
