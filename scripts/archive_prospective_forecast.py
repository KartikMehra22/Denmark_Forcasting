"""Timestamp and lock a future-day forecast before its 10:00 Copenhagen issue cutoff.
Usage: python archive_prospective_forecast.py --delivery-date YYYY-MM-DD --input predictions.csv
CSV must contain time_utc, area, model, prediction. Actual outcomes are forbidden.
This archival guard does not train or schedule models.
"""

from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--delivery-date", required=True)
p.add_argument("--input", required=True)
a = p.parse_args()
delivery = pd.Timestamp(a.delivery_date, tz="Europe/Copenhagen")
issue = pd.Timestamp(
    str(delivery.date() - timedelta(days=1)) + " 10:00", tz="Europe/Copenhagen"
)
now = pd.Timestamp.now(tz="UTC")
if now > issue:
    raise SystemExit(
        "Cannot archive as prospective: the declared issue cutoff has already passed."
    )
f = pd.read_csv(a.input)
if any(c.lower() in ["actual", "target", "observed", "truth"] for c in f):
    raise SystemExit("Actual outcome columns are forbidden in a forecast archive.")
required = {"time_utc", "area", "model", "prediction"}
if not required <= set(f):
    raise SystemExit("Required columns: " + ", ".join(sorted(required)))
f.time_utc = pd.to_datetime(f.time_utc, utc=True)
if not (f.time_utc.dt.tz_convert("Europe/Copenhagen").dt.date == delivery.date()).all():
    raise SystemExit("Delivery timestamps fall outside the declared delivery day.")
if (
    not np.isfinite(f.prediction).all()
    or f.duplicated(["time_utc", "area", "model"]).any()
):
    raise SystemExit("Invalid or duplicate forecast values.")
root = Path(__file__).resolve().parents[1] / "results/prospective" / a.delivery_date
root.mkdir(parents=True, exist_ok=True)
dest = root / "forecast.csv"
if dest.exists():
    raise SystemExit(
        "Archive exists: forecasts cannot overwrite an earlier prospective record."
    )
raw = f.to_csv(index=False).encode()
dest.write_bytes(raw)
(root / "manifest.json").write_text(
    json.dumps(
        {
            "delivery_date": a.delivery_date,
            "issue_cutoff": str(issue),
            "archived_at_utc": str(now),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "status": "awaiting outcomes",
        },
        indent=2,
    )
)
print(dest)
