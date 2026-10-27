"""Pull Danish electricity data from Energi Data Service (Energinet) and run quick checks.

Run:   python fetch_eds.py            (skips files already in ./data/)
       python fetch_eds.py --force    (re-download everything)
Needs: pip install requests pandas matplotlib
Output: CSVs and overview.png in ./data/
The script waits between requests and retries on HTTP 429 (rate limit), so it can take a few minutes.
"""
import json
import os
import sys
import time

import pandas as pd
import requests

BASE = "https://api.energidataservice.dk/dataset"
OUT = "data"
MIN_GAP = 2.0  # seconds between any two requests
_last = [0.0]


def get(url, params):
    """GET with a global throttle and retries on 429/5xx (honours Retry-After)."""
    for attempt in range(8):
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, params=params, timeout=300)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            delay = min(10 * 2 ** attempt, 120)
            print(f"  network error ({type(exc).__name__}); waiting {delay}s (attempt {attempt + 1}/8)")
            time.sleep(delay)
            continue
        _last[0] = time.time()
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After", "")
            delay = float(ra) if ra.replace(".", "", 1).isdigit() else min(10 * 2 ** attempt, 120)
            print(f"  HTTP {r.status_code}; waiting {delay:.0f}s (attempt {attempt + 1}/8)")
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("gave up after 8 retries")


def fetch(dataset, start, end, price_area=None, page=10000):
    rows, offset = [], 0
    while True:
        params = {"start": start, "end": end, "limit": page, "offset": offset}
        if price_area:
            params["filter"] = json.dumps({"PriceArea": price_area})
        recs = get(f"{BASE}/{dataset}", params).json().get("records", [])
        rows.extend(recs)
        print(f"  {dataset}: {len(rows)} rows so far")
        if len(recs) < page:
            break
        offset += page
    return pd.DataFrame(rows)


def time_col(df):
    cols = [c for c in df.columns if "utc" in c.lower()]
    return cols[0] if cols else None


def summarize(name, df):
    print(f"\n=== {name} ===")
    if df.empty:
        print("No rows returned.")
        return
    print("rows:", len(df))
    print("columns:", list(df.columns))
    tc = time_col(df)
    if tc:
        df = df.assign(_t=pd.to_datetime(df[tc]))
        print("range:", df["_t"].min(), "->", df["_t"].max())
        groups = df.groupby("PriceArea") if "PriceArea" in df.columns else [("all", df)]
        for area, g in groups:
            step = g["_t"].sort_values().diff().dropna()
            mode = step.mode().iloc[0] if len(step) else "n/a"
            gaps = int((step > mode).sum()) if len(step) else 0
            print(f"  {area}: {len(g)} rows, step {mode}, gaps larger than step: {gaps}")
    miss = df.isna().sum()
    print("columns with missing values:\n", miss[miss > 0].to_string() if (miss > 0).any() else " none")


PROD = ["CentralPowerMWh", "LocalPowerMWh", "CommercialPowerMWh", "OffshoreWindLt100MW_MWh",
        "OffshoreWindGe100MW_MWh", "OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh", "HydroPowerMWh",
        "SolarPowerLt10kW_MWh", "SolarPowerGe10Lt40kW_MWh", "SolarPowerGe40kW_MWh",
        "SolarPowerSelfConMWh", "UnknownProdMWh", "LocalPowerSelfConMWh"]


