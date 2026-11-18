"""Dataset facts + naive baselines for the meeting. Run from the folder that contains data/:
    python baselines.py
Uses the CSVs written by fetch_eds.py. Prints numbers you can quote on the call.
"""
import os

import numpy as np
import pandas as pd

D = "data"


def read(names, tcol):
    parts = [pd.read_csv(f"{D}/{n}.csv") for n in names if os.path.exists(f"{D}/{n}.csv")]
    if not parts:
        raise SystemExit(f"No files found for {names} in ./{D}/ (run fetch_eds.py first)")
    df = pd.concat(parts, ignore_index=True)
    df["t"] = pd.to_datetime(df[tcol])
    return df


# ---------- Demand (GrossConsumptionMWh), hourly, per price area ----------
cons = read(["production_consumption_history", "production_consumption"], "HourUTC")
cons = cons.drop_duplicates(["t", "PriceArea"]).sort_values(["PriceArea", "t"])
demand = cons.pivot(index="t", columns="PriceArea", values="GrossConsumptionMWh").asfreq("h")
print(f"Demand: {demand.index.min()} -> {demand.index.max()} (UTC), {len(demand)} hours per area")
print("Missing hours:", demand.isna().sum().to_dict())
print("Mean demand (MWh/h):", demand.mean().round(0).to_dict())

end = demand.index.max()
win = demand.index > end - pd.Timedelta(days=365)
print(f"\nNaive demand baselines, last 365 days (to {end.date()}):")
for lag, name in ((24, "same hour yesterday"), (168, "same hour last week")):
    err = (demand - demand.shift(lag))[win].abs()
    mape = (err / demand[win].abs()).mean() * 100
    print(f"  {name:22s} MAPE  " + "  ".join(f"{a}: {mape[a]:.2f}%" for a in demand.columns)
          + "   MAE(MWh) " + "  ".join(f"{a}: {err[a].mean():.0f}" for a in demand.columns))

# ---------- Prices (EUR/MWh), hourly: Elspot until 2025-09-30, then day-ahead (15-min -> hourly mean) ----------
parts = []
for area in ("DK1", "DK2"):
    old = read([f"elspot_{area}_history"], "HourUTC").set_index("t")["SpotPriceEUR"]
    new_df = read([f"dayahead_{area}"], "TimeUTC")
    new = new_df.set_index("t")["DayAheadPriceEUR"].resample("h").mean()
    s = pd.concat([old, new[new.index > old.index.max()]]).rename(area)
    parts.append(s)
price = pd.concat(parts, axis=1).asfreq("h")
print(f"\nPrices: {price.index.min()} -> {price.index.max()} (UTC)")
print("Share of hours with negative price:", (price < 0).mean().round(4).to_dict())
print("Min / max (EUR/MWh):", price.min().round(1).to_dict(), price.max().round(1).to_dict())
print("\nMean price and naive 'same hour last week' MAE, by year (shows how non-stationary it is):")
err = (price - price.shift(168)).abs()
tab = pd.concat([price.groupby(price.index.year).mean().add_suffix(" mean"),
                 err.groupby(err.index.year).mean().add_suffix(" MAE")], axis=1).round(1)
print(tab.to_string())

# ---------- Is the Great Belt link consistent between DK1 and DK2? ----------
gb = cons.pivot(index="t", columns="PriceArea", values="ExchangeGreatBelt_MWh")
if {"DK1", "DK2"} <= set(gb.columns):
    s = (gb["DK1"] + gb["DK2"]).abs().dropna()
    print(f"\nGreat Belt: |DK1 flow + DK2 flow| median {s.median():.2f}, 99th pct {s.quantile(.99):.2f}, "
          f"max {s.max():.1f} MWh (typical |flow| {gb['DK1'].abs().median():.0f} MWh)")