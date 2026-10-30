"""Offline, daily rolling-origin hourly price study. Run: python3 day_ahead_study.py."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def load_area(area):
    audit = []
    series = []
    for name, tc, pc, freq in [(f'elspot_{area}_history','HourUTC','SpotPriceEUR','h'),
                               (f'dayahead_{area}','TimeUTC','DayAheadPriceEUR','15min')]:
        path = ROOT / 'data' / f'{name}.csv'
        raw = pd.read_csv(path)
        t = pd.to_datetime(raw[tc], utc=True)
        s = pd.Series(pd.to_numeric(raw[pc], errors='raise').values,index=t).sort_index()
        if s.index.duplicated().any():
            raise ValueError(f'{name}: duplicate timestamps')
        if not (raw.PriceArea == area).all():
            raise ValueError(f'{name}: unexpected area')
        expected = pd.date_range(s.index.min(),s.index.max(),freq=freq)
        audit.append(dict(file=path.name,rows=len(s),start=str(s.index.min()),end=str(s.index.max()),
                          missing_intervals=len(expected.difference(s.index)),missing_prices=int(s.isna().sum())))
        if freq == '15min':
            counts = s.resample('h').count()
            if not (counts == 4).all():
                raise ValueError('Incomplete quarter-hour aggregation')
            s = s.resample('h').mean()
        series.append(s)
    old,new = series
    if new.index.min() != old.index.max()+pd.Timedelta(hours=1):
        raise ValueError(f'{area}: series do not join cleanly')
    s = pd.concat(series).asfreq('h')
    if s.isna().any():
        raise ValueError('Missing prices in combined series')
    return s,audit

def features(s):
    x = pd.DataFrame(index=s.index)
    # At 10:00 Copenhagen on D-1, the whole D-1 price curve is already known.
    # Use D-2 and older curves so all features are safely available at this cutoff.
    for lag in [48,72,168,336]:
        x[f'lag_{lag}h'] = s.shift(lag)
    for w in [24,168]:
        x[f'mean_{w}h'] = s.shift(48).rolling(w).mean()
    local = s.index.tz_convert('Europe/Copenhagen')
    for name,values,period in [('hour',local.hour,24),('weekday',local.dayofweek,7),('year',local.dayofyear,365.25)]:
        x[f'{name}_sin'] = np.sin(2*np.pi*values/period)
        x[f'{name}_cos'] = np.cos(2*np.pi*values/period)
    x['weekend'] = (local.dayofweek >= 5).astype(int)
    return x

