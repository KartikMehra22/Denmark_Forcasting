"""Offline, daily rolling-origin hourly price study. Run: python3 day_ahead_study.py."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
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

def run():
    audits, predictions = {},[]
    for area in ['DK1','DK2']:
        s,audits[area] = load_area(area)
        x = features(s)
        valid = x.notna().all(axis=1)
        localdates = s.index.tz_convert('Europe/Copenhagen').date
        dates = sorted(set(localdates))
        # Last 180 complete local delivery days; no random split or test tuning.
        complete = [d for d in dates if len(s[localdates == d]) == int((pd.Timestamp(d,tz='Europe/Copenhagen')+pd.DateOffset(days=1)-pd.Timestamp(d,tz='Europe/Copenhagen')).total_seconds()/3600)]
        for d in complete[-180:]:
            delivery = pd.Timestamp(d,tz='Europe/Copenhagen')
            issue = delivery-pd.DateOffset(days=1)+pd.Timedelta(hours=10)
            test = (localdates == d) & valid
            # Conservative: train only on completed historical delivery intervals.
            train = valid & (s.index < issue.tz_convert('UTC').floor('D')) & (s.index >= issue-pd.Timedelta(days=730))
            assert s.index[train].max() < issue
            model = make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            model.fit(x.loc[train],s.loc[train])
            pred = model.predict(x.loc[test])
            frame = pd.DataFrame({'time_utc':s.index[test],'area':area,'issue_utc':issue.tz_convert('UTC'),
                'actual':s.loc[test].values,'previous_day':s.shift(24).loc[test].values,
                'previous_week':s.shift(168).loc[test].values,'ridge':pred})
            # Latest lagged value used by ridge is >=48h before target, earlier than issue.
            assert (frame.time_utc-pd.Timedelta(hours=48) < issue).all()
            predictions.append(frame)
        print(f'{area}: data audited; 180 delivery days evaluated',flush=True)
    p = pd.concat(predictions,ignore_index=True)
    p.to_csv(OUT/'hourly_predictions.csv',index=False)
    metrics=[]
    for area,g in p.groupby('area'):
        for name in ['previous_day','previous_week','ridge']:
            e=g[name]-g.actual
            metrics.append(dict(area=area,model=name,n=len(g),MAE=float(e.abs().mean()),RMSE=float(np.sqrt((e**2).mean()))))
    m=pd.DataFrame(metrics)
    m.to_csv(OUT/'metrics.csv',index=False)
    (OUT/'data_audit.json').write_text(json.dumps(audits,indent=2))
    report = '''# Initial hourly price forecasting results\n\nOffline study using existing CSV extracts; no data refresh performed.\n\nForecast origin: 10:00 Europe/Copenhagen on the previous day. Target: next local delivery day, including 23/25-hour daylight-saving days. Recent quarter-hour prices are averaged to hourly prices.\n\nEvaluation: latest 180 complete delivery days in the extracts; daily ridge refit on up to 730 days of history. Ridge alpha fixed at 10, with no tuning on this evaluation period. Scaler fitted only on each training window. Baselines use yesterday and last week's matching UTC hour.\n\nPrices and errors are EUR/MWh.\n\n```\n'''+m.to_string(index=False)+'''\n```\n\nLimitations: this is an initial retrospective benchmark, not final held-out research evidence. No LightGBM, transformer, holidays, external forecasts or probabilistic models yet. The previous-day baseline is feasible only if the D-1 auction results are available by the chosen issue time; publication-time assumptions need confirmation. UTC lag baselines can shift relative to local hour at DST transitions. Input snapshots are not publication-vintage archives. Future model selection requires an earlier validation period and a separately reserved final test.\n'''
    (OUT/'initial_report.md').write_text(report)
    print(m.to_string(index=False))

if __name__ == '__main__':
    run()
