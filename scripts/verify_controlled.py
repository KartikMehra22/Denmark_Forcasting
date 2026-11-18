from datetime import timedelta
import json,numpy as np,pandas as pd
from controlled_study import source,samples,OUT
cfg=json.loads((OUT/'configuration.json').read_text())
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
    s=source(area,mode);expected_step=pd.Timedelta(hours=1) if mode=='hourly' else pd.Timedelta(minutes=15)
    assert not s.index.duplicated().any() and (s.index.to_series().diff().dropna()==expected_step).all()
    p=pd.read_csv(OUT/f'{mode}_{area}_predictions.csv',parse_dates=['time_utc'])
    assert not p.time_utc.duplicated().any() and np.isfinite(p.select_dtypes('number')).all().all()
    for date,g in p.groupby('date'):
        st=pd.Timestamp(date).tz_localize('Europe/Copenhagen');en=st+pd.DateOffset(days=1)
        assert set(g.time_utc)==set(pd.date_range(st,en,freq=expected_step,inclusive='left').tz_convert('UTC'))
        actual=s.reindex(pd.DatetimeIndex(g.time_utc)).values
        np.testing.assert_allclose(actual,g.actual,atol=5e-5,rtol=1e-6)
    # Changing every target and future value must leave the prediction context unchanged.
    d=pd.Timestamp(cfg[f'{mode}_{area}']['test_start']).date()
    row=next(r for r in samples(s,mode) if r[0]==d)
    altered=s.copy();altered.loc[altered.index>=row[1]]=123456.
    other=next(r for r in samples(altered,mode) if r[0]==d)
    np.testing.assert_array_equal(row[2],other[2])
    assert np.isfinite(p[['chronos_q10','chronos_q90']]).all().all()
    assert (p.chronos_q10<=p.chronos_q90).all()
    print(mode,area,'PASS: continuity, actuals, delivery days, causal contexts, quantiles')
