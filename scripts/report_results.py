from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache'/'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
OUT=ROOT/'results'/'benchmark_v2'
p=pd.read_csv(OUT/'all_model_predictions.csv',parse_dates=['time_utc'])
m=pd.read_csv(OUT/'all_model_metrics.csv')
models=['previous_day','previous_week','ridge','lightgbm','patchtst','chronos_bolt_tiny']
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,area in zip(axs,['DK1','DK2']):
    g=m[m.area==area].sort_values('MAE',ascending=False)
    ax.barh(g.model,g.MAE,color=['#158a8a' if x=='lightgbm' else '#8194ae' for x in g.model])
    ax.set(title=area,xlabel='MAE (EUR/MWh)')
fig.suptitle('Hourly day-ahead prices: exploratory 180-day comparison')
fig.savefig(OUT/'model_comparison.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(13,7),layout='constrained')
for ax,area in zip(axs,['DK1','DK2']):
    g=p[(p.area==area)&(p.time_utc>=p.time_utc.max()-pd.Timedelta(days=7))]
    for col,label,c in [('actual','Actual','#172b4d'),('lightgbm','LightGBM','#168a8a'),('chronos_bolt_tiny','Chronos-Bolt tiny','#d98835')]:
        ax.plot(g.time_utc,g[col],label=label,color=c,lw=1.2)
    ax.set(title=area,ylabel='EUR/MWh');ax.legend(ncol=3)
fig.suptitle('Last seven delivery days in the existing extract')
fig.savefig(OUT/'forecast_example.png',dpi=160);plt.close(fig)
regimes=[]
for area,g in p.groupby('area'):
    threshold=g.actual.quantile(.95)
    for label,mask in [('all',np.ones(len(g),dtype=bool)),('negative',g.actual<0),('upper_5pct',g.actual>=threshold)]:
        for model in models:
            e=g.loc[mask,model]-g.loc[mask,'actual']
            regimes.append(dict(area=area,regime=label,model=model,n=len(e),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean())))
pd.DataFrame(regimes).to_csv(OUT/'regime_metrics.csv',index=False)
# Paired moving-block bootstrap of daily MAE improvement; uncertainty is descriptive.
rng=np.random.default_rng(42);intervals=[]
for area,g in p.groupby('area'):
    daily=g.groupby('date').apply(lambda x:pd.Series({model:(x[model]-x.actual).abs().mean() for model in models}),include_groups=False)
    for model in ['ridge','lightgbm','patchtst','chronos_bolt_tiny']:
        delta=(daily.previous_day-daily[model]).to_numpy();n=len(delta)
        means=[]
        for _ in range(2000):
            starts=rng.integers(0,n,size=int(np.ceil(n/7)))
            idx=np.concatenate([(np.arange(k,k+7)%n) for k in starts])[:n]
            means.append(delta[idx].mean())
        lo,hi=np.quantile(means,[.025,.975]);intervals.append(dict(area=area,model=model,daily_MAE_reduction=delta.mean(),low95=lo,high95=hi))
pd.DataFrame(intervals).to_csv(OUT/'daily_error_bootstrap.csv',index=False)
rows=['| Zone | Model | MAE | RMSE | MAE improvement vs yesterday |','|---|---|---:|---:|---:|']
for r in m.itertuples():rows.append(f'| {r.area} | {r.model} | {r.MAE:.2f} | {r.RMSE:.2f} | {r.improvement_pct:.1f}% |')
text='''# Denmark day-ahead price forecasting — initial research results

Prepared 2 November 2026. All numbers were computed locally from the existing Energinet CSV extracts; no new price data has been downloaded.

## Main finding

LightGBM has the lowest MAE in both bidding zones, approximately 19.5–19.7% below the previous-day baseline. Chronos-Bolt-tiny has the lowest RMSE. The initial PatchTST configuration improves MAE slightly in DK1 but does not beat the previous-day baseline in DK2. These results do not establish that one model family is universally superior.

## Data checked

Per zone: 32,855 historical hourly prices and 37,344 quarter-hour prices. No missing timestamps or price values, no duplicate timestamps, and a continuous join. Quarter-hour data ends 24 October 2026 at 21:45 UTC. Quarter-hour prices were averaged within each UTC hour. Raw files were preserved.

## Experimental setup

- Forecast issue: 10:00 Europe/Copenhagen on D-1, before the D auction results.
- Price-availability assumption: the entire D-1 delivery price curve was published on D-2 and is available at issue time. Known future delivery prices are allowed; tomorrow's auction prices are never inputs.
- Validation: 28 January–27 April 2026. Exploratory evaluation: 28 April–24 October 2026 (180 days, 4,320 hourly predictions per zone).
- Statistical/LightGBM inputs: complete DK1/DK2 price curves for D-1, D-2, D-3 and D-7, matching-hour lags, delivery hour, weekday, month, and Danish public holiday flag.
- Ridge: alpha selected only on the earlier validation period. Scaling fitted on each training window.
- LightGBM: fixed modest configuration, 150 trees, 15 leaves, learning rate 0.05, absolute-error objective. No test-set hyperparameter search.
- Ridge and LightGBM refit monthly using up to 730 preceding days. Training delivery dates are strictly before the issue date.
- PatchTST: Hugging Face implementation, trained from scratch per zone. 168-hour context; 25-step head covers DST day lengths; two layers, 32-dimensional embeddings. Epoch selected on earlier validation (DK1: 11, DK2: 15), then retrained before the evaluation period. No evaluation-period refitting.
- Chronos: `amazon/chronos-bolt-tiny`, zero-shot median forecast, 168-hour context, 25-step forecast sliced to the delivery day's actual length. It is the Bolt tiny variant, not original Chronos-T5 or Chronos-2.
- Deep models use univariate context ending at the start of the target local delivery day; the final D-1 prices in that context are already published under the assumption above.
- DST: forecast origin stays at 10:00 local time. Evaluation retains real 23/25-hour delivery days. Repeated historical fall-back hour averaged, missing spring-forward hour interpolated only when creating already-published daily lag curves.

## Results (EUR/MWh)

'''+ '\n'.join(rows)+'''

![Model comparison](model_comparison.png)

![Example forecasts](forecast_example.png)

## Checks completed

Raw timestamp continuity and missing values; uniqueness/completeness of model predictions; forecast origin before target; chronological validation/evaluation separation; and explicit verification of the 23-hour spring DST delivery day. All passed. A DST origin construction error was corrected before reporting: it affected recorded issue times, not training samples, inputs or predictions.

## Interpretation and limitations

1. The 180-day evaluation period was already viewed in the first exploratory run. It is not a pristine final holdout. Reserve a newly collected future period for a locked final evaluation.
2. This is not an equal-input/equal-refit comparison: LightGBM/ridge have both zones and calendar features plus monthly refits; deep models have univariate price context, and PatchTST is trained once. Differences cannot be attributed solely to architecture.
3. Pretrained Chronos training-corpus overlap with historical electricity series has not been audited. Its zero-shot result means no local fitting, not proven unseen pretraining data.
4. Data are downloaded price snapshots, not archived publication vintages. The assumed publication cutoff still needs confirmation, especially on exceptional delayed-auction days.
5. Only one small PatchTST configuration and seed have been tested; it is an initial benchmark, not a definitive model optimisation.
6. No weather, wind, solar, load forecast, native quarter-hour modelling, or physical conservation term is included yet.
7. Monthly refits differ from the originally proposed daily-refit experiment and are documented as such.

`regime_metrics.csv` contains negative-price and upper-tail error breakdowns. `daily_error_bootstrap.csv` gives descriptive 7-day-block bootstrap intervals for paired daily MAE reduction; these are not confirmatory significance claims after model exploration.

## Next research steps

Lock the initial specification and obtain a prospective evaluation period; compare models with matched inputs/refit schedules; then evaluate native 15-minute data and additional forecast datasets as agreed with the supervisor.

## Reproduce

From the Denmark project: `.venv/bin/python benchmark.py`, `.venv/bin/python deep_benchmark.py`, `.venv/bin/python verify_study.py`, `.venv/bin/python report_results.py`.

Sources: [Energinet price metadata](https://api.energidataservice.dk/meta/dataset/Elspotprices), [LightGBM](https://lightgbm.readthedocs.io/en/stable/Python-API.html), [PatchTST](https://github.com/yuqinie98/PatchTST), [Chronos](https://github.com/amazon-science/chronos-forecasting).
'''
(OUT/'research_report.md').write_text(text)
print('Report, charts, regime metrics and bootstrap intervals saved.')
