# Reading the forecasting code

This is a short tour for someone new to the project. The [research workflow](RESEARCH_WORKFLOW.md) has the full file map, exact dates, outputs and limitations.

## 1. Start with the inputs

The price files are local artifacts under `data/`. For each zone, the project has an older hourly price series and a native 15-minute day-ahead price series. The hourly study joins the older hourly data with hourly averages of the newer prices. The 15-minute study uses genuine 15-minute values.

- `scripts/day_ahead_study.py`: joins the hourly source files and checks their continuity.
- `scripts/controlled_study.py`: `source()` selects the series; `samples()` turns it into one example per local delivery day.
- `scripts/rolling_completion.py`: `extended()` adds the frozen later-period prices.

A sample contains five pieces: delivery date, UTC start, the preceding seven days of prices, a fixed-length target array, and the number of intervals actually delivered that day. The final number matters at daylight-saving changes: a day may have 23 or 25 hours, or 92 or 100 quarter-hours.

## 2. Follow one forecast

Read `run()` in `scripts/rolling_completion.py`:

1. Load the selected price series and turn it into delivery-day samples.
2. Load the pinned Chronos model and calculate its quantile forecasts.
3. For each seven-day block, select eligible training days and fit ridge, LightGBM and PatchTST. LightGBM runs in `scripts/fit_lgb_worker.py` to keep its native runtime separate from PyTorch.
4. Forecast each delivery day, add the previous-day and previous-week benchmarks, and produce Chronos intervals. Hourly LEAR is calibrated separately.
5. Save predictions and checkpoints in `results/completion/`, and record each training window.

The forecast for day D is issued at 10:00 Copenhagen time on D−1. The code assumes the prices for delivery on D−1 were already published in the previous day's auction. The frozen price files do not establish publication or revision time for every individual observation.

## 3. Read the follow-up analysis

The `scripts/followup/` folder contains the work requested after the initial report:

- `supervisor_audit.py` and `supervisor_calibration_calendar.py` document data periods, fit windows and interval-calibration dates.
- `supervisor_lear_weekly.py` matches LEAR's refit frequency to the weekly models.
- `supervisor_winter.py` and `supervisor_spring.py` run the additional seasonal experiments. Their corresponding analysis and window scripts summarize the outputs.
- `supervisor_tune_lgb.py` tests a small set of LightGBM settings. This analysis was designed after seeing the original results and is exploratory.
- `supervisor_statistics.py` and `supervisor_seasonal_tests.py` compare daily errors against the previous-day benchmark. A positive mean daily improvement means the model had lower error than the benchmark.
- `supervisor_verify.py` and the checkpoint-verification scripts check the saved data, forecasts and model files.

## 4. Find the results

Once the separate research artifacts have been restored, start with:

| Question | Local file |
| --- | --- |
| Which dates were used? | `results/supervisor_response/periods.csv` |
| How much training data was used at each refit? | `training_windows.csv`, `winter_training_windows.csv`, `spring_training_windows.csv` in the same folder |
| How did the models compare? | `summary_metrics.csv`, `spring_metrics.csv`, `tuned_lgb_metrics.csv` |
| Are the differences statistically supported? | `dm_tests_with_spring.csv`, `seasonal_dm_tests_with_spring.csv` |
| Did the clean rerun reproduce the original forecasts? | `clean_run_verification.json` |
| Which source values and API snapshots were checked? | `snapshot_comparison.csv` and `api_snapshots/manifest.json` |

`results/completion/` holds the original rolling predictions and checkpoints. `results/supervisor_response/` holds the follow-up outputs. These directories are ignored by Git, so they are not present in a code-only clone.

MAE is mean absolute error; lower is better. The winter and spring runs were tuned separately from the original test, and the later test covers only 23 days. See [limitations](LIMITATIONS.md) before interpreting the numbers.
