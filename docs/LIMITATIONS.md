# Scope and limitations

This repository implements hourly and native quarter-hour day-ahead price forecasting for DK1 and DK2. It includes seasonal baselines, ridge, LightGBM, PatchTST, Chronos-Bolt-tiny, an hourly price-only LEAR implementation, and sequential interval expansion.

The completion pipeline uses weekly refits for ridge, LightGBM and PatchTST; LEAR recalibrates daily and is reported separately. The controlled fixed-fit comparison is a different experiment. Model inputs, refit schedules and pretraining must be accounted for when interpreting comparisons.

Chronos-Bolt-tiny's native quarter-hour forecast exceeds its recommended 64-step horizon. Serial dependence and market shifts mean interval calibration does not guarantee future nominal coverage. A later historical backtest must not be called a live prospective trial.

The original shared report and supervisor's requests require further experimental clarifications, matched LEAR refit comparisons, seasonal significance testing and fuller artifact sharing. Importing this code does not imply those follow-up requests have been completed.

Load, generation, aFRR and mFRR are later project stages.
