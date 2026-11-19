# Scope and limitations

The project studies DK1 and DK2 day-ahead prices at hourly and native 15-minute resolution. It compares previous-day and previous-week baselines, ridge, LightGBM, PatchTST, Chronos-Bolt-tiny and an hourly price-only LEAR model. Load, generation, aFRR and mFRR are later project stages, not completed parts of this study.

The original rolling comparison uses weekly refits for ridge, LightGBM and PatchTST and daily LEAR recalibration. A separate matched weekly LEAR comparison is included. These models differ in input features and pretraining, and the controlled fixed-fit experiment is a different comparison.

The later historical test contains 23 delivery days and cannot establish a stable ranking. The added winter and spring evaluations are retrospective and use settings selected before their respective tests; those settings and available training histories differ from the original experiment. The first spring interval-calibration window overlaps its model-selection period by 26 days, and the earlier original intervals also have overlapping dates. Dependence and market shifts mean interval calibration does not guarantee future nominal coverage. Chronos-Bolt's quarter-hour horizon also exceeds its native 64-step recommendation; its continuation errors are confounded with time of day.

The code assumes D−1 day-ahead prices are available at 10:00 Copenhagen on D−1 when forecasting day D. Nord Pool's normal auction schedule supports the assumption, but the retained snapshots have no per-observation publication or revision timestamps. Historical original HTTP response bodies for the initial extracts were not retained; fresh responses match the retained historical price values but are not original responses.

Later-period LEAR and LightGBM errors remain higher than the previous-day benchmark in the observed 23-day sample. Input distribution shift was observed and a small post-hoc LightGBM tuning experiment improved its MAE, but neither proves the cause of deterioration. The retrospective analysis is not a live prospective trial.

Data, forecasts, checkpoints and results are excluded from Git and require a separate reproducibility package. The [research workflow](RESEARCH_WORKFLOW.md) maps each request to the relevant code and local artifact.
