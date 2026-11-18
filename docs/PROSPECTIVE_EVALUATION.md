# Prospective evaluation

A live evaluation requires recording each input snapshot and forecast before the corresponding auction results. The existing scripts/archive_prospective_forecast.py accepts --delivery-date and --input, refuses late submissions, rejects actual-outcome columns and prevents overwriting an existing forecast archive.

This guard does not train models or schedule a daily run. A historical forecast cannot be relabelled as prospective. Keep actual outcomes separate until forecasts are archived, then join them for evaluation.
