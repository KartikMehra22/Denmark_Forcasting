# Reproducing the implementation

Use Python 3.12 and a dedicated virtual environment. Install requirements-lock.txt. Restore the required raw CSVs as described in DATA_SETUP.md. On a fresh checkout run python tools/bootstrap_settings.py once to install the model-only configuration in its expected location. Existing configurations are never overwritten.

The latest pipeline is python scripts/run_completion.py. For the first Chronos checkpoint download use DENMARK_ALLOW_MODEL_DOWNLOAD=1 python scripts/run_completion.py. It includes later data retrieval, rolling training, result summaries and independent verification. Model fitting can take around 20 minutes on the original machine.

Individual steps are scripts/fetch_later.py, scripts/rolling_completion.py, scripts/completion_summary.py and scripts/verify_completion.py. They write to untracked results/completion/.

Earlier alternatives are scripts/benchmark.py, scripts/deep_benchmark.py, scripts/controlled_study.py and scripts/final_figures.py. scripts/audit_study.py additionally requires saved original provenance and validation records, not just the model-only settings file. Exact numerical reproduction requires frozen data and the original full configuration artifacts.
