# Reproducing the implementation

The [research workflow and file guide](RESEARCH_WORKFLOW.md) maps each experiment and supervisor request to its code, settings, data and outputs. This page gives the basic setup.

Use Python 3.12 and a dedicated virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -r requirements-supervisor.txt
python -m pip check
```

Restore the frozen raw inputs and saved artifacts described in [data setup](DATA_SETUP.md). Data, forecasts, model files and other results are ignored by Git. On a fresh code checkout, `python tools/bootstrap_settings.py` installs the model-only configuration without overwriting an existing configuration. Exact reproduction of the saved study additionally requires its frozen data and full configuration artifacts.

The original rolling pipeline is `python scripts/run_completion.py`. Its first Chronos download can be enabled with `DENMARK_ALLOW_MODEL_DOWNLOAD=1`; alternatively restore the pinned local model cache from the reproducibility package. The pipeline writes to `results/completion/` and can take substantial time. Preserve the frozen artifacts before running it.

Follow-up checks and summaries are run from the project root:

```sh
python scripts/followup/supervisor_verify.py
python scripts/followup/supervisor_spring_analysis.py
python scripts/followup/supervisor_tune_lgb_analysis.py
```

The winter, spring and tuned-LightGBM forecasting scripts refit many models and write to `results/supervisor_response/`. The script map and full interpretation limits are in the [workflow guide](RESEARCH_WORKFLOW.md). Earlier reproducibility archives capture the code layout before the follow-up scripts were moved into `scripts/followup/`. Use the paths in each archive’s own README when working from an archive.
