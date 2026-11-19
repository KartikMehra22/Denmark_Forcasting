# Denmark electricity price forecasting

Research code for day-ahead electricity price forecasting in Denmark's DK1 and DK2 bidding zones, at hourly and native 15-minute resolution. The project compares the previous-day benchmark with ridge regression, LightGBM, PatchTST, Chronos-Bolt and an hourly LEAR model.

Start with the [research workflow and file guide](docs/RESEARCH_WORKFLOW.md). It maps the supervisor's requests to the code, experimental periods, local result files, verification and remaining limitations.

## Project structure

```text
config/                 Model settings used for bootstrapping
scripts/                Original data, forecasting and evaluation pipeline
scripts/followup/       Follow-up experiments, diagnostics and verification
docs/                   Research workflow, data setup and limitations
tools/                  Configuration bootstrap
completion_protocol.json Original rolling experiment settings
```

`data/`, `results/` and `reports/` are local research artifacts excluded from Git. They include datasets, API responses, forecasts, figures and checkpoints. A separate reproducibility package is required to share or reproduce the saved results.

## Setup and checks

Use Python 3.12, restore the frozen data and result artifacts, and install the pinned dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -r requirements-supervisor.txt
python -m pip check
python scripts/followup/supervisor_verify.py
```

See [data setup](docs/DATA_SETUP.md), [reproduction](docs/REPRODUCING.md) and the [research workflow](docs/RESEARCH_WORKFLOW.md) for full commands and prerequisites. Experiments are retrospective; the short later test and other limitations are described in the guide.

## History provenance

This repository was retrospectively imported in 55 reconstructed stages. The author dates organise that import and do not establish when the work was originally performed. See [history provenance](HISTORY_PROVENANCE.md). New changes are recorded as ordinary commits when made.
