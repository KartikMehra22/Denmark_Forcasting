# Denmark day-ahead electricity price forecasting

This project studies how well different models forecast the next day's electricity prices in Denmark. It covers the DK1 and DK2 price zones at hourly and 15-minute resolution.

The starting point is a simple question: **can a model beat using the previous day's price?** The experiments compare that benchmark with ridge regression, LightGBM, PatchTST, Chronos-Bolt and, for hourly prices, LEAR.

## Start here

- [Research workflow and file guide](docs/RESEARCH_WORKFLOW.md) explains the experiments, maps the supervisor's questions to specific files, and records what is complete and what remains uncertain.
- [Code tour](docs/READING_THE_CODE.md) follows the data from raw prices to forecasts and result tables.
- [Data setup](docs/DATA_SETUP.md) explains which price files are needed and where they come from.
- [Reproduction instructions](docs/REPRODUCING.md) describe the environment and verification commands.
- [Limitations](docs/LIMITATIONS.md) explains the short later test, data provenance and publication-time assumptions.

## How the study works

For a delivery day **D**, the forecast is made at 10:00 Copenhagen time on D−1. The models use earlier day-ahead prices to forecast all intervals of D. Most locally fitted models are refitted weekly; the original LEAR comparison was refitted daily, and a separate weekly LEAR comparison matches the other models' refit schedule.

The main comparison has an earlier test and a later 23-day test. Separate winter and spring experiments add seasonal coverage. The results are retrospective, and the separately tuned seasonal experiments should not be treated as one continuous, fixed-settings test.

## Find the code and results

```text
config/                     Saved model settings used for setup
scripts/                    Original forecasting and evaluation pipeline
scripts/followup/           Seasonal experiments, comparisons and checks
docs/                       Data, methods, reproduction and limitations
tools/                      Configuration bootstrap
completion_protocol.json    Settings for the original rolling experiment
```

The main forecasting code is in `scripts/rolling_completion.py`; sample construction and PatchTST helpers are in `scripts/controlled_study.py`. The [workflow guide](docs/RESEARCH_WORKFLOW.md) lists the purpose of every follow-up script and the result files it produces.

`data/`, `results/` and `reports/` are local research artifacts excluded from Git. They contain the datasets, saved forecasts, figures and model checkpoints. A separate reproducibility package is needed to reproduce or share the saved results.

## Set up and check the project

Use Python 3.12 and first restore the frozen data and result artifacts. Then run:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -r requirements-supervisor.txt
python -m pip check
python scripts/followup/supervisor_verify.py
```

The last command checks saved forecasts and statistical calculations. It requires the local data and result files; a code-only clone does not contain them. See [reproduction instructions](docs/REPRODUCING.md) before rerunning the full experiments, which retrain models and write results.

## What to keep in mind

The later test covers only 23 days. The original historical API response bodies were not saved, and the retained data do not contain publication or revision timestamps for every observation. The analysis assumes the previous day's day-ahead prices were published before the 10:00 forecast cutoff. These points limit the conclusions; see [limitations](docs/LIMITATIONS.md).

## History

The initial repository history was retrospectively imported in 55 reconstructed stages. Its author dates organise that import and do not establish when the work was originally performed. See [history provenance](HISTORY_PROVENANCE.md). Later changes are recorded as ordinary commits when made.
