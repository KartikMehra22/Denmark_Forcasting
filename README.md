# Denmark electricity price forecasting

Python implementation of day-ahead electricity price experiments for DK1 and DK2.

## History provenance

This repository was retrospectively imported in 55 reconstructed stages. Author dates organise the import from 26 October to 18 November 2026; committer dates record the actual import. They are not evidence of original execution dates. Read [HISTORY_PROVENANCE.md](HISTORY_PROVENANCE.md).

## Included

All original top-level Python sources, pinned dependencies, the locked completion protocol, model-only settings and reproduction documentation. The raw source files are imported byte for byte.

## Running

1. Create a Python 3.12 virtual environment and install requirements-lock.txt.
2. Restore the required raw data files using [the data setup guide](docs/DATA_SETUP.md). Frozen snapshots are required for exact reproduction.
3. Run python tools/bootstrap_settings.py on a fresh checkout.
4. Run DENMARK_ALLOW_MODEL_DOWNLOAD=1 python scripts/run_completion.py for the first model download.

See [reproduction details](docs/REPRODUCING.md), [limitations](docs/LIMITATIONS.md) and [sources](docs/SOURCES.md).

## Publication scope

This is a code-only public repository. Datasets, API responses, fitting arrays, forecasts, checkpoints and reports remain in the local project and are not in this Git history. They require separate approval before public publication.

The professor's latest requests for experimental clarification, matched comparisons and seasonal significance testing are follow-up work. This code import does not claim that those requests have already been completed.
