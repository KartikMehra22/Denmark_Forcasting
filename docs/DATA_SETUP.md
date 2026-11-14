# Data setup

This public repository contains code and configuration only. Raw datasets, API snapshots, processed fitting arrays, saved forecasts and model weights are intentionally untracked. Existing frozen research artifacts are retained locally in the Denmark project and its supporting archive.

The original scripts expect four price CSVs under data/: dayahead_DK1.csv, dayahead_DK2.csv, elspot_DK1_history.csv, elspot_DK2_history.csv. baselines.py also expects production_consumption.csv and production_consumption_history.csv.

fetch_eds.py retrieves Energinet data and preserves existing CSVs unless --force is used. A new download uses current data and is not the exact historical snapshot. To reproduce historical numbers exactly, obtain the frozen files and verify their published experiment hashes from the separately retained research artifacts. Do not replace original snapshots silently.

fetch_later.py retrieves the period fixed in completion_protocol.json and caches raw API responses under data/later_locked/. It respects API rate-limit responses.
