"""Rebuild per-horizon boosters, check predictions against saved results, persist text models."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import sys
import gzip
import json
import numpy as np
from lightgbm import LGBMRegressor

z = np.load(sys.argv[1])
pred = []
models = []
for h in range(z["y"].shape[1]):
    m = LGBMRegressor(
        n_estimators=80,
        num_leaves=7,
        min_child_samples=20,
        learning_rate=0.05,
        objective="regression_l1",
        verbosity=-1,
        n_jobs=1,
        random_state=42,
    )
    m.fit(z["x"], z["y"][:, h])
    pred.append(m.predict(z["ex"]))
    models.append(m.booster_.model_to_string())
pred = np.stack(pred, axis=1)
np.save(sys.argv[2], pred)
with gzip.open(sys.argv[3], "wt", encoding="utf8") as f:
    json.dump(
        {
            "format": "LightGBM booster model strings per forecast horizon",
            "models": models,
        },
        f,
    )
