"""Train an isolated LightGBM candidate and save predictions plus optional boosters."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import sys
import json
import gzip
import numpy as np
from lightgbm import LGBMRegressor

z = np.load(sys.argv[1])
cfg = json.loads(sys.argv[3])
pred = []
models = []
for h in range(z["y"].shape[1]):
    m = LGBMRegressor(
        n_estimators=cfg["trees"],
        num_leaves=cfg["leaves"],
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
np.save(sys.argv[2], np.stack(pred, axis=1))
if len(sys.argv) > 4:
    with gzip.open(sys.argv[4], "wt") as f:
        json.dump(
            {
                "format": "LightGBM booster strings by forecast step",
                "settings": cfg,
                "models": models,
            },
            f,
        )
