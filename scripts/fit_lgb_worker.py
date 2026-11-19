"""Isolate LightGBM's native runtime from PyTorch on macOS."""

import sys
import numpy as np
from lightgbm import LGBMRegressor

z = np.load(sys.argv[1])
pred = []
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
np.save(sys.argv[2], np.stack(pred, axis=1))
