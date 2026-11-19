"""Export validated normalized price series without mixing time resolutions."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from rolling_completion import extended

R = Path(__file__).resolve().parents[2]
O = R / "results/supervisor_response/processed"
O.mkdir(parents=True, exist_ok=True)
meta = {}
for mode in ["hourly", "quarter_hour"]:
    for area in ["DK1", "DK2"]:
        s = extended(area, mode)
        p = O / f"{mode}_{area}.csv.gz"
        local = s.index.tz_convert("Europe/Copenhagen")
        f = pd.DataFrame(
            {
                "time_utc": s.index,
                "time_copenhagen": local,
                "price_area": area,
                "price_eur_per_mwh": s.to_numpy(),
            }
        )
        f.to_csv(p, index=False, compression="gzip")
        check = pd.read_csv(p, parse_dates=["time_utc"])
        assert len(check) == len(s)
        assert pd.DatetimeIndex(check.time_utc).equals(s.index)
        np.testing.assert_allclose(check.price_eur_per_mwh, s.values, rtol=0, atol=1e-9)
        meta[p.name] = {
            "rows": len(f),
            "first_utc": str(s.index.min()),
            "last_utc": str(s.index.max()),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "description": "Hourly: historical hourly plus mean of 4 quarter-hour prices. Quarter-hour: genuine quarter-hour data only.",
        }
(O / "manifest.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
