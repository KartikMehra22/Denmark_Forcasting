from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "results/completion"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

models = ["previous_day", "previous_week", "ridge", "lightgbm", "patchtst", "chronos"]
metrics = []
intervals = []
regimes = []
for mode in ["hourly", "quarter_hour"]:
    for area in ["DK1", "DK2"]:
        p = pd.read_csv(O / f"{mode}_{area}_predictions.csv", parse_dates=["time_utc"])
        cols = models + (["lear"] if "lear" in p else [])
        assert np.isfinite(p.select_dtypes("number").values).all()
        assert not p.time_utc.duplicated().any()
        for split, g in p.groupby("split"):
            base = (g.previous_day - g.actual).abs().mean()
            for model in cols:
                e = g[model] - g.actual
                metrics.append(
                    dict(
                        mode=mode,
                        area=area,
                        split=split,
                        model=model,
                        n=len(g),
                        days=g.date.nunique(),
                        MAE=e.abs().mean(),
                        RMSE=np.sqrt((e**2).mean()),
                        improvement_pct=100 * (1 - e.abs().mean() / base),
                    )
                )
            for name, lo, hi in [
                ("raw", "raw_lower", "raw_upper"),
                ("calibrated", "cal_lower", "cal_upper"),
            ]:
                intervals.append(
                    dict(
                        mode=mode,
                        area=area,
                        split=split,
                        interval=name,
                        n=len(g),
                        coverage=((g.actual >= g[lo]) & (g.actual <= g[hi])).mean(),
                        mean_width=(g[hi] - g[lo]).mean(),
                    )
                )
            for regime, mask in [
                ("negative", g.actual < 0),
                ("upper_5pct", g.actual >= g.actual.quantile(0.95)),
            ]:
                for model in cols:
                    e = g.loc[mask, model] - g.loc[mask, "actual"]
                    regimes.append(
                        dict(
                            mode=mode,
                            area=area,
                            split=split,
                            regime=regime,
                            model=model,
                            n=len(e),
                            MAE=e.abs().mean(),
                        )
                    )
m = pd.DataFrame(metrics)
u = pd.DataFrame(intervals)
m.to_csv(O / "metrics.csv", index=False)
u.to_csv(O / "intervals.csv", index=False)
pd.DataFrame(regimes).to_csv(O / "regimes.csv", index=False)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axs = plt.subplots(1, 2, figsize=(10.8, 3.5), layout="constrained")
for ax, mode in zip(axs, ["hourly", "quarter_hour"]):
    g = m[(m["mode"] == mode) & (m.split == "later_locked")]
    names = cols if mode == "hourly" else models
    names = [
        n
        for n in ["previous_day", "ridge", "lightgbm", "patchtst", "chronos", "lear"]
        if n in set(g.model)
    ]
    for area, offset, color in [("DK1", -0.17, "#217b7e"), ("DK2", 0.17, "#879db2")]:
        vals = g[g.area == area].set_index("model").loc[names, "MAE"]
        ax.barh(
            np.arange(len(names)) + offset, vals, height=0.3, label=area, color=color
        )
    ax.set_yticks(
        range(len(names)),
        [
            {
                "previous_day": "Previous day",
                "ridge": "Ridge",
                "lightgbm": "LightGBM",
                "patchtst": "PatchTST",
                "chronos": "Chronos",
                "lear": "LEAR",
            }[n]
            for n in names
        ],
    )
    ax.invert_yaxis()
    ax.set_title("Hourly" if mode == "hourly" else "Native 15 minute")
    ax.set_xlabel("MAE in EUR/MWh")
    ax.legend(fontsize=8)
fig.savefig(O / "later_comparison.png", dpi=220)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(10.8, 3.1), layout="constrained", sharey=True)
for ax, mode in zip(axs, ["hourly", "quarter_hour"]):
    g = u[(u["mode"] == mode) & (u.split == "later_locked")]
    for area, offset, color in [("DK1", -0.16, "#217b7e"), ("DK2", 0.16, "#879db2")]:
        vals = (
            g[g.area == area]
            .set_index("interval")
            .loc[["raw", "calibrated"], "coverage"]
            * 100
        )
        ax.bar(np.arange(2) + offset, vals, width=0.3, color=color, label=area)
        for i, v in enumerate(vals):
            ax.text(i + offset, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    ax.axhline(80, color="#ab6c35", ls="--", lw=1)
    ax.set_xticks([0, 1], ["Raw interval", "Calibrated interval"])
    ax.set_ylim(0, 104)
    ax.set_title("Hourly" if mode == "hourly" else "Native 15 minute")
axs[0].set_ylabel("Observed interval coverage in percent")
axs[1].legend(fontsize=8)
fig.savefig(O / "calibration_coverage.png", dpi=220)
plt.close(fig)
print(m[m.split == "later_locked"].to_string(index=False))
print(u[u.split == "later_locked"].to_string(index=False))
