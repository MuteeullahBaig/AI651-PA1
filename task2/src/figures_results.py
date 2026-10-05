"""Result figures for Task 2: ablation per block, example validation weeks, and the final forecast."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines import ridge_direct  # noqa: E402
from data import HORIZON, N_KNOWN, load, validation_origins  # noqa: E402
from experiment import RESULTS  # noqa: E402
from figures import OUT  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ARMS = {"No external data": ("abl2_none", "#74848f"), "Past values only": ("abl2_past", "#a85f00"),
        "Past + horizon values": ("s5_anchor", "#1d4e9e")}


def block_rmse(truth, pred):
    return np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2, axis=-1))


def ablation_blocks():
    y, cov = load()
    origins = validation_origins()
    truth = np.stack([y[o:o + HORIZON] for o in origins])
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    x = np.arange(len(origins))
    for i, (name, (tag, color)) in enumerate(ARMS.items()):
        runs = json.loads((RESULTS / f"{tag}.json").read_text())["runs"]
        per = np.stack([block_rmse(truth, np.array(r["val_predictions"])) for r in runs])   # [seeds, blocks]
        ax.errorbar(x + (i - 1) * 0.22, per.mean(0), yerr=[per.mean(0) - per.min(0), per.max(0) - per.mean(0)],
                    fmt="o", ms=3.5, color=color, capsize=2, lw=1, label=f"Autoformer, {name.lower()} (5 seeds, min–max)")
    ridge = np.stack(ridge_direct(y, cov, origins, with_covariates=True))
    ax.plot(x, block_rmse(truth, ridge), "x", color="#b0355a", ms=6, label="Ridge on lags + covariates (reference)")
    mean168 = np.stack([np.full(HORIZON, y[o - 168:o].mean()) for o in origins])
    ax.plot(x, block_rmse(truth, mean168), "_", color="#222", ms=10, mew=2, label="Mean of last 168 (reference)")
    ax.set(xlabel="validation block (168 steps each, in time order)", ylabel="block RMSE",
           title="Validation RMSE per block: the spread between blocks dwarfs the spread between seeds")
    ax.set_xticks(x)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.savefig(OUT / "ablation_blocks.pdf")
    plt.close(fig)


def example_weeks(blocks=(2, 9, 15)):
    y, cov = load()
    origins = validation_origins()
    runs = json.loads((RESULTS / "s5_anchor.json").read_text())["runs"]
    none_runs = json.loads((RESULTS / "abl2_none.json").read_text())["runs"]
    ridge = ridge_direct(y, cov, origins, with_covariates=True)
    fig, axes = plt.subplots(len(blocks), 1, figsize=(10, 2.3 * len(blocks)), layout="constrained", sharex=True)
    for ax, b in zip(axes, blocks):
        o = origins[b]
        t_ctx = np.arange(-168, 0)
        t_h = np.arange(HORIZON)
        ax.plot(t_ctx, y[o - 168:o], color="#74848f", lw=0.9, label="history")
        ax.plot(t_h, y[o:o + HORIZON], color="#14202a", lw=1.4, label="truth")
        for k, r in enumerate(runs):
            ax.plot(t_h, np.array(r["val_predictions"])[b], color="#1d4e9e", lw=0.8, alpha=0.7,
                    label="Autoformer with external data (5 seeds)" if k == 0 else None)
        ax.plot(t_h, np.array(none_runs[0]["val_predictions"])[b], color="#a85f00", lw=1, ls="--",
                label="Autoformer without external data (seed 0)")
        ax.plot(t_h, ridge[b], color="#b0355a", lw=1, ls=":", label="Ridge on lags + covariates")
        ax.axvline(0, color="#999", ls=":", lw=0.8)
        ax.set(ylabel="value", title=f"Validation block {b} (origin time_idx {o + 1})")
    axes[0].legend(fontsize=7, ncol=3, loc="upper left")
    axes[-1].set_xlabel("steps relative to the forecast origin")
    fig.savefig(OUT / "example_weeks.pdf")
    plt.close(fig)


def final_forecast(attempts=(("final", "attempt 1: single log-target model", "#a85f00", "--"),
                             ("ensemble5", "attempt 2: mean of 5 log-target seeds", "#74848f", ":"),
                             ("blend10", "attempt 3 (counted): 5 log + 5 raw-scale models", "#1d4e9e", "-"))):
    y, _ = load()
    fig, ax = plt.subplots(figsize=(10, 3.0), layout="constrained")
    ax.plot(np.arange(N_KNOWN - 336, N_KNOWN) + 1, y[-336:], color="#14202a", lw=1, label="last 336 observations")
    for tag, label, color, ls in attempts:
        fc = pd.read_csv(RESULTS / f"{tag}_forecast.csv")
        ax.plot(fc.time_idx, fc.value, color=color, lw=1.6 if ls == "-" else 1.1, ls=ls, label=label)
    ax.axvline(N_KNOWN + 0.5, color="#999", ls=":", lw=0.8)
    ax.set(xlabel="time_idx", ylabel="value", title="Leaderboard forecasts for time_idx 43657–43824")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(OUT / "final_forecast.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "final" in sys.argv:
        final_forecast()
    else:
        ablation_blocks()
        example_weeks()
    print("done")
