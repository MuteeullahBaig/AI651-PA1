"""Figures for the Task 2 part of the report (written to report/figures/task2/)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data import FIT_END, HORIZON, N_KNOWN, load  # noqa: E402
from experiment import RESULTS  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "report" / "figures" / "task2"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def eda():
    y, cov = load()
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), layout="constrained")
    ax = axes[0, 0]
    ax.plot(np.arange(N_KNOWN), y, lw=0.3, color="#1d4e9e")
    ax.axvspan(FIT_END, N_KNOWN, color="#b0355a", alpha=0.15, label="validation blocks (last 18 × 168)")
    ax.set(title="Full history", xlabel="time_idx", ylabel="value")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[0, 1]
    seg = slice(N_KNOWN - 4 * HORIZON, N_KNOWN)
    ax.plot(np.arange(N_KNOWN)[seg], y[seg], lw=0.8, color="#1d4e9e")
    ax.set(title="Last four 168-step blocks", xlabel="time_idx", ylabel="value")
    ax = axes[1, 0]
    centred = y - y.mean()
    f = np.fft.rfft(centred, n=2 * len(y))
    acf = np.fft.irfft(f * np.conj(f))[:len(y)]
    acf /= acf[0]
    lags = np.arange(1, 400)
    ax.plot(lags, acf[lags], color="#1d4e9e", lw=1)
    for k in (24, 48, 168, 336):
        ax.axvline(k, color="#999", ls=":", lw=0.8)
    ax.set(title="Autocorrelation (dotted: 24, 48, 168, 336)", xlabel="lag", ylabel="ACF")
    ax = axes[1, 1]
    power = np.abs(np.fft.rfft(centred)) ** 2
    k = np.arange(1, len(power))
    periods = len(y) / k
    keep = (periods >= 6) & (periods <= 20000)
    ax.loglog(periods[keep], power[1:][keep], lw=0.5, color="#1d4e9e")
    ax.axvline(24, color="#b0355a", ls=":", lw=1)
    ax.set(title="Periodogram (dotted: 24)", xlabel="period (steps, log scale)", ylabel="power")
    fig.savefig(OUT / "eda.pdf")
    fig.savefig(OUT / "eda.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0), layout="constrained")
    ax.hist(np.log1p(y), bins=80, color="#1d4e9e")
    ax.set(title="Distribution of log1p(value)", xlabel="log1p(value)", ylabel="count")
    fig.savefig(OUT / "distribution.pdf")
    plt.close(fig)


def probe():
    rows = json.loads((RESULTS / "covariate_probe.json").read_text())
    names = list(rows)
    vals = [rows[n]["block_rmse"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    colors = ["#74848f" if "except" in n else "#1d4e9e" for n in names]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set(xlabel="validation block RMSE (lower is better)", title="Linear probe: which covariates, and which portion?")
    ax.set_xlim(0, max(vals) * 1.12)
    fig.savefig(OUT / "covariate_probe.pdf")
    plt.close(fig)


if __name__ == "__main__":
    eda()
    probe()
    print("figures written to", OUT)
