"""Reference forecasts scored on the same 18 rolling validation blocks as the Autoformer runs.

These are not submissions (only an Autoformer may be submitted); they show whether a model earns its cost.
"""
from __future__ import annotations

import json

import numpy as np

from data import FIT_END, HORIZON, Preprocessor, block_metrics, load, validation_origins
from experiment import RESULTS


def seasonal_naive(history, season):
    last = history[-season:]
    return np.array([last[h % season] for h in range(HORIZON)])


def daily_profile(history, period=24, days=7):
    recent = history[-period * days:].reshape(days, period).mean(0)
    return np.array([recent[h % period] for h in range(HORIZON)])


def ridge_direct(y, cov, origins, lag=168, lam=1.0, with_covariates=False, stride=4):
    """Direct 168-output ridge on log1p-scaled lags, optionally with horizon covariates (linear)."""
    pre = Preprocessor(target="log").fit(y, cov)
    ys, cs = pre.transform_y(y), pre.transform_covariates(cov)

    def features(o):
        parts = [ys[o - lag:o]]
        if with_covariates:
            parts.append(cs[o:o + HORIZON, :4].reshape(-1))     # continuous + wind-counter columns over the horizon
            parts.append(cs[o - 24:o, :4].reshape(-1))
        return np.concatenate(parts)

    train_origins = np.arange(lag, FIT_END - HORIZON + 1, stride)
    X = np.stack([features(o) for o in train_origins])
    Z = np.stack([ys[o:o + HORIZON] for o in train_origins])
    mean_x, mean_z = X.mean(0), Z.mean(0)
    Xc, Zc = X - mean_x, Z - mean_z
    W = np.linalg.solve(Xc.T @ Xc / len(X) + lam * np.eye(X.shape[1]), Xc.T @ Zc / len(X))
    preds = [(features(o) - mean_x) @ W + mean_z for o in origins]
    return [pre.inverse_y(p) for p in preds]


def main():
    y, cov = load()
    origins = validation_origins()
    truth = [y[o:o + HORIZON] for o in origins]
    forecasts = {
        "Persistence (last value)": [np.full(HORIZON, y[o - 1]) for o in origins],
        "Mean of last 24": [np.full(HORIZON, y[o - 24:o].mean()) for o in origins],
        "Mean of last 168": [np.full(HORIZON, y[o - 168:o].mean()) for o in origins],
        "Seasonal naive, 24": [seasonal_naive(y[:o], 24) for o in origins],
        "Seasonal naive, 168": [seasonal_naive(y[:o], 168) for o in origins],
        "7-day mean daily profile": [daily_profile(y[:o]) for o in origins],
        "Ridge on 168 lags": ridge_direct(y, cov, origins),
        "Ridge on 168 lags + covariates": ridge_direct(y, cov, origins, with_covariates=True),
    }
    rows = {}
    for name, pred in forecasts.items():
        m = block_metrics(truth, pred)
        rows[name] = {k: m[k] for k in ("block_rmse", "block_mae", "block_smape", "pooled_rmse")}
        print(f"{name:>32}: block RMSE {m['block_rmse']:7.2f} | MAE {m['block_mae']:6.2f} | "
              f"sMAPE {m['block_smape']:5.1f} | pooled RMSE {m['pooled_rmse']:7.2f}")
    (RESULTS / "baselines.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
