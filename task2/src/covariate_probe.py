"""Linear probe of the external file: which variables and which portions (past vs horizon) carry signal.

Direct 168-output ridge on log1p-scaled target lags, with different covariate blocks added. Scored on the
same 18 validation blocks as everything else. This is a cheap screen that guides the Autoformer ablation;
it is not the ablation itself.
"""
from __future__ import annotations

import json

import numpy as np

from data import COVARIATES, FIT_END, HORIZON, Preprocessor, block_metrics, load, validation_origins
from experiment import RESULTS

IDX = {name: i for i, name in enumerate(COVARIATES)}
GROUPS = {"A,B,C": ["feature_A", "feature_B", "feature_C"], "D": ["feature_D"], "E,F": ["feature_E", "feature_F"],
          "G-J": ["feature_G", "feature_H", "feature_I", "feature_J"]}


def run(y, cov, cols_past, cols_future, lag=168, past_len=24, lam=1.0, stride=4):
    pre = Preprocessor(target="log").fit(y, cov)
    ys, cs = pre.transform_y(y), pre.transform_covariates(cov)
    ip, iff = [IDX[c] for c in cols_past], [IDX[c] for c in cols_future]

    def features(o):
        parts = [ys[o - lag:o]]
        if ip:
            parts.append(cs[o - past_len:o][:, ip].reshape(-1))
        if iff:
            parts.append(cs[o:o + HORIZON][:, iff].reshape(-1))
        return np.concatenate(parts)

    train = np.arange(lag, FIT_END - HORIZON + 1, stride)
    X = np.stack([features(o) for o in train])
    Z = np.stack([ys[o:o + HORIZON] for o in train])
    mx, mz = X.mean(0), Z.mean(0)
    W = np.linalg.solve((X - mx).T @ (X - mx) / len(X) + lam * np.eye(X.shape[1]), (X - mx).T @ (Z - mz) / len(X))
    origins = validation_origins()
    preds = [pre.inverse_y((features(o) - mx) @ W + mz) for o in origins]
    return block_metrics([y[o:o + HORIZON] for o in origins], preds)


def main():
    y, cov = load()
    everything = COVARIATES
    settings = {
        "no covariates": ([], []),
        "past 24 steps only (all variables)": (everything, []),
        "horizon only (all variables)": ([], everything),
        "past + horizon (all variables)": (everything, everything),
    }
    for group, cols in GROUPS.items():
        settings[f"past + horizon, only {group}"] = (cols, cols)
    for group, cols in GROUPS.items():
        rest = [c for c in everything if c not in cols]
        settings[f"past + horizon, all except {group}"] = (rest, rest)
    rows = {}
    for name, (past, future) in settings.items():
        m = run(y, cov, past, future)
        rows[name] = {k: m[k] for k in ("block_rmse", "block_mae", "block_smape")}
        print(f"{name:>38}: block RMSE {m['block_rmse']:6.2f} | MAE {m['block_mae']:6.2f} | sMAPE {m['block_smape']:5.1f}", flush=True)
    (RESULTS / "covariate_probe.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
