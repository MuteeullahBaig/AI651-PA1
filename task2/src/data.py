"""Loading, leakage-safe preprocessing and window batching for the Task 2 series."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path(__file__).resolve().parents[1] / "data"
N_KNOWN = 43_656          # observed history
HORIZON = 168             # pred_len required by the leaderboard
N_VAL_BLOCKS = 18         # rolling-origin validation blocks at the end of the history
FIT_END = N_KNOWN - N_VAL_BLOCKS * HORIZON        # 40_632: model fitting only sees targets before this
CONTINUOUS = ["feature_A", "feature_B", "feature_C"]
COUNTERS = ["feature_D", "feature_E", "feature_F"]      # running totals that reset
CATEGORY = ["feature_G", "feature_H", "feature_I", "feature_J"]  # one-hot, exactly one active
COVARIATES = CONTINUOUS + COUNTERS + CATEGORY


def engineer(cov, feature_set="base"):
    """Return (matrix, kinds) with kinds in {"cont", "counter", "binary"} for each column.

    base : the ten raw variables.
    noEF : drop E and F (the linear probe found no signal in them).
    eng  : base + log1p increment of the D counter (D restarts whenever the G-J category changes) +
           trailing 6- and 24-step means of A, B, C and that increment (computed at every step of the file).
    """
    kinds = ["cont"] * 3 + ["counter"] * 3 + ["binary"] * 4
    if feature_set == "base":
        return cov.copy(), kinds
    if feature_set == "noEF":
        keep = [0, 1, 2, 3, 6, 7, 8, 9]
        return cov[:, keep].copy(), [kinds[i] for i in keep]
    if feature_set != "eng":
        raise ValueError(feature_set)
    d_counter = cov[:, 3]
    category = cov[:, 6:].argmax(1)
    restart = np.r_[True, np.diff(category) != 0]
    increment = np.clip(np.where(restart, d_counter, np.r_[d_counter[0], np.diff(d_counter)]), 0, None)
    signals = np.column_stack([cov[:, 0], cov[:, 1], cov[:, 2], np.log1p(increment)])
    csum = np.cumsum(np.vstack([np.zeros((1, signals.shape[1])), signals]), axis=0)
    idx = np.arange(1, len(cov) + 1)
    extra = [np.log1p(increment)[:, None]]
    for window in (6, 24):
        lo = np.maximum(idx - window, 0)
        extra.append((csum[idx] - csum[lo]) / (idx - lo)[:, None])
    extra = np.hstack(extra)
    return np.column_stack([cov, extra]), kinds + ["cont"] * extra.shape[1]


def load():
    train = pd.read_csv(DATA / "student_train.csv")
    external = pd.read_csv(DATA / "optional_external_data.csv")
    assert (np.diff(train.time_idx) == 1).all() and len(train) == N_KNOWN
    assert len(external) == N_KNOWN + HORIZON and (external.time_idx.to_numpy() == np.arange(1, N_KNOWN + HORIZON + 1)).all()
    return train.value.to_numpy(np.float64), external[COVARIATES].to_numpy(np.float64)


def validation_origins(n_blocks: int = N_VAL_BLOCKS, fit_end: int = FIT_END, step: int = HORIZON):
    """0-based origins: the forecast for origin t covers indices t .. t + 167."""
    return np.array([fit_end + step * j for j in range(n_blocks)])


@dataclass
class Preprocessor:
    """Statistics are fitted on indices [0, fit_end) only."""
    target: str = "log"          # "log": z-score of log1p(y); "standard": z-score of y
    fit_end: int = FIT_END
    feature_set: str = "base"

    def fit(self, y, covariates):
        base = self._target_base(y[:self.fit_end])
        self.mu, self.sd = float(base.mean()), float(base.std())
        _, self.kinds = engineer(covariates[:2], self.feature_set)
        cov = self._covariate_base(engineer(covariates, self.feature_set)[0][:self.fit_end])
        self.cmu, self.csd = cov.mean(0), cov.std(0) + 1e-8
        binary = np.array([k == "binary" for k in self.kinds])
        self.cmu[binary], self.csd[binary] = 0.0, 1.0          # keep one-hot columns as 0/1
        return self

    def _target_base(self, y):
        return np.log1p(y) if self.target == "log" else y

    def _covariate_base(self, c):
        c = c.copy()
        counters = [i for i, k in enumerate(self.kinds) if k == "counter"]
        c[:, counters] = np.log1p(c[:, counters])
        return c

    def transform_y(self, y):
        return (self._target_base(y) - self.mu) / self.sd

    def inverse_y(self, z):
        base = z * self.sd + self.mu
        out = np.expm1(base) if self.target == "log" else base
        return np.clip(out, 0.0, None)

    def transform_covariates(self, c):
        return (self._covariate_base(engineer(c, self.feature_set)[0]) - self.cmu) / self.csd


class WindowSampler:
    """Builds batches directly on the device by index arithmetic (no per-sample Python work)."""

    def __init__(self, y_scaled, cov_scaled, seq_len, label_len, pred_len, covariate_mode, device):
        self.y = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)
        self.c = torch.as_tensor(cov_scaled, dtype=torch.float32, device=device)
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.mode = covariate_mode           # "none" | "past" | "future"
        self.device = device

    def batch(self, origins, with_target=True):
        o = torch.as_tensor(origins, device=self.device).view(-1, 1)
        enc_idx = o + torch.arange(-self.seq_len, 0, device=self.device).view(1, -1)
        dec_idx = o + torch.arange(-self.label_len, self.pred_len, device=self.device).view(1, -1)
        x_enc = self.y[enc_idx].unsqueeze(-1)
        marks_enc = marks_dec = None
        if self.mode != "none":
            marks_enc = self.c[enc_idx]
            marks_dec = self.c[dec_idx]
            if self.mode == "past":          # horizon covariates treated as unknown
                marks_dec = marks_dec.clone()
                marks_dec[:, self.label_len:] = 0.0
        target = None
        if with_target:
            tgt_idx = o + torch.arange(self.pred_len, device=self.device).view(1, -1)
            target = self.y[tgt_idx]
        return x_enc, marks_enc, marks_dec, target


def metrics(truth, pred):
    """RMSE, MAE, sMAPE (0-200 scale) pooled over all points."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    err = pred - truth
    denom = np.abs(truth) + np.abs(pred)
    smape = 100 * np.mean(np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0))
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))), "smape": float(smape)}


def block_metrics(truth_blocks, pred_blocks):
    """Mean over blocks of each block's metric (a block mirrors one leaderboard submission) plus pooled."""
    per = [metrics(t, p) for t, p in zip(truth_blocks, pred_blocks)]
    out = {f"block_{k}": float(np.mean([m[k] for m in per])) for k in ("rmse", "mae", "smape")}
    out.update({f"pooled_{k}": v for k, v in metrics(np.concatenate(truth_blocks), np.concatenate(pred_blocks)).items()})
    out["block_rmse_list"] = [m["rmse"] for m in per]
    return out
