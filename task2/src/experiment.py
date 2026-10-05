"""Train/validate Autoformer configurations over several seeds and log everything to results/.

Example:
    python experiment.py --tag base --seeds 0 1 2 --set d_model=32 cov_mode=future target=log
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from autoformer import Autoformer, count_parameters
from data import (FIT_END, HORIZON, N_KNOWN, Preprocessor, WindowSampler, block_metrics, load,
                  validation_origins, COVARIATES, engineer)

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEFAULTS = dict(seq_len=168, label_len=84, d_model=32, n_heads=4, e_layers=1, d_layers=1, d_ff=64,
                kernel=25, factor=1.0, dropout=0.05, target="log", cov_mode="future",
                lr=1e-3, batch=64, stride=1, lr_decay=0.5, feature_set="base", cov_kernel=1, anchor=0, trend_init="mean",
                loss_space="model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg):
    return Autoformer(seq_len=cfg["seq_len"], label_len=cfg["label_len"], pred_len=HORIZON,
                      n_covariates=0 if cfg["cov_mode"] == "none" else n_covariates(cfg),
                      d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
                      d_layers=cfg["d_layers"], d_ff=cfg["d_ff"], kernel=cfg["kernel"],
                      factor=cfg["factor"], dropout=cfg["dropout"], cov_kernel=cfg.get("cov_kernel", 1),
                      anchor=cfg.get("anchor", 0), trend_init=cfg.get("trend_init", "mean"))


def n_covariates(cfg):
    return engineer(np.zeros((2, len(COVARIATES))), cfg.get("feature_set", "base"))[0].shape[1]


def prepare(cfg, fit_end):
    y, cov = load()
    pre = Preprocessor(target=cfg["target"], fit_end=fit_end, feature_set=cfg.get("feature_set", "base")).fit(y, cov)
    y_scaled = np.concatenate([pre.transform_y(y), np.zeros(HORIZON)])   # horizon targets unknown (never used)
    sampler = WindowSampler(y_scaled, pre.transform_covariates(cov), cfg["seq_len"], cfg["label_len"],
                            HORIZON, cfg["cov_mode"], DEVICE)
    return y, pre, sampler


@torch.no_grad()
def predict(model, sampler, pre, origins):
    model.eval()
    x_enc, m_enc, m_dec, _ = sampler.batch(origins, with_target=False)
    z = model(x_enc, m_enc, m_dec).squeeze(-1).cpu().numpy().astype(np.float64)
    return pre.inverse_y(z)


def train(cfg, seed, max_epochs, patience=None, fit_end=FIT_END, evaluate=True, log=print):
    """Returns dict(history, best_epoch, best_metrics, params, state, seconds)."""
    y, pre, sampler = prepare(cfg, fit_end)
    raw_scale = float(np.std(y[:fit_end]))          # only used when loss_space == "raw"
    origins = np.arange(cfg["seq_len"], fit_end - HORIZON + 1, cfg["stride"])
    val = validation_origins() if evaluate else None
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(1000 + seed)
    history, best, best_state, best_epoch, since = [], float("inf"), None, 0, 0
    best_pred = None
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * cfg["lr_decay"] ** (epoch - 1)
        model.train()
        order = torch.as_tensor(origins)[torch.randperm(len(origins), generator=generator)]
        losses = []
        for chunk in order.split(cfg["batch"]):
            x_enc, m_enc, m_dec, target = sampler.batch(chunk.numpy())
            out = model(x_enc, m_enc, m_dec).squeeze(-1)
            if cfg.get("loss_space", "model") == "raw" and cfg["target"] == "log":
                # Loss on the original scale so the model learns the conditional MEAN, not a median-like
                # back-transform. Values are divided by the training std of y to keep the loss near 1.
                to_raw = lambda z: torch.expm1(torch.clamp(z * pre.sd + pre.mu, max=8.0))
                loss = ((to_raw(out) - to_raw(target)) / raw_scale).square().mean()
            else:
                loss = (out - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": time.perf_counter() - started}
        if evaluate:
            pred = predict(model, sampler, pre, val)
            truth = [y[o:o + HORIZON] for o in val]
            record.update(block_metrics(truth, list(pred)))
            score = record["block_rmse"]
            if score < best - 1e-9:
                best, best_epoch, since = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
                best_pred = np.round(np.asarray(pred), 2)
            else:
                since += 1
        else:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
        history.append(record)
        log(f"  seed {seed} epoch {epoch}: loss {record['train_loss']:.4f}"
            + (f" | val block RMSE {record['block_rmse']:.2f} MAE {record['block_mae']:.2f} sMAPE {record['block_smape']:.1f}"
               if evaluate else "") + f" | {record['seconds']:.0f}s")
        if evaluate and patience is not None and since >= patience:
            break
    model.load_state_dict(best_state)
    out = {"history": history, "best_epoch": best_epoch, "params": count_parameters(model),
           "epochs_run": len(history), "seconds": time.perf_counter() - started, "model": model,
           "preprocessor": pre, "sampler": sampler, "y": y}
    if evaluate:
        out["best_metrics"] = {k: v for k, v in history[best_epoch - 1].items() if k != "epoch"}
        out["val_predictions"] = best_pred
    return out


def parse_value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = parse_value(value)
    RESULTS.mkdir(exist_ok=True)
    runs = []
    print(f"[{args.tag}] {cfg}", flush=True)
    for seed in args.seeds:
        result = train(cfg, seed, args.max_epochs, args.patience, log=lambda s: print(s, flush=True))
        runs.append({"seed": seed, "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
                     "params": result["params"], "seconds": result["seconds"],
                     "best_metrics": result["best_metrics"], "history": result["history"],
                     "val_predictions": result["val_predictions"].tolist()})
    rm = [r["best_metrics"]["block_rmse"] for r in runs]
    summary = {"tag": args.tag, "config": cfg, "seeds": args.seeds, "params": runs[0]["params"],
               "block_rmse_mean": float(np.mean(rm)), "block_rmse_std": float(np.std(rm, ddof=1)) if len(rm) > 1 else 0.0,
               "block_mae_mean": float(np.mean([r["best_metrics"]["block_mae"] for r in runs])),
               "block_smape_mean": float(np.mean([r["best_metrics"]["block_smape"] for r in runs])),
               "best_epochs": [r["best_epoch"] for r in runs], "runs": runs}
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"[{args.tag}] params {summary['params']} | block RMSE {summary['block_rmse_mean']:.2f} ± "
          f"{summary['block_rmse_std']:.2f} | MAE {summary['block_mae_mean']:.2f} | sMAPE "
          f"{summary['block_smape_mean']:.1f} | best epochs {summary['best_epochs']}", flush=True)


if __name__ == "__main__":
    main()
