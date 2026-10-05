"""Produce the leaderboard forecast for time_idx 43657..43824.

The chosen configuration is refit on the whole history (all 43,656 steps; scaling statistics refitted on all of
it) for a fixed number of epochs chosen on the validation blocks, then forecasts the hidden block from the last
seq_len observations and the external file. Writes results/final_forecast.csv, results/final_submission.txt
(168 comma-separated values in chronological order) and results/final_declaration.json (P and E).

Example:
    python final.py --tag final --epochs 3 --validation-epochs 5 --seed 0 --set d_model=16 d_ff=32
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from data import DATA, HORIZON, N_KNOWN
from experiment import DEFAULTS, RESULTS, parse_value, predict, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="final")
    parser.add_argument("--epochs", type=int, required=True, help="refit epochs on the full history")
    parser.add_argument("--validation-epochs", type=int, required=True,
                        help="epochs run by the validation run that chose --epochs (declared in E as well)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = parse_value(value)

    result = train(cfg, args.seed, args.epochs, fit_end=N_KNOWN, evaluate=False,
                   log=lambda s: print(s, flush=True))
    model, pre, sampler = result["model"], result["preprocessor"], result["sampler"]
    forecast = predict(model, sampler, pre, np.array([N_KNOWN]))[0]
    assert forecast.shape == (HORIZON,) and np.isfinite(forecast).all() and (forecast >= 0).all()

    test = pd.read_csv(DATA / "student_test.csv")
    assert test.time_idx.tolist() == list(range(N_KNOWN + 1, N_KNOWN + HORIZON + 1))
    out = pd.DataFrame({"time_idx": test.time_idx, "value": np.round(forecast, 3)})
    out.to_csv(RESULTS / f"{args.tag}_forecast.csv", index=False)
    text = ", ".join(f"{v:.3f}" for v in out.value)
    assert len(text.split(",")) == HORIZON
    (RESULTS / f"{args.tag}_submission.txt").write_text(text + "\n")
    declaration = {
        "config": cfg, "seed": args.seed,
        "trainable_parameters_P": result["params"],
        "epochs_refit_on_full_history": args.epochs,
        "epochs_of_validation_run_that_chose_them": args.validation_epochs,
        "declared_epochs_E": args.epochs + args.validation_epochs,
        "refit_seconds": result["seconds"],
        "forecast_summary": {"min": float(forecast.min()), "median": float(np.median(forecast)),
                             "max": float(forecast.max()), "first_5": [float(v) for v in forecast[:5]]},
        "last_observed_values": [float(v) for v in pre.inverse_y(sampler.y[N_KNOWN - 5:N_KNOWN].cpu().numpy())],
    }
    (RESULTS / f"{args.tag}_declaration.json").write_text(json.dumps(declaration, indent=1))
    torch.save({"state": model.state_dict(), "config": cfg}, RESULTS / f"{args.tag}_model.pt")
    print(json.dumps({k: v for k, v in declaration.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
