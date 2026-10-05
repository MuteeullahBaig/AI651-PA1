"""Average the forecasts of several final.py refits (one per seed) into one submission.

The handout counts an ensemble's cost as the SUM of its members' parameters and epochs.
Usage: python ensemble.py --tag ensemble5 --members final final_s1 final_s2 final_s3 final_s4
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from data import HORIZON
from experiment import RESULTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="ensemble5")
    parser.add_argument("--members", nargs="+", required=True)
    args = parser.parse_args()
    forecasts, declarations = [], []
    for member in args.members:
        forecasts.append(pd.read_csv(RESULTS / f"{member}_forecast.csv"))
        declarations.append(json.loads((RESULTS / f"{member}_declaration.json").read_text()))
    idx = forecasts[0].time_idx
    assert all((f.time_idx == idx).all() for f in forecasts) and len(idx) == HORIZON
    values = np.mean([f.value.to_numpy() for f in forecasts], axis=0)
    out = pd.DataFrame({"time_idx": idx, "value": np.round(values, 3)})
    out.to_csv(RESULTS / f"{args.tag}_forecast.csv", index=False)
    text = ", ".join(f"{v:.3f}" for v in out.value)
    assert len(text.split(",")) == HORIZON
    (RESULTS / f"{args.tag}_submission.txt").write_text(text + "\n")
    declaration = {
        "members": [{"tag": m, "seed": d["seed"], "P": d["trainable_parameters_P"], "E": d["declared_epochs_E"]}
                    for m, d in zip(args.members, declarations)],
        "trainable_parameters_P": int(sum(d["trainable_parameters_P"] for d in declarations)),
        "declared_epochs_E": int(sum(d["declared_epochs_E"] for d in declarations)),
    }
    (RESULTS / f"{args.tag}_declaration.json").write_text(json.dumps(declaration, indent=1))
    print(json.dumps(declaration, indent=1))
    print("first values:", text[:80], "...")


if __name__ == "__main__":
    main()
