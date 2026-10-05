"""Supplementary validation analysis for Response 4(a): errors by how prominent the slow component is.

Loads the full-preset checkpoints written by the notebook run (no retraining) and splits established-station
validation windows into terciles of slow-component change across the 96-sample context
(the harness's `slow_change`, the peak-to-peak drift inside each context). Validation data only.
"""
import json
import math
import sys
from pathlib import Path

import nbformat
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "harness"))
import design_controls as design  # noqa: E402

nb = nbformat.read(here / "Assignment1.ipynb", as_version=4)
namespace = {"torch": torch, "nn": nn, "F": F, "math": math, "design": design}
for index in (6, 13, 19, 22, 25):           # class and function definitions only
    exec(nb.cells[index].source, namespace)
source15 = nb.cells[15].source
exec("class " + source15.split("\nclass ", 1)[1].split("assert design")[0], namespace)
exec(nb.cells[29].source.split("assert design")[0], namespace)
design.register_components(raw_forecaster_type=namespace["RawAttentionForecaster"],
                           decomposition_type=namespace["SeriesDecomposition"],
                           decomposed_attention_forecaster_type=namespace["DecomposedAttentionForecaster"],
                           delay_scores=namespace["delay_scores"], aggregate=namespace["aggregate_delays"],
                           forecaster_type=namespace["AutoformerInspiredForecaster"])
study = design.Study("full", 0)
names = ["Raw Attention", "Raw delay mixer", "Attention + decomposition", "Autoformer-inspired"]
for name in names:
    study.train(name)                         # loads the cached full-preset checkpoint
part = study.region("validation")
keep = study.mask("established", "validation")
slow = part["slow_change"][keep]
edges = np.quantile(slow, [1 / 3, 2 / 3])
groups = {"low drift": slow <= edges[0], "middle": (slow > edges[0]) & (slow <= edges[1]), "high drift": slow > edges[1]}
rows = {}
predictions = {n: study.neural_prediction(n, "validation")[keep] for n in names}
predictions["Period-routed ridge"] = study.linear_prediction("routed", "validation")[keep]
target = part["target"][keep]
for name, pred in predictions.items():
    rows[name] = {g: float(np.sqrt(np.mean((pred[m] - target[m]) ** 2))) for g, m in groups.items()}
print("slow-change tercile edges (m/s^2):", np.round(edges, 2), "| windows per group:", [int(m.sum()) for m in groups.values()])
for name, r in rows.items():
    print(f"{name:>27}: " + "  ".join(f"{g} {v:.3f}" for g, v in r.items()))
(here / "results" / "supplementary_slow_component.json").write_text(json.dumps({"edges": edges.tolist(), "rmse": rows}, indent=1))
