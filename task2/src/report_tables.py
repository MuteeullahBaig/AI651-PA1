"""Build LaTeX table fragments for the report from the JSON logs in results/ (no numbers typed by hand)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiment import RESULTS

OUT = Path(__file__).resolve().parents[2] / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
PM = r" $\pm$ "
NL = r" \\"
BOTTOM = "\\bottomrule\n"      # written inside each file: \bottomrule after an \input breaks tabular

LABELS = {
    "s1_base": "d=32, log target (starting point)", "s1_std": "d=32, raw-scale target",
    "s1_lr3e4": r"d=32, lr $3\times10^{-4}$", "s1_std_lr3e4": r"d=32, raw scale, lr $3\times10^{-4}$",
    "s1_seq96": "d=32, context 96", "s1_seq336": "d=32, context 336",
    "s1_d16": "d=16", "s1_d64": "d=64", "s1_drop1": "d=32, dropout 0.1", "s1_enc2": "d=32, 2 encoder layers",
    "s1_k49": "d=32, moving-average kernel 49", "s1_bs128": "d=32, batch 128",
    "s2_d16_log": "d=16, log target", "s2_d16_std": "d=16, raw-scale target", "s2_d8_log": "d=8, log target",
    "s2_d8_std": "d=8, raw-scale target", "s2_d16_log_lr3e4": r"d=16, lr $3\times10^{-4}$",
    "s2_d16_log_seq336": "d=16, context 336", "s2_d16_log_enc2": "d=16, 2 encoder layers",
    "s2_d16_log_k49": "d=16, kernel 49", "s2_d16_log_f3": "d=16, factor $c=3$",
    "s3_eng": "d=16, engineered covariates", "s3_ck5": "d=16, conv covariate embedding (k=5)",
    "s3_eng_ck5": "d=16, engineered + conv (k=5)", "s3_eng_ck9": "d=16, engineered + conv (k=9)",
    "s3_noEF": "d=16, without E and F", "s3_dl2": "d=16, 2 decoder layers", "s3_lab168": "d=16, label length 168",
    "s4_d16_eng_ck5": "d=16, engineered + conv (k=5)", "s4_d16_eng_ck9": "d=16, engineered + conv (k=9)",
    "s4_d8_eng_ck5": "d=8, engineered + conv (k=5)",
    "s5_anchor": "stage-4 choice + anchoring (final)",
    "s6_tlast": "trend start = end of MA trend", "s6_tlastobs": "trend start = last observation",
    "s6_anchor_tlast": "anchoring + trend start = end of MA trend",
    "s7_decay": "trend start decays from last value to mean", "s7_anchor_decay": "anchoring + decaying trend start",
}


def load(tag):
    path = RESULTS / f"{tag}.json"
    return json.loads(path.read_text()) if path.exists() else None


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def sweep_table():
    lines = []
    for stage in ("s1", "s2", "s3", "s4", "s5", "s6", "s7"):
        rows = [(t, load(t)) for t in LABELS if t.startswith(stage + "_")]
        rows = [(t, r) for t, r in rows if r]
        if not rows:
            continue
        seeds = len(rows[0][1]["seeds"])
        lines.append(r"\multicolumn{6}{l}{\emph{Stage " + stage[1] + f" ({seeds} seeds each)" + "}}" + NL)
        best = min(r["block_rmse_mean"] for _, r in rows)
        for t, r in rows:
            mean = f"{r['block_rmse_mean']:.1f}"
            if abs(r["block_rmse_mean"] - best) < 1e-9:
                mean = r"\textbf{" + mean + "}"
            epochs = ", ".join(str(e) for e in r["best_epochs"])
            lines.append(f"{LABELS[t]} & {thousands(r['params'])} & {mean}{PM}{r['block_rmse_std']:.1f} & "
                         f"{r['block_mae_mean']:.1f} & {r['block_smape_mean']:.1f} & {epochs}{NL}")
        lines.append(r"\midrule")
    (OUT / "sweep_rows.tex").write_text("\n".join(lines[:-1]) + "\n" + BOTTOM)


def simple_rows(source, target):
    rows = json.loads((RESULTS / source).read_text())
    lines = [f"{name} & {r['block_rmse']:.1f} & {r['block_mae']:.1f} & {r['block_smape']:.1f}{NL}"
             for name, r in rows.items()]
    (OUT / target).write_text("\n".join(lines) + "\n" + BOTTOM)


def ablation_table(tags):
    runs = {name: load(tag) for name, tag in tags.items()}
    runs = {k: v for k, v in runs.items() if v}
    if len(runs) < 2:
        return
    ref_name = next(iter(runs))
    ref = {r["seed"]: r["best_metrics"]["block_rmse"] for r in runs[ref_name]["runs"]}
    lines = []
    for name, r in runs.items():
        per_seed = {x["seed"]: x["best_metrics"]["block_rmse"] for x in r["runs"]}
        seeds = sorted(per_seed)
        diffs = [ref[s] - per_seed[s] for s in seeds if s in ref]
        seed_str = ", ".join(f"{per_seed[s]:.1f}" for s in seeds)
        if name == ref_name:
            diff_str = "--"
        else:
            wins = sum(d > 0 for d in diffs)
            diff_str = f"{np.mean(diffs):.1f}{PM}{np.std(diffs, ddof=1):.1f} ({wins}/{len(diffs)})"
        lines.append(f"{name} & {thousands(r['params'])} & {r['block_rmse_mean']:.1f}{PM}{r['block_rmse_std']:.1f} & "
                     f"{r['block_mae_mean']:.1f} & {r['block_smape_mean']:.1f} & {seed_str} & {diff_str}{NL}")
    (OUT / "ablation_rows.tex").write_text("\n".join(lines) + "\n" + BOTTOM)


if __name__ == "__main__":
    sweep_table()
    simple_rows("baselines.json", "baseline_rows.tex")
    simple_rows("covariate_probe.json", "probe_rows.tex")
    if len(sys.argv) > 1:
        ablation_table({"No external data": "abl2_none", "Past values only": "abl2_past",
                        "Past and horizon values": sys.argv[1]})
    print("wrote", sorted(p.name for p in OUT.iterdir()))
