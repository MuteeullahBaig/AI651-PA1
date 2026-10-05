# AI651 · Assignment 1 (DL4STG-PA1)

Deep Learning for Space, Time and Graphs, LUMS, Fall 2026.

| Folder | Contents |
|---|---|
| `task1/` | The completed and executed notebook (`Assignment1.ipynb`, `PA1_PRESET=full`); the supplied `harness/`; `predictions_before_running.md` (Responses 1A and 4a, written before any run); `run_notebook.py` (headless runner); `analysis_slow_component.py` (supplementary validation analysis); `logs/` |
| `task2/src/` | A compact Autoformer (`autoformer.py`), data and preprocessing (`data.py`), the training and validation runner (`experiment.py`), sweeps (`sweep.sh`), baselines, the covariate probe, figures, and the final forecast (`final.py`) |
| `task2/results/` | JSON logs for every run (per seed, per epoch), sweep logs, and the final forecast and declaration |
| `report/` | LaTeX source (`main.tex`, `task1.tex`, `task2.tex`, …), figures, and the compiled `main.pdf` |
| `AI_USAGE.md` | Log of generative-AI use (prompts, outputs, edits) |

## Environment

Python 3.11, with PyTorch 2.5 (CUDA optional), NumPy, pandas, Matplotlib, nbclient and JupyterLab. `task1/requirements.txt` is the course's list, and Task 2 needs nothing beyond it.

## Reproduce

```bash
# Task 1: full-preset run of the whole notebook (about 5 minutes on an RTX 4070 Laptop GPU)
cd task1
python run_notebook.py full Assignment1.ipynb

# Task 2: place the three course CSV files in task2/data/, then:
cd task2/src
python baselines.py
python covariate_probe.py
bash sweep.sh "0 1 2" "<tag>:<key=value ...>"          # any configuration, e.g. "s2_d16_log:d_model=16 d_ff=32"
python final.py --tag final --epochs <E_refit> --validation-epochs <E_val> --seed 0 --set <chosen configuration>
```

`final.py` prints the trainable parameter count P and the declared epochs E, and writes
`results/final_submission.txt`: 168 comma-separated values for `time_idx` 43657 to 43824, in chronological order.

Leaderboard attempts:

- **Attempt 1, the counted best:** `final_submission.txt`. P = 8,577, E = 6, RMSE 101.10.
- **Attempt 2:** `ensemble5_submission.txt`, the mean of seeds 0–4. It is built with `python ensemble.py --tag ensemble5 --members final final_s1 final_s2 final_s3 final_s4` after running `final.py` for each seed. P = 42,885, E = 31, RMSE 106.56.

## What is in `task2/results/`

| Files | Contents |
|---|---|
| `baselines.json`, `covariate_probe.json` | Reference forecasts and the linear covariate probe (report §2.4) |
| `s1_*` … `s7_*`, `sweep1.log` … `sweep7.log` | The seven search stages, per seed and per epoch (report §2.5) |
| `s5_anchor.json`, `abl2_past.json`, `abl2_none.json`, `ablation2.log` | The final configuration and the two other arms of the covariate ablation (report §2.6) |
| `abl_future.json`, `abl_past.json`, `abl_none.json`, `ablation.log` | An earlier ablation run, before anchoring was added; superseded by the row above and kept for transparency |
| `final_*`, `final_s1_*` … `final_s4_*`, `ensemble5_*` | The full-history refits, the submitted forecasts, and the P/E declarations (report §2.7) |

## Data

The course's Task 2 CSV files belong in `task2/data/`. They are not redistributed in this repository.
