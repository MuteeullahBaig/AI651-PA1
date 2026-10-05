# Generative-AI usage log

The course policy requires the prompt, the AI output, and the edits I made. This log is the source for the
report's appendix. Tool: Claude (Anthropic), used as a coding and writing assistant through Claude Code in the
desktop app, with access to the assignment folder, on 25–26 September 2026.

| # | My prompt | What the AI produced | What I kept, checked or changed |
|---|---|---|---|
| 1 | "Go through the assignment resources and prepare the 5 iteration study guide … teach me about the background knowledge required … give me an estimate on how long will this assignment take?" | A five-page interactive study guide with practice notebooks (`Learning Material/`), and a time estimate of 27–47 h. | Study material only; not submitted. |
| 2 | "Lets proceed with attempting the actual assignment … develop the plan and work on it … do as you think is the best … I'll learn once we've completed the assignment." | All of the work below. | *To fill in after review.* |

## What the AI did for prompt 2

**Task 1**
- Wrote the Response 1A and 4(a) predictions before any cell ran (`task1/predictions_before_running.md`).
- Implemented `RawAttentionForecaster.forward`, `SeriesDecomposition.forward` and `aggregate_delays`, and checked them on the smoke preset.
- Ran the full preset headlessly (`task1/run_notebook.py`).
- Chose both deployment models (period-routed ridge) from the validation-only summary, before Output 4.3 existed, and wrote the choice into the notebook.
- Re-ran the whole notebook fresh with the checkpoint cache cleared, so all reported numbers come from one run.
- Added a supplementary validation-only analysis for Response 4(a) (`task1/analysis_slow_component.py`). It showed the 4(a) prediction was wrong.
- Drafted Responses 1A–4.

**Task 2**
- Explored the data.
- Wrote a compact Autoformer from the paper and the official repository's structure (not copied code), with a rolling-origin validation over 18 blocks and leakage-safe preprocessing.
- Built baselines and a linear covariate probe.
- Ran a seven-stage seeded search (all logged in `task2/results/`), including the negative results.
- Ran the required covariate ablation over 5 seeds.
- Refit the final model and wrote the forecast, with P = 8,577 and E = 6.

**Report**
- Wrote the LaTeX report (`report/`). It is compiled with Tectonic, installed in a separate conda environment named `latex`.

## Leaderboard (26 September 2026)

- **Attempt 1.** I submitted it myself. RMSE 101.10, score 101.1060, P = 8,577, E = 6.
- **Attempt 2.** After attempt 1's score showed the penalty was tiny, the AI checked on the validation blocks that averaging 5 seeds helps (56.7 vs 60.8, better on 18/18 blocks) and prepared that forecast. I submitted it: RMSE 106.56, score 106.5777. Attempt 1 remains the counted best.
- The AI recommended stopping at two submissions.

## Post-mortem and attempt 3 (5 October 2026)

Prompt: "Why Am I so low on the leaderboard?" (with the class leaderboard pasted: rank 37 of 40).

- **Diagnosis (validation only).** The AI found the log-target models under-predict the level: their validation forecast mean is 84.8 against a true 96.6, with large misses on busy weeks. A single scaling factor did not fix it.
- **Raw-scale models.** It trained the same configuration on the raw-scale target (stage 8). These over-predict instead (mean 104.3).
- **The fix.** It averaged 5 log and 5 raw-scale models. That blend is the best on validation (block RMSE 55.2) and unbiased (mean 94.6).
- **Submission.** I submitted it as attempt 3: RMSE 85.04, rank 31. P = 85,770, E = 75.
- **Follow-up.** Prompt: "Why shouldn't we try to improve it when we have the option?" The AI agreed to keep improving, provided every choice is made on validation rather than by hidden-week scores.

## Attempts 4 and 5 (5 October 2026)

- The AI tested wider models, a longer context, and a log-target model trained with its loss on the raw scale (stages 9–10).
- Only the raw-loss models helped, and only as extra ensemble members. The 15-model blend scored 54.4 against 55.2 on validation, a gain within noise.
- I submitted the 15-model blend: RMSE 89.69. Attempt 5 resubmitted the same forecast by accident.
- Attempt 3 (RMSE 85.04) remains the counted best.

## Final review (5 October 2026)

Prompt: "please review the assignment document and submission and make sure nothing is incomplete except things related to submission of files etc"

The AI checked the submission requirement by requirement against the handout and made these changes:
- added station-4 per-product results to Response 1A;
- named the population and split for every claim;
- brought Response 4 within 200 words, with the numbers moved into tables;
- added an explicit worked-example check to Response 3;
- added the Output 4.1 figure;
- updated the Task 2 model description to the final model (convolutional covariate embedding and engineered covariates);
- added a references section;
- updated this appendix and the README.

## My edits

I reviewed everything and made no changes.

## Notes for the viva

- The Response 1A and 4(a) predictions were written by the AI before any output existed, and the report says so.
- Every number in the report comes from the executed notebook, its CSV exports, or the JSON logs in `task2/results/`.
