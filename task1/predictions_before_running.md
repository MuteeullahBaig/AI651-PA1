# Predictions recorded before running the notebook

Written 2026-09-25 20:37 (local clock), before any cell of `Assignment1.ipynb` was executed and before any
Output existed. The report quotes these unchanged and revises them against the measurements.
They were drafted with an AI assistant (see the AI-usage appendix).

## Response 1A: expected ordering (best to worst, validation RMSE)

**Established stations 1–3:** Period-routed ridge < Raw Attention < Shared ridge ≈ Sensor-specific ridge.

**Held-out station 4:** Period-routed ridge < Raw Attention < Shared ridge; Sensor-specific ridge unavailable.

Reasoning:
- Pace, not station, is what changes the continuation. Period routing conditions on exactly that variable, and within
  a narrow pace band a linear map is close to ideal for a sinusoid plus a smooth ramp.
- Raw Attention can build a context-dependent rule, but it has to learn around the drift ramp and to discover the
  period from about 1,800 training windows. I expect it to beat the unconditioned ridges but not the router.
- Shared and sensor-specific ridge make the same compromise across paces. Sensor-specific sees a third of the data per
  matrix and conditions on a variable that does not determine pace, so it should be no better than shared.
- All ridge maps are linear and fitted over random phases, so a new station's amplitude and phase should not hurt
  them. Raw Attention may degrade slightly on station 4, whose amplitude, offset and drift lie at the edge of the
  training stations' range.

## Response 4(a): which upgrade should help most when the slow component is prominent

Decomposition should help most. It removes the drift ramp before the mixer sees it and hands the ramp to a linear trend
head. Delay mixing on a raw, ramp-dominated context would score delays by correlations that the ramp dominates
(favouring short lags), so the delay switch alone cannot exploit the periodicity there.
