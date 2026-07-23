# Stage-0 attention results — IBM PG transient droop

First GPU training run of the superposition-attention architecture
(`eason/attention_model.py`) on the IBM power-grid transient benchmarks.
Holdout grid **ibmpg2t**; the other five grids (pg1t, pg3t–pg6t) are the
training context. Full-graph steps, m=128 DC impedance sketch, residual target
(`log droop − log static-drop`) unless noted. See `docs/GPU_HANDOFF.md` for the
experiment design and `docs/analysis/ibmpg_attn_main.history.json` for per-epoch
curves.

Environment: H100 NVL, `.venv` (Python 3.12, torch 2.11+cu128, PyG 2.8).
Datasets rebuilt from the UCSB benchmarks (transient solve at 2× native dt);
SPICE solver validated against the shipped `.output` on pg1t at 0.6% median
error. All 6 grids patched with `Rg_/Lg_` ground ties.

## Verdict: the hypothesis is NOT confirmed

Success was defined as beating the static-IR baseline's within-net ranking on
held-out pg2t (pooled Spearman > 0.68, within-VDD > 0.44, within-GND > 0.40,
MAE ≪ 133 mV). The model clears none of these.

### Main run (m=128, residual, 60 epochs, best ckpt @ epoch 8)

| metric (held-out pg2t)   | **model** | static-IR baseline | target   |
|--------------------------|-----------|--------------------|----------|
| within-VDD Spearman      | 0.324     | 0.437              | > 0.44   |
| within-GND Spearman      | 0.256     | 0.397              | > 0.40   |
| pooled Spearman (all)    | 0.590     | 0.680              | > 0.68   |
| MAE (mV)                 | 316.9     | 133.0              | ≪ 133    |
| p99 abs err (mV)         | 749.0     | 163.5              | —        |
| top-1% hotspot recall    | 0.091     | 0.087              | —        |

The model underperforms the crude static-IR baseline on **every** ranking
metric and its absolute error is 2.4× worse.

## Why — the benchmarks are inductance-dominated, the sketch is DC-only

The static-IR baseline predicts DC IR drop, and its `rel_mae ≈ 1.0` is the tell:
DC drop is negligible next to the peak transient droop.

| pg2t grid nodes (126,905) | median | p99   | max   |
|---------------------------|--------|-------|-------|
| static DC drop (mV)       | 0.27   | 0.48  | 0.50  |
| peak droop (mV)           | 133.8  | 164.0 | 169.9 |
| ratio                     | 0.002  |       |       |

Peak droop is ~500× the static DC drop — it is set by the **dynamic** response
(package L·di/dt + on-die RC), not by DC transfer resistance. The impedance
sketch encodes exactly `r_i·r_j ≈ Z_dc`, i.e. the resistive part that is 0.2%
of the answer. So:

- The residual target `log droop − log static-drop` is a large, almost-entirely
  *dynamic* offset that the DC geometry cannot express, which is why the model's
  absolute calibration blows up (MAE 317 mV).
- The static-IR baseline still ranks at 0.68 pooled because *where* DC drop is
  worst (far from the pads) is loosely correlated with where droop is worst —
  but that correlation is weak within a net, and the DC-only model cannot
  sharpen it.

This is a clean, interpretable negative result: **exact DC impedance geometry is
the wrong physics anchor for L-dominated transient droop.** It directly
motivates the handoff's Stage 1 (a frequency-aware / learned R,C,L,ω sketcher).

## Ablations (holdout pg2t, 60 epochs)

| run                          | within-VDD ρ | within-GND ρ | pooled ρ | MAE mV |
|------------------------------|--------------|--------------|----------|--------|
| **static-IR baseline**       | **0.437**    | **0.397**    | **0.680**| **133**|
| main (m=128, residual)       | 0.324        | 0.256        | 0.590    | 317    |
| A: m=8 (geometry starved)    | 0.278        | 0.199        | 0.582    | 843    |
| D: m=64                      | 0.320        | 0.295        | 0.598    | 616    |
| B: absolute target, m=128    | 0.023        | 0.065        | 0.316    | 15.7   |
| C: no local conv, m=128      | 0.204        | 0.136        | 0.449    | 128    |

Three findings, each pointing the same way:

**A/D/main — the geometry-resolution sweep is flat.** Going from m=8 → 64 → 128
moves within-VDD ρ only 0.278 → 0.320 → 0.324 and within-GND ρ wobbles
0.199 → 0.295 → 0.256 (non-monotonic noise, not a returns curve). The handoff
predicted "drop to m=8 crushes it" *if* geometry were load-bearing. It does not
crush it — so the exact DC impedance sketch is **not** the signal driving the
model's (already sub-baseline) ranking. This is the cleanest single result.

**C — pure attention is worse than attention+conv.** Removing the local conv
stack (n-conv=0, geometry attention only) drops within-net ρ to 0.204 / 0.136,
*below* the full model. What little ranking the model has comes from the local
message passing, not the DC-geometry attention.

**B — the absolute target constant-collapses.** Predicting absolute log-droop
gives an excellent MAE (15.7 mV, 8× better than baseline) but within-net ρ near
zero (0.02 / 0.07): the model learns each net's mean droop level and stops. Low
MAE here is the mean-prediction trap the selection metric is designed to reject,
not a win.

## Bottom line

Across all five configurations, **no run beats the static-IR baseline on
within-net ranking** (best model within-VDD 0.324 vs baseline 0.437; best
within-GND 0.295 vs 0.397). The stage-0 hypothesis — that exact DC impedance
geometry + superposition attention closes the within-grid ranking gap — is
**falsified on the IBM transient benchmarks**, and the ablations show *why*:
the benchmarks' peak droop is inductance/dynamic-dominated (static IR is 0.2%
of it), so the DC-only geometry models the wrong physics and the network cannot
extract a usable within-net ranking from it at any sketch resolution.

This is a useful negative result, not a dead end. It is exactly the evidence
that motivates **Stage 1** (`docs/GPU_HANDOFF.md`): replace the exact DC solve
with a frequency-aware / learned sketcher over (R, C, L, per-head ω) so the
geometry sees the dynamic response that actually sets peak droop. The attention
stack, training loop, and data pipeline validated here stay unchanged; only the
sketch's physics needs to become dynamic.

### Suggested next step

Before building the learned sketcher, one cheap check would sharpen the story:
regenerate the sketch from the **RC (or RCL) system's dominant-pole / AC
transfer impedance** at the benchmark switching frequency (≈ the load pulse
rate) instead of the DC Laplacian, and rerun stage 0 unchanged. If a
single-frequency AC sketch already lifts within-net ρ toward the baseline, that
confirms "make the geometry dynamic" is the whole fix and de-risks Stage 1.
