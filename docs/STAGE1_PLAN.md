# Stage-1 plan — what the stage-0 data actually says, and what to run next

Post-mortem of the stage-0 GPU run (`docs/STAGE0_RESULTS.md`, commit
`bcb791f`), re-verified locally from the raw histories, the npz datasets, and
new training-free probes. Facts first; each experiment below traces to a fact.

## Verified facts (reproduced locally, not taken from prior analysis)

**F1 — All 5 runs finished below the static baseline** on within-net ranking
(main: 0.324 VDD / 0.256 GND vs baseline 0.437 / 0.397 on held-out pg2t).
Residual runs peaked on val at epoch 7–8 and declined; train loss fell to
~0.02 — the 5 training graphs are memorized. Validation was 5% node masks
**on the training grids**, so checkpoint selection cannot observe cross-grid
overfitting.

**F2 — The model ended below a floor it provably has.** With the residual
target, predicting a constant 0 reproduces the static baseline's ranking
exactly (eval adds `sd_log` back). Training made holdout ranking *worse* than
doing nothing. The final head is default-initialized (not zero), so the model
does not even start at that floor.

**F3 — The DC geometry anchor carries zero ranking information beyond the
baseline.** Exact DC superposition `|Z_dc @ I_peak|` (full solve, no sketch,
no learning) is rank-identical (Spearman = 1.000) to the t=0 static drop on
both pg1t and pg2t. The stage-0 hypothesis could not have beaten the baseline
at any sketch resolution — the flat m=8/64/128 sweep is a corollary, not an
independent finding.

**F4 — Single-frequency AC geometry is not the fix either.** `|Z(ω) @ I_peak|`
at ω = 1e9–1e11 rad/s is strongly *anti*-correlated with droop (pg1t VDD
−0.70) but nearly collinear with DC electrical distance: a DC+AC rank
combination fitted on pg1t transfers to pg2t at +0.005. The
`STAGE0_RESULTS.md` "suggested next step" (single-frequency AC sketch) would
likely have failed.

**F5 — The missing input is load timing.** The benchmark loads are ~10 ps
pulses with per-load delays spanning 0 to >1.2 ns and mixed 2 ns / 3 ns
periods. The model's features carry only the per-node **peak** current;
which loads fire together — the actual driver of within-net droop
differences — was never an input.

**F6 — Quasi-static superposition + real timing demolishes the bar.**
`v_i(t) = Z_dc @ I(t)` over one 6 ns hyperperiod (240 solves, ~3 s per grid,
one LU factorization, zero learning) ranks held-out pg2t at:

| within-net ρ (pg2t) | timing-QS | static baseline | best trained model |
|---------------------|-----------|-----------------|--------------------|
| VDD                 | **0.846** | 0.437           | 0.324              |
| GND                 | **0.793** | 0.397           | 0.295              |
| pooled              | **0.880** | 0.680           | 0.598              |

pg1t: 0.868 / 0.849 / 0.918. Amplitude overpredicts ~2× (no damping) — a
learnable correction.

**F7 — The sketch cannot carry the timing readout.** The JL version
`max_t |r_i · Σ_j r_j I_j(t)|` collapses to 0.25 pooled at m=128 on pg2t:
the within-net differences are near-field-dominated, exactly where JL noise
is largest. Timing-QS must enter as an **exact precomputed feature/floor**;
sketched attention and local convs learn corrections on top.

## Experiments / fixes

**E1 — Timing-QS baseline swap (highest priority, mostly CPU).**
- New `scripts/patch_ibmpg_timing.py` (mirror of `patch_ibmpg_rg.py`):
  backfill into each npz (a) `tqs_peak` — the exact per-node timing-QS peak
  (F6, 3 s/grid); (b) per-node local load waveform binned over the 6 ns
  hyperperiod (~24 bins), from the netlist PULSE params.
- Trainer: residual target becomes `log droop − log tqs_peak`; report
  `tqs_peak` ranking as the new baseline. New success bar on pg2t: beat
  **0.846 VDD / 0.793 GND / 0.880 pooled**, and beat the floor's MAE after a
  single global log-offset calibration fitted on training grids.

**E2 — Floor-preserving training (fixes F2).**
- Zero-init the final head layer → the model starts exactly at the floor.
- Residual penalty λ·mean(ŷ²) so departures from the floor must earn their
  keep; sweep λ ∈ {0, 1e-3, 1e-2}.

**E3 — Honest model selection (fixes F1).**
- Leave-one-grid-out validation: train on 4 grids, validate on 1 full held-out
  grid (pg4t), test pg2t. Checkpoint selection on val-grid within-net
  Spearman. Keep the training-grid masks only as a train-fit diagnostic.

**E4 — Time-structured attention values (stage 1 proper, motivated by F5–F7).**
- Values become per-load time signatures: KV cache `s(t) = Σ_j ψ(k_j) v_j b(t)`
  with ~24 time bins; per-head temporal kernel + peak-pool over t. Heads as
  frequency bands now operate on real temporal structure. Basis-invariance
  rule unchanged (time bins are basis-free).
- Local convs consume the timing bins directly — the near-field part the
  sketch cannot represent (F7).

**E5 — Controls for the next GPU run.**
- ŷ ≡ 0 control: pipeline must reproduce the 0.846/0.793/0.880 floor exactly.
- No-attention control: `tqs_peak` + timing features + local convs only. If
  this matches E4, attention still isn't paying rent on these benchmarks.
- Label-noise bound (optional): re-solve pg1t at 1× vs 2× native dt, Spearman
  of per-node peaks, to bound how much of the remaining gap is target noise.

Probe scripts used for F3/F4/F6/F7 are in `scripts/probes/`
(`dc_ceiling.py`, `ac_ceiling.py`, `timing_ceiling.py`, `sketch_timing.py`;
run as `python scripts/probes/<name>.py ibmpg1t ibmpg2t` from the repo root);
E1 productionizes the F6 computation.

## Why timing must be *encoded*, not just scored (the generative goal)

The end goal is inverse design: given a graph and a droop budget, generate
the changes (wire widths, decap) that meet it. That requires the surrogate's
*local sensitivities* — ∂(droop at i)/∂(knob at j) — to be right, and whether
decap at a spot helps depends on whether the loads near it fire together.
So load timing is workload *conditioning* the model must encode, and the
metric that ultimately matters is sensitivity fidelity (perturb a knob,
compare surrogate delta vs re-simulation), not ranking alone. IBM grids
have no knobs — they validate the forward model; the gradient-fidelity
check lives on the synthetic 3-knob track, where load freq/duty/phase are
currently fixed constants and should eventually become a varied axis.

## Implemented (2026-07-23, CPU smoke-tested, ready for GPU)

- **E1** `scripts/patch_ibmpg_timing.py` — backfilled all 6 npz files with
  `tqs_peak` (exact quasi-static timing peak; 1–71 s/grid) and sparse
  `wave_node`/`wave_bins` (signed max-|·| per 250 ps bin, 24 bins over the
  6 ns hyperperiod — all benches use 2/3 ns periods). `load_graph` pools
  both through the short-merge.
- **E1/E2/E3** `scripts/train_ibmpg_attn.py` — residual base is now
  `tqs_peak` (static drop kept as a comparison baseline in the report);
  per-grid z of log-tqs + shape-normalized wave bins appended to node
  features; `--val-bench ibmpg4t` fully held-out selection grid;
  `--res-penalty 1e-3` L2 on the residual; train-grid masks demoted to a
  diagnostic.
- **E2/E4** `eason/attention_model.py` — final head zero-initialized
  (training starts exactly at the floor); `SuperpositionAttention` gains
  time-structured values: per head a temporal cache `T_c = Σ_j ψ_c(h_j)
  (r_j ⊗ w_j)` read as a sketched droop profile `r_iᵀT_c ∈ R^B`, pooled
  (max |·|, mean) — ablate with `--no-time-values`.
- CPU smoke (2 epochs, pg1t only, m=16): pipeline floors on pg2t reproduce
  the probes exactly — tqs baseline 0.880 / 0.846 / 0.793 pooled/VDD/GND —
  and the model stays at the floor while already improving MAE.

## GPU launch (stage 1)

Success bar on held-out pg2t: **beat the tqs floor** — within-net Spearman
> 0.846 (VDD) / 0.793 (GND), pooled > 0.880 — and beat its MAE (~132 mV
uncalibrated) decisively; the ŷ≡0 control must reproduce the floor exactly.

```bash
# main
python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t --val-bench ibmpg4t \
    --device cuda --epochs 60 --ckpt checkpoints/ibmpg_attn_s1.pt

# control 1: floor reproduction (freeze at init ≈ epoch-0 eval; res-penalty
# high enough that the model cannot leave the floor)
python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t --val-bench ibmpg4t \
    --device cuda --epochs 1 --lr 0 --ckpt checkpoints/ibmpg_attn_s1_floor.pt

# control 2: no attention time-values (are temporal values paying rent
# beyond the tqs feature + local wave convs?)
python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t --val-bench ibmpg4t \
    --device cuda --epochs 60 --no-time-values \
    --ckpt checkpoints/ibmpg_attn_s1_notv.pt

# control 3: no attention at all (local convs + timing features only)
python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t --val-bench ibmpg4t \
    --device cuda --epochs 60 --n-conv 3 --heads 1 --m-sketch 8 \
    --ckpt checkpoints/ibmpg_attn_s1_localish.pt

# sweep: residual penalty (0 = stage-0 behavior risk; 1e-2 = tight floor)
python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t --val-bench ibmpg4t \
    --device cuda --epochs 60 --res-penalty 1e-2 \
    --ckpt checkpoints/ibmpg_attn_s1_pen1e2.pt
```

Notes: pg4t is the atypical grid (3 ns periods only, tqs median 4.3 mV) —
a deliberately hard validation grid; if selection looks pathological, retry
with `--val-bench ibmpg3t`. Datasets: npz files must be the locally patched
ones (tqs_peak present); re-run `scripts/patch_ibmpg_timing.py` after any
rebuild (it needs `Rg_*`/`Lg_*`, so run `patch_ibmpg_rg.py` first).
