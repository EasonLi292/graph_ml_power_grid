# Stage-1 results — load timing wins, the learned model still doesn't

GPU execution of the stage-1 launch plan (`docs/STAGE1_PLAN.md`, commit
`a9c4ee8`) on the H100. Holdout **ibmpg2t**, validation grid **ibmpg4t**
(fully held out, E3), train = pg1t/3t/5t/6t. Datasets rebuilt locally from the
UCSB benchmarks and patched with `patch_ibmpg_rg.py` → `patch_ibmpg_timing.py`.

## The floor reproduces exactly (E5 control passes)

The `--lr 0` control is bit-identical to `baseline_tqs` and reproduces the
plan's independently-derived probe numbers to 3 decimals — a genuine
cross-validation, since these npz files were built from scratch here:

| pg2t within-net ρ | this run | STAGE1_PLAN F6 |
|-------------------|----------|----------------|
| VDD               | 0.8455   | 0.846          |
| GND               | 0.7930   | 0.793          |
| pooled            | 0.8800   | 0.880          |

Zero-init head + `lr=0` holds the model exactly at the floor, as designed (E2).

## Headline: the *feature* won, the *model* did not

| run                | ρ VDD  | ρ GND  | ρ pooled | MAE mV | best ep | last val |
|--------------------|--------|--------|----------|--------|---------|----------|
| **tqs floor**      |**0.8455**|**0.7930**|**0.8800**| 132.0 | —       | —        |
| main               | 0.8454 | 0.7910 | 0.8808   | 122.9  | 1       | 0.667    |
| C2 no-time-values  | 0.8440 | 0.7917 | 0.8818   | 122.2  | 1       | 0.673    |
| C3 no-attention    | 0.8455 | 0.7901 | 0.8795   | 125.5  | 1       | 0.614    |
| D res-penalty 1e-2 | 0.8454 | 0.7910 | 0.8808   | 122.9  | 1       | 0.626    |
| static (stage 0)   | 0.4372 | 0.3972 | 0.6799   | 133.0  | —       | —        |

**Encoding load timing is a large, real win — with zero learning.** The
quasi-static timing peak moves held-out within-net ranking from
0.437 → 0.846 (VDD) and 0.397 → 0.793 (GND) over the static baseline. F5/F6
of the postmortem are confirmed end-to-end through the full pipeline.

**No learned configuration beats that floor.** The success bar (> 0.846 VDD /
0.793 GND / 0.880 pooled) is **not met**: GND is strictly *below* the floor in
every run, VDD ties at best, and the pooled gain (+0.0008…+0.0018) is noise.
Learning buys only a ~5–7% MAE reduction (132 → 122–126 mV), i.e. an amplitude
calibration on top of a ranking it cannot improve.

## Three diagnoses from the controls

**Attention pays no rent (E5).** C3 — 1 head, m=8, no usable geometry, i.e.
effectively local convs + timing features — matches the full attention model
(0.8455/0.7901 vs 0.8454/0.7910). The plan's own criterion ("if this matches
E4, attention still isn't paying rent") is met: it matches.

**Time-structured attention values pay no rent (E4).** C2 removes them and
changes nothing (0.8440/0.7917, MAE 122.2 — if anything marginally better).

**Training actively degrades held-out ranking.** Best epoch is **1 in all five
runs**; val within-net ρ falls 0.74 → 0.61–0.67 by epoch 60 while the
train-grid diagnostic holds ~0.79–0.81. The model memorizes the four training
grids and does not transfer. E2's floor-preserving machinery worked — unlike
stage 0 it no longer collapses far below the floor — but it preserves the floor
rather than improving on it. The residual penalty is inert here: λ=1e-2 is
bit-identical to λ=1e-3 because selection lands at epoch 1 either way.

## What this means

Stage 0 and stage 1 now agree on the same structural finding from opposite
directions: **on these benchmarks, essentially all of the achievable within-net
ranking comes from exactly-computed physics features, and none from the learned
graph model.** Stage 0's DC geometry was rank-identical to the static baseline
(F3), so it could add nothing; stage 1's timing quasi-static is a far better
feature, and the network still adds nothing on top of it.

The open question is no longer "which geometry?" but **"is there any residual
signal left for a model to learn?"** The gap from 0.846 to 1.0 is real, but
these runs give no evidence the current architecture can reach into it, and
three independent architectural knobs (attention on/off, temporal values
on/off, geometry resolution) all move the answer by < 0.003.

### Suggested next steps

1. **Bound the label noise first (E5, optional in the plan — now the priority).**
   Re-solve pg1t at 1× vs 2× native dt and Spearman the per-node peaks. If
   label noise caps ranking near ~0.85, the remaining gap is not learnable and
   the forward task on IBM is effectively *solved* by timing-QS.
2. **Improve the floor's physics, not the model.** The plan notes amplitude
   overpredicts ~2× (no damping). A quasi-static + first-order damping
   correction is cheap, has no learned parameters, and would attack MAE
   directly — where learning currently gains only 7%.
3. **Move the model question to where knobs exist.** Per the plan's own
   framing, IBM grids have no design knobs and only validate the forward
   model. Sensitivity fidelity — ∂droop/∂knob vs re-simulation — is the metric
   that matters for the generative goal, and it lives on the synthetic 3-knob
   track (where load freq/duty/phase should become a varied axis, since timing
   is now known to be the dominant conditioning variable).
