# GPU handoff — IBM fine-tune, stage 0 (attention diagnostic)

Everything below is built, smoke-tested on CPU, and ready to launch.
Nothing has been trained beyond smoke tests.

## What to copy to the server

- the repo (branch `refactor/3-knob-design-space`)
- `datasets/ibmpg/graphs/*.npz` — the 6 per-node graphs (74 MB, gitignored;
  already patched with `Rg_*`/`Lg_*` ground ties — do NOT rebuild)
- `datasets/ibmpg/graphs/_sketch/` if present (sketch cache; recomputed
  automatically if missing — needs scipy, few min for the 1.5M-node grids)
- `checkpoints/droop_v6_peredge_edgeconv.pt` — synthetic init for the
  patch-model baselines (not needed by the attention model)

Python deps: torch, torch_geometric, scipy, numpy, h5py.

## Stage 0 — the experiment that decides the architecture

Hypothesis: exact DC impedance geometry + superposition attention closes
the within-grid ranking gap. Success = beat the static-IR baseline on the
held-out grid: **pooled Spearman > 0.68, within-net > 0.44 (VDD) / 0.40
(GND)**, with MAE staying ≪ 133 mV.

```bash
# main run (~small model, full-graph steps; residual target on by default)
python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t \
    --device cuda --epochs 60 --m-sketch 64 \
    --ckpt checkpoints/ibmpg_attn.pt

# ablation A: no attention benefit from geometry? (drop to m=8 crushes it)
python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t \
    --device cuda --epochs 60 --m-sketch 8 \
    --ckpt checkpoints/ibmpg_attn_m8.pt

# ablation B: absolute target (is the residual trick doing the work?)
python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t \
    --device cuda --epochs 60 --m-sketch 64 --no-residual \
    --ckpt checkpoints/ibmpg_attn_abs.pt
```

Also worth rerunning the *patch* model at proper scale for the paper
comparison (it was CPU-starved at 40 epochs, val still climbing):

```bash
python3.12 scripts/train_ibmpg.py --holdout ibmpg2t --residual \
    --init checkpoints/droop_v6_peredge_edgeconv.pt \
    --device cuda --epochs 100 --batch-size 64 --patches-per-graph 8000 \
    --ckpt checkpoints/ibmpg_node_residual_gpu.pt
```

## Memory expectations (attention model, full-graph)

Activations ≈ N × hidden × ~12 tensors + bidirectional edge messages.
pg1t/2t/5t trivial; pg3t/4t (~1.2M nodes, ~3.4M dir-edges) ≈ 8–12 GB;
pg6t (1.5M nodes, 4M dir-edges) ≈ 12–16 GB. Fits a 24 GB card in fp32;
if it OOMs, drop pg6t from training (`--train-benches ibmpg1t ibmpg3t
ibmpg4t ibmpg5t`) — it contributes context, not the holdout.

## Reading results

Each run writes `<ckpt>.history.json` with per-epoch val within-grid
Spearman and a final `test` block: model vs `baseline_*` (static IR
drop) for all/vdd/gnd — R², MAE, p99, Spearman, top-1% hotspot recall.
Selection metric is within-grid Spearman (pooled linear R² rewards
constant-collapse — see project memory).

## Known caveats (deliberate, documented)

- The sketch's per-pair `r_i·r_j ≈ Z_ij` carries absolute JL noise
  (~Z_self/√m); *distances* (R_eff) are the reliable object (validated
  6–8 % rel err at m=32–64 on pg1t). The attention's learned modulation
  + the `‖r_i‖²` input feature can express linear-in-R_eff scores
  exactly (they factorize through the same caches). If ranking plateaus
  just below target, try `--m-sketch 128` before architecture changes.
- The npz ground ties (`Rg_*`, `Lg_*`) were backfilled by
  `scripts/patch_ibmpg_rg.py`; without them the entire GND net floats
  in any grounded-system computation. Re-run that script after any npz
  rebuild.
- Basis invariance: the sketch enters the model ONLY via inner products
  and norms. Never add a learned linear map on raw `sketch` dims — the
  random basis differs per grid and it will silently fail OOD.

## Stage 1 (after stage 0 confirms geometry helps)

Learned impedance sketcher: replace the exact DC solve with a learned
multigrid relaxation over (R, C, L, per-head ω) so geometry becomes
frequency-aware and differentiable w.r.t. design knobs. Pretrain it on
synthetic grids (exact R_eff supervision is cheap there via
`tools/transient_solver`), auxiliary-supervise with `v_dc`/static drop
on IBM, then swap it in for `load_or_compute_sketch`. The attention
stack and training script stay unchanged.
