# GPU handoff — Sobolev (gradient-supervised) training, v7 synthetic track

Step 1 of docs/OBJECTIVE.md. Everything below is built and CPU-validated;
nothing has been trained beyond smoke tests. Acceptance test =
`scripts/sensitivity_gate.py` (bar: sign ≥ 0.95, site-rank ρ ≥ 0.8 per
anchor — current checkpoints all FAIL, docs/SENSITIVITY_GATE.md).

## What was validated locally (2026-07-24)

- `tools/torch_sim.py` — differentiable twin of the transient solver.
  Forward matches the numpy pipeline to ≤ 5e-8 rel; adjoint gradients
  match central finite differences through the *numpy* sim to < 0.1 % on
  all significant sensitivities. Supports per-site decap (numpy solver
  is scalar-only). Full exact Jacobian: 0.3 s (small die) / 1.4 s (13,13).
- `scripts/gen_jacobian_labels.py` — labels for 120 train designs
  generated locally (0.34 s/design on 8 workers); torch-sim droop
  cross-checks against the stored dataset labels at ≤ 1.2e-7 rel.
- Sobolev loss (`tools/training.py:sobolev_loss`) — bidir-fold verified
  against per-edge model finite differences (0.3–1.8 % rel, FD noise);
  double backward runs; term ≈ 7e-4 at the trained v7-edgeconv
  checkpoint in relative-sensitivity units.

## Run on the GPU box

```bash
# 1. finish the Jacobian labels for the FULL train split (~16k designs;
#    CPU-bound, ~0.3-1.4 s/design/worker — use all cores). Batches with
#    any unlabeled sample silently skip the Sobolev term, so run to 100%.
python3.12 scripts/gen_jacobian_labels.py --split train --workers 32

# 2. control: repeat v7 edgeconv training exactly (lambda=0) for a
#    same-code baseline
python3.12 scripts/train_droop.py --data datasets/regular_v7_anchors/dataset.h5 \
    --conv-type edgeconv --device cuda --epochs 50 \
    --ckpt checkpoints/droop_v7_edgeconv_l0.pt

# 3. Sobolev sweep (term ~7e-4 vs forward MSE ~1e-3 at convergence, so
#    lambda must be O(1)-O(30) to matter)
for LAM in 0.3 3 30; do
python3.12 scripts/train_droop.py --data datasets/regular_v7_anchors/dataset.h5 \
    --conv-type edgeconv --device cuda --epochs 50 \
    --jac-file datasets/regular_v7_anchors/jacobians.h5 --sobolev-lambda $LAM \
    --ckpt checkpoints/droop_v7_edgeconv_sob$LAM.pt
done

# 4. acceptance: the sensitivity gate on every checkpoint (seconds each)
for CK in l0 sob0.3 sob3 sob30; do
python3.12 scripts/sensitivity_gate.py --ckpt checkpoints/droop_v7_edgeconv_$CK.pt \
    --conv-type edgeconv --out docs/analysis/sensitivity_gate_$CK.json
done
```

Expectations / what to look at:

- Double backward makes each Sobolev epoch ~2–3× the baseline epoch.
- Success = gate improvement on sign and site-rank ρ, *especially on the
  held-out anchors* (4,7) and (7,13) — that is the falsifiable claim:
  exact derivative supervision fixes derivative transfer where forward
  supervision could not. Forward MAE is allowed to get slightly worse.
- If (7,13) stays at ρ ≈ 0 with sign ~0.5 while (3,7)/(4,7) improve, the
  bottleneck there is the 7-hop receptive field, not supervision →
  proceed to OBJECTIVE step 2 (global-reach/attention term) before
  re-judging.
- Known limitation: with `has_jac.all()` gating, partial label coverage
  disables the term on mixed batches — hence step 1 runs to completion.

Do not launch from the assistant session — user runs GPU jobs.
