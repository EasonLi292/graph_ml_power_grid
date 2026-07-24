# Sensitivity gate — can we trust backprop through the surrogate for repair?

`scripts/sensitivity_gate.py` measures the derivative fidelity the
gradient-repair loop actually consumes, against re-simulation, per anchor:
single-edge width perturbations (+25%, 12 bot + 6 top edges) and global
decap (×2 / ×0.5), on 3 heterogeneous-width designs per anchor. Pass bar:
sign ≥ 95 %, mean site-ranking Spearman ≥ 0.8. Cost: ~4 s per checkpoint
(CPU) — cheap enough to run as a standard eval after every training run.

## Results (2026-07-24, seed 0)

| checkpoint / anchor    | sign | rank ρ | mag ratio | decap sign | verdict |
|------------------------|------|--------|-----------|------------|---------|
| v7 edgeconv @ (3,7) T  | 0.79 | 0.745  | 0.79      | 1.00       | FAIL    |
| v7 edgeconv @ (4,7) O  | 0.88 | 0.766  | 1.11      | 1.00       | FAIL    |
| v7 edgeconv @ (7,13) O | 0.29 | 0.017  | 0.00      | 1.00       | FAIL    |
| v7 admitt.  @ (3,7) T  | 0.83 | 0.827  | 0.80      | 1.00       | FAIL    |
| v7 admitt.  @ (4,7) O  | 0.53 | −0.049 | 1.11      | 1.00       | FAIL    |
| v7 admitt.  @ (7,13) O | 0.47 | 0.292  | 0.01      | 1.00       | FAIL    |
| v6 edgeconv @ (3,7) T  | 0.83 | 0.845  | 0.84      | 1.00       | FAIL    |
| v6 edgeconv @ (4,7) O  | 0.44 | −0.337 | 0.92      | 1.00       | FAIL    |

(T = train anchor, O = held-out anchor. Small samples — 3 designs ×
18 edges — read the pattern, not the third decimal.)

## What the numbers mean

1. **The ground truth is mixed-sign.** Only ~72 % of single-edge widenings
   *reduce* worst droop at (3,7) — current redistribution makes the rest
   neutral or harmful. So sign accuracy is a real test (the trivial
   "widening always helps" rule scores 0.72), and the best model's 0.79–0.88
   is only marginally better than trivial. One in five proposed edge moves
   goes the wrong way.

2. **Near training anchors there is real ranking signal** (ρ 0.75–0.85) —
   just under the bar. Gradient repair *with a verifier in the loop* would
   mostly work on trained topologies.

3. **Derivative fidelity collapses on held-out anchors** — ρ 0.02 / −0.05 /
   −0.34. The v7 negative-transfer finding (cluster-memorization) is now
   demonstrated in the *derivative* domain, which is the one repair uses.

4. **Big-die sensitivities are architecturally zero.** At (7,13) the model's
   response to most single-edge changes is literally 0.0: with 7 message-
   passing hops, a random edge is outside the worst load's receptive field
   on a 13×13 die. (The sim's own deltas there are tiny too — median 2×10⁻⁵
   relative — single-edge moves barely matter on big dies; but the *ranking*
   among candidate sites is exactly what a generator needs and the model has
   none.) This is the synthetic-track version of the reach problem the
   superposition-attention architecture was built to solve on IBM.

5. **What works:** global decap direction is 100 % correct everywhere, and
   autograd·Δ matches the model's own finite difference within ~14–24 %
   (median) — one backward pass is a faithful proposer *of the model's
   beliefs*. The beliefs are the problem, not the linearization.

## Implications for the repair experiment

- Gradient repair through today's checkpoints is usable only near training
  anchors, only with simulator verification of every step, and not on big
  dies. The gate FAILs as a prerequisite for the repair harness.
- The most direct fix is **gradient supervision (Sobolev training)**: the
  synthetic system is linear, so exact ∂droop/∂(every edge R, every decap C)
  is one adjoint solve per design — nearly free label generation. Train the
  model to match droop *and* its Jacobian; re-run this gate as the
  acceptance test.
- The big-die reach problem needs a global term (the attention/KV-cache
  machinery from the IBM track) or deeper stacks; gradient supervision will
  make its absence visible in the loss rather than silently unlearned.
- Load timing should become a varied axis in the next dataset build so the
  learned sensitivities condition on workload (IBM lesson).
