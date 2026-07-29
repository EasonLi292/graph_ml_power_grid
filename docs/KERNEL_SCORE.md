# Nonlinear kernel score — what it fixes, and what it measurably buys

Implements the user's proposal: pass the impedance score through a learned
function instead of using it bilinearly. Softmax-like in spirit, never
softmax (contributions superpose rather than compete). The layer stays
**one-shot and purely global** — no neighbour lists, no message passing.

## The two mechanisms, separated

A nonlinearity can do two different things here, and conflating them wastes
effort. Measured directly (check 9):

| | effect |
|---|---|
| **scalar** monotone `f(z_ij)` | per-pair ranking **exactly unchanged** (ρ = 1.000000), aggregated output `Σⱼ f(z_ij)vⱼ` changes **3.9×** |
| **multivariate** `f(z_ij, z_jj, …)` | ranking **does** change (ρ 0.79 vs the bilinear order) |

So a monotone squashing function *reweights* (which sources dominate a
node's droop) but cannot *reorder*. Reordering requires the score to see
per-node quantities — the source's own self-impedance. Both are now in the
design:

- **Reweighting** — multi-scale Gaussian kernel of effective resistance,
  `K_ij = Σ_t w_ht·exp(−γ_ht‖F_i − F_j‖²)`, with `γ` learnable per head.
  Large `γ` concentrates droop on electrically nearby sources: a soft
  near/far split with no neighbour list.
- **Reordering** — `φ`/`ψ` now consume **per-channel** self-impedance
  (previously only a single summed scalar reached the encoder).

## Why it stays O(N)

Random Fourier features (Rahimi–Recht): with `w ~ N(0, 2γI)` and
`z(u) = √(2/D)·cos(w·u + b)`, `⟨z(u), z(v)⟩ ≈ exp(−γ‖u−v‖²)`. The score
remains an inner product, so the existing Kronecker cache is unchanged and
no `[N,N]` tensor is built. `γ` stays learnable because the frequency is
`√(2γ)·w₀` with `w₀` frozen. Verified factorized-vs-naive at **2.4e-15**
(forward) and **3.7e-16** (gradient).

This needs `R_eff` to be a true Euclidean distance, which the asymmetric
`(p, s)` pair does not provide. `dc_symmetric_factor` returns `F` with
`F_i·F_j = Z_ij` **and** `‖F_i−F_j‖² = R_eff` (both exact to 1.5e-15 at
full rank; `R_eff` median error 7.7 % at m=32).

### Taylor was tried first and rejected on measurement

Expanding `exp(2γ⟨p,s⟩)` as a series is an *exact* inner product, so it was
the obvious route. It is unusable: the series diverges once `2γz_ij > 1`.

| γ | Taylor error | kernel spread |
|---|---|---|
| 0.1 | 8 % | 3.7× (barely discriminates) |
| 0.3 | 59 % | 50× |
| ≥1 | ~100 % | — |

`k=3` is no better than `k=2`. The regime where the kernel is actually
useful is exactly the regime where Taylor fails. RFF error is instead flat
in `γ` (≈0.07 at D=512 for γ=0.1, ≈0.16 at γ=100) and controlled by `D`
alone. `kernel_feature="taylor"` is kept only for the exactness test.

## What it buys on IBM — the measurement that matters

Per-source Spearman of predicted vs **exact** `Z_ij` on ibmpg1t, rank-32
factors, fitted on 8 sources and evaluated on 8 **held-out** sources:

| predictor | train | held-out |
|-----------|-------|----------|
| `z_ij` — current bilinear score | 0.314 | **0.245** |
| `−R_eff` — any monotone kernel of distance | 0.174 | 0.356 |
| learned multivariate `f(z_ij, R, z_ii, z_jj)` | 0.581 | **0.577** |

**A learned function of the same rank-32 invariants more than doubles the
achievable fidelity (0.245 → 0.577), and it generalises** (0.581 train vs
0.577 held-out — no overfitting).

This corrects a claim in `docs/IBM_FACTOR_SCALING.md` and in the design
note: "no pointwise function of the factors can raise the IBM ceiling."
That was inferred from hand-chosen forms (`z_ij`, `−R_eff`), not from a
learned one, and it is wrong. The ceiling for a *learned* multivariate
function is materially higher. The high-rank obstruction is real for
reconstructing `Z` exactly, but the model does not need `Z` — it needs a
useful similarity, and that is far more reachable.

Caveat kept honest: 0.577 is a ceiling for an unconstrained pointwise MLP
on exact invariants. The deployed form (RFF kernel + `φ`/`ψ` gains) is more
restricted, and RFF adds approximation noise (≈0.13 absolute at D=128).
How much of the 0.577 the deployed form actually reaches is what the
synthetic training run and a follow-up IBM pass will show.

## Structural limit: the kernel carries no decap gradient

The kernel is a function of DC effective resistance, and a capacitor is an
open circuit at DC. Measured, not assumed — autograd through
`dc_symmetric_factor`:

| path | d/d(C_decap) | d/d(ww_top) |
|---|---|---|
| kernel term (`fdc`) | **None** (never participates) | 3.30 |
| bilinear term, AC channels | 2.1e10 | — |

So decap sensitivity rides *entirely* on the bilinear term's AC channels;
the kernel can only reweight the resistive picture. Since decap is one of
the three design knobs the repair loop turns, this bounds what the kernel
can contribute to that knob to exactly zero, and it is why the bilinear
term is kept rather than replaced. If the gate shows decap sign accuracy
is the binding failure, adding scales is the wrong fix — the kernel would
need an AC-distance analogue (a complex `R_eff`), which is not built.

## Choosing `n_rff` — measured, and the metric matters

The trainer default was `n_rff=128`, inherited from check 8 which reports
**max-abs** pairwise kernel error at D=64. That metric is the wrong one and
it is pessimistic: at sharp `gamma` most true kernel values are ~0 and
mutually indistinguishable, so ranking them is both hopeless and irrelevant.
What the layer computes is `out_i = sum_j K_ij v_j`.

Per-pair ranking (pessimistic) vs aggregated output, anchor (7,13), N=270,
5 draws of the random basis, at the three gammas the model initialises to:

| D | per-pair rank rho, g=1.70 | out rel-err, g=1.70 | node-order rho, g=1.70 |
|---|---|---|---|
| 128 | 0.70 | 0.356 | 0.926 |
| 256 | 0.78 | 0.242 | 0.956 |
| 512 | 0.84 | 0.175 | 0.978 |
| 1024 | 0.88 | 0.136 | 0.985 |

So the node ordering the repair loop consumes survives even at D=128, but
the **magnitude** carries 26-36 % approximation noise there — and the
sensitivity gate scores magnitude ratio. Since `kw` starts at zero (the
kernel begins silent and the model learns how much to trust it), a noisy
kernel would be learned *away*, and we would wrongly conclude the kernel
does not help when it was the approximation failing. **D=512 is the run
default** (rel-err halved, node order 0.98-0.99); a D=128 arm is trained
alongside to measure the sensitivity directly rather than assume it.

Error grows only mildly with N (rel-err 0.30 at N=79 vs 0.36 at N=270,
g=1.70, D=128), which is the encouraging direction for IBM scale.

### Orthogonal random features: tested, rejected

Yu et al. 2016 orthogonalise the RFF directions (keeping chi-distributed
lengths) and prove lower variance for the Gaussian kernel. It is a frozen
buffer, so `gamma` would stay learnable. On this factor geometry it does
not pay: rms 0.0711 -> 0.0555 at D=128/g=0.31, but essentially nothing
where it is needed (0.0863 -> 0.0815 at g=1.70, rank rho 0.702 -> 0.708).
The limitation is not direction correlation but the *absolute* error floor
against tiny kernel values. Not adopted — no complexity added for it.

## Use

```bash
# kernel score at the measured D (see above); D=128 arm quantifies D-sensitivity
python scripts/train_impedance_attention.py --score kernel --n-rff 512 \
    --epochs 50 --seed 0 --ckpt checkpoints/imp_attn_kernel512_s0.pt
# unchanged bilinear baseline, same harness
python scripts/train_impedance_attention.py --score bilinear --epochs 50 \
    --seed 0 --ckpt checkpoints/imp_attn_bilinear_s0.pt
# gate at 12 designs
python scripts/sensitivity_gate.py --arch impedance --n-designs 12 \
    --ckpt checkpoints/imp_attn_kernel512_s0.pt \
    --out docs/analysis/sensgate_kernel512_s0.json
```

Device: **CPU, not MPS** — measured 8 s vs 17 s per epoch on the same
config. These are batch-1 graphs of 72-390 nodes; accelerator launch
overhead dominates the arithmetic. A CUDA box will not change this much
either; the useful axis is running seeds concurrently, not per-run speed.

Factors are cached to `datasets/regular_v7_anchors/_factors` (9.0 GB for
both scores, ~8 min to build, reused by every subsequent run and by every
seed). Each training process then holds ~5 GB resident, so keep concurrency
at 3 on a 24 GB machine.

`--score kernel` adds one extra DC solve per sample to the factor cache
(`_fdc` suffix in the cache key, so bilinear caches are not invalidated).
All 9 checks pass for both scores.
