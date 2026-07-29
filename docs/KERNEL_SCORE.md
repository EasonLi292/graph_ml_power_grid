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

## Use

```bash
# new kernel score
python scripts/train_impedance_attention.py --score kernel --epochs 50 \
    --ckpt checkpoints/imp_attn_kernel.pt
# unchanged bilinear baseline, same harness
python scripts/train_impedance_attention.py --score bilinear --epochs 50 \
    --ckpt checkpoints/imp_attn_bilinear.pt
# gate both at 12 designs
python scripts/sensitivity_gate.py --arch impedance --n-designs 12 \
    --ckpt checkpoints/imp_attn_kernel.pt --out docs/analysis/sensgate_kernel.json
```

`--score kernel` adds one extra DC solve per sample to the factor cache
(`_fdc` suffix in the cache key, so bilinear caches are not invalidated).
All 9 checks pass for both scores.
