# IBM sparse factors — the machinery works, the low-rank assumption does not

Item 3 of the agreed plan: cached sparse forward factors for IBM
(`tools/impedance_factors_sparse.py`). Two separable results — one positive,
one negative — measured locally on the patched IBM npz graphs.

## Positive: the sparse path scales fine

Same mechanism as the dense prototype (randomized subspace iteration,
conjugate-transpose Galerkin projection), assembled sparsely and solved with
SuperLU. Measured, `m=16`, `q=2`, two frequencies (DC + 1 GHz):

| bench | n_free | LU | factors | peak RSS |
|-------|--------|-----|---------|----------|
| ibmpg1t | 25,372 | 0.0 s | 0.3 s | 0.6 GB |
| ibmpg2t | 163,907 | 0.6 s | 5.2 s | 1.3 GB |
| ibmpg3t | 475,297 | 1.2 s | 11.6 s | 2.6 GB |

Roughly linear in node count and cheap enough to cache all six benches. The
dense prototype could not run any of these.

## Negative: `Z` is not low-rank at IBM scale

Rank-16 factors give ~1.9 % droop error on the synthetic grids. On IBM they
do not reconstruct `Z` usefully, and **raising the rank barely helps**.

Within-source Spearman of `|Z_ij|` (rank the 400 observers for one source —
the physically meaningful question), pg1t:

| m | DC | 1 GHz |
|----|------|------|
| 32 | −0.23 … +0.31 | −0.69 … +0.53 |
| 128 | +0.16 … +0.49 | −0.18 … +0.53 |
| 256 | +0.04 … +0.50 | +0.14 … +0.53 |

Pearson on `|Z|` is meanwhile 0.94–0.999 — the classic signature of a fit
dominated by a few large entries while the ordering is wrong.

**Mechanism — the spectrum is flat.** Ratio of the largest to the 16th
eigenvalue of `Z(DC)`:

| system | n_free | λ₀/λ₁₅ | top-16 share |
|--------|--------|--------|--------------|
| synthetic (13,13) | 325 | **35.4** | 78 % of full mass |
| ibmpg1t | 25,372 | **1.77** | 17 % of top-300 mass |
| ibmpg2t | 163,907 | 2.32 | — |
| ibmpg3t | 475,297 | 2.31 | — |

On the synthetic grids a handful of global modes (few pads, small die)
dominate, so rank-16 captures the response. On IBM every node has its own
ground tie and package inductor, which makes `Z` close to diagonal-dominant
— i.e. genuinely high rank. Capturing the near field would need rank in the
thousands, which destroys both the O(1)-reach property and the linear
complexity that motivated the architecture.

This is the same wall hit in stage 1, where the JL-sketched timing readout
collapsed to 0.25 pooled at m=128 while the exact version scored 0.88.
Consistent, independent confirmation: **global low-rank structure captures
the far field; within-grid droop ranking is near-field dominated.**

## Consequences for the plan

- **Item 2 (synthetic training + gate) is unaffected** — the low-rank
  assumption is verified good exactly there (λ₀/λ₁₅ = 35, 1.9 % error), and
  that is where the design knobs, the sensitivity gate and the repair
  objective live. Proceed.
- **Item 4 (IBM learning) should not be run as specified.** The impedance
  term would supply far-field structure only, and IBM's target is
  near-field-ranking dominated, so it is predicted to fail the same way
  stage 0 did — before any GPU time is spent. The prediction is falsifiable:
  if run anyway, expect within-net Spearman at or below the timing-QS floor
  (0.846 / 0.793).
- Three honest options for IBM, in increasing order of work:
  1. **Hierarchical / H-matrix factorization.** `Z` is not globally low
     rank, but its *off-diagonal blocks* are — this is precisely the FMM /
     hierarchical-matrix setting. Principled, and the standard answer for
     this kernel class. Significant implementation.
  2. **Hybrid**: global attention for the far field + a small local stack
     for the near field. Cheap, and stage 0's ablation already showed local
     rounds carry real signal (0.590 → 0.449 without them) — but it is
     explicitly excluded by the current experiment brief.
  3. **Drop IBM as a learning target** and keep it as a forward realism
     check, per `docs/IMPEDANCE_ATTENTION_DESIGN.md` §9.
- **Item 5 (implicit-adjoint solver) keeps its value**, since it serves the
  synthetic/repair track where the rank assumption holds. Its priority
  relative to IBM work goes *up*, not down.

## Status of this module

Forward-only. The solve runs in SciPy, so factors carry **no gradient** to
R, C or L. Adequate for IBM learning experiments (those grids have no design
knobs) but it does **not** satisfy `docs/OBJECTIVE.md`; results computed
from these factors must not be presented as repair-capable.
