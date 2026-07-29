# Factor-stability and reciprocity audit

Does the reduced-rank reciprocity defect matter? **No — but a different
defect in the same area matters a great deal, and it is in the architecture,
not the factorization.**

Reproduce: `python scripts/probes/factor_stability_audit.py --out docs/analysis/factor_audit.json`
4 probe seeds, frequencies {0, 5.62e9, 2.81e10} rad/s, anchors (3,7)
(n_free=55, exact rank reachable) and (7,13) (n_free=211).

The learned attention score was never symmetrized and is not touched here.
Everything below concerns the physical impedance representation only.

## 1. The reciprocity defect is real and rank-dependent

`||Z_hat - Z_hat^T|| / ||Z_hat||` at w > 0, anchor (7,13):

| variant | m=8 | m=16 | m=32 | exact |
|---|---|---|---|---|
| A hermitian (current) | 0.24 | 0.21 | 0.15 | 1.4e-15 |
| B symmetrized | 3.6e-16 | 5.1e-16 | 7.2e-16 | 8.8e-16 |
| C complex-symmetric | 5.1e-16 | 6.1e-16 | 8.5e-16 | 9.3e-16 |

DC is reciprocal at every rank for every variant (~3e-16): `Y` is real there.

## 2. It does not cause the instability

Max spread of model sensitivities across the 4 probe seeds — same circuit,
same design parameters, only the random probe basis differs:

**`d(worst droop)/d(ww_top)`**, anchor (7,13):

| variant | m=8 | m=16 | m=32 | exact (3,7) |
|---|---|---|---|---|
| A, raw-channel model | 1.0 | 1.0 | 1.1 | **1.2** |
| B, raw-channel model | 1.5 | 1.2 | 1.3 | **1.1** |
| A, invariant-channel model | 0.37 | 0.099 | 0.079 | **1.5e-08** |
| B, invariant-channel model | 0.38 | 0.097 | 0.078 | **7.4e-09** |

**`d(worst droop)/d(C_decap)`**:

| variant | m=8 | m=16 | m=32 | exact (3,7) |
|---|---|---|---|---|
| A, raw-channel model | 0.80 | 0.32 | 1.3 | **1.9** |
| A, invariant-channel model | 0.049 | 0.031 | 0.015 | **1.4e-10** |

Three things follow, and they are the point of the audit:

1. **Symmetrization changes nothing about stability.** B makes reciprocity
   exact and leaves every sensitivity spread statistically where A had it
   (1.5/1.2/1.3 vs 1.0/1.0/1.1 raw; 0.38/0.097/0.078 vs 0.37/0.099/0.079
   invariant). Reciprocity error and gradient instability are unrelated here.

2. **The raw-channel architecture is not a deterministic function of the
   physics.** At exact rank the factors reconstruct `Z` to 1.3e-15 and the
   invariant channels agree to 1.4e-15, yet the raw-channel model's
   wire-width sensitivity still moves by **120 %** and its decap sensitivity
   by **190 %** when only the probe seed changes. The same circuit, the same
   design, different gradients. This is the defect that matters, because
   gradients are what repair consumes.

3. **Consuming invariant channels fixes it.** The same measurement with the
   invariant-channel model is 1.5e-08 / 1.4e-10 at exact rank, and at
   reduced rank the residual spread falls monotonically with rank
   (0.37 -> 0.099 -> 0.079), which is ordinary sketch error rather than
   basis dependence.

## 3. Reconstruction — and why symmetrization loses at equal width

`||Z_hat - Z_exact|| / ||Z_exact||` at w > 0, anchor (7,13):

| variant | m=8 (wid 8/16) | m=16 (wid 16/32) | m=32 (wid 32/64) | exact |
|---|---|---|---|---|
| A hermitian | 0.27 | 0.22 | 0.16 | 1.3e-15 |
| B symmetrized | 0.25 | 0.19 | 0.14 | 1.2e-15 |
| C complex-symmetric | 0.61 | 0.66 | 0.68 | **1.2** |

B beats A at equal *rank* by ~13 % relative — but B costs double the factor
width. **At equal width, A wins:** A at m=16 (width 16) reconstructs to 0.22
where B at m=8 (width 16) gives 0.25; A at m=32 (width 32) gives 0.16 where
B at m=16 (width 32) gives 0.19. Since width drives the attention cache, the
factor cache on disk and the contraction cost, equal width is the right
comparison.

**C is invalid.** `Qr (Qr^T Z Qr) Qr^T` is exactly symmetric, but it does not
converge to `Z` — 1.2 relative error *at exact rank*. `Qr` is unitary, so
`Qr Qr^T != I` and the bilinear-form projection is not consistent. Rejected
on measurement, not on principle.

## Decision

Per the stated rule:

- The reciprocity defect has **negligible effect** on predictions and
  gradients. **Leave the factorization unchanged** (variant A).
- Symmetrization improves agreement with exact `Z` at equal rank but
  **loses at equal width** and does **not** improve stability. Not adopted.
  It stays available as `proj="symmetrized"` for any future genuinely
  non-reciprocal device set, where global passive symmetrization would be
  wrong anyway.
- The complex-symmetric projection is rejected.
- **The correction belongs in the architecture**, and it already exists:
  regroup the four raw real channels per frequency into the two invariant
  ones (`Re Z = rr - ii`, `Im Z = ri + ir`) before any learned gain touches
  them. `invariant_channels()` in `tools/impedance_factors.py`.
- Learned attention remains fully directional. Nothing here symmetrizes the
  score.

### Consequence for the existing arms

`bilinear` and `kernel` both consume raw channels, so every checkpoint
trained so far has ~100 % probe-seed dependence in its wire and decap
sensitivities. Their forward predictions are much less affected (spread
0.07-0.43), which is consistent with the long-standing observation on this
project that good forward R² coexists with unusable repair gradients — part
of that gap may simply be this.

Back-porting invariant channels to those two scores is cheap and would make
the eventual three-way comparison meaningful. Until then, the honest
statement is that the AC half of their impedance score is partly a function
of the random probe draw.
