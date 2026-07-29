# Per-frequency normalization + physics initialization

Reproduce: `python scripts/probes/freq_norm_audit.py --out docs/analysis/freq_norm_audit.json`
No training run. m=16, 3 frequencies {0, ω, 5ω}, 4 probe seeds, anchors
(3,7) / (7,13) / (13,25).

## The proposed scale does not work; a related one does

Two candidate per-frequency scales, both basis-invariant, per-node or
Gram-based (never `[N,N]`), differentiable in R and C, applied per
frequency, and factorization-preserving:

- **`diag`** — `s_ω² = mean_{loads} |Z_ii(ω)|²`, the suggested starting point.
- **`frob`** — `s_ω² = ‖Z(ω)‖_F² / N²`, the mean squared entry over *all*
  pairs. Computed as `tr(AᵀA · BᵀB)` from two `[d,d]` Grams, so it is
  O(N d² + d³) and never forms `[N,N]`.

Cross-frequency alignment (ratio of median block magnitude between the two
frequencies; **1.0 is perfect alignment**):

| anchor | mode | re¹ | re² | im¹ | im² |
|---|---|---|---|---|---|
| (3,7) | off | 1.4 | 2.0 | 4.7 | 22.3 |
| (3,7) | diag | 5.2 | 26.8 | 1.6 | 2.4 |
| (3,7) | **frob** | **1.0** | **1.0** | 3.3 | 10.7 |
| (7,13) | off | 1.5 | 2.3 | 3.7 | 13.9 |
| (7,13) | diag | 8.8 | **78.1** | 3.6 | 12.8 |
| (7,13) | **frob** | **1.2** | **1.4** | 3.0 | 8.8 |
| (13,25) | off | 1.2 | 1.5 | 3.6 | 13.2 |
| (13,25) | diag | 11.4 | **129.1** | 3.9 | 14.9 |
| (13,25) | **frob** | **1.1** | **1.2** | 3.2 | 10.1 |

**`diag` is worse than no normalization at all, and degrades with grid
size** (re² misalignment 26.8 → 78.1 → 129.1). The reason is physical:
diagonal impedance falls faster with ω than off-diagonal impedance does, so
dividing by the diagonal over-corrects the high frequency and *anti-aligns*
the ranges it was meant to align. The score is built from off-diagonals, so
the scale has to come from off-diagonals.

`frob` aligns the real channels essentially perfectly (1.0–1.2) and improves
the imaginary ones over no normalization (22.3 → 10.7, 13.9 → 8.8,
13.2 → 10.1) rather than trading one for the other. It is also
**size-consistent**: 1.0 / 1.2 / 1.1 across three anchor sizes.

## What it costs: probe-seed stability

| anchor | off | diag | frob |
|---|---|---|---|
| (3,7) | **+0.971** | +0.781 | +0.686 |
| (7,13) | **+0.944** | +0.913 | +0.723 |
| (13,25) | **+0.919** | +0.641 | +0.705 |

(mean pairwise Spearman of `d/dww` across 4 probe seeds)

**Any** normalization reduces gradient-ranking reproducibility. Reported
rather than hidden.

Hypothesis, *not* isolated: without normalization the DC block dominates the
score numerically, and DC is the best-conditioned channel (reconstruction
error 0.098 vs 0.22 for AC at m=16, and DC is basis-invariant at every
rank). Normalization brings the AC blocks into play, and AC information is
intrinsically noisier at this rank. If so the cost is the price of *using*
the multi-frequency information at all, and the lever is higher `m`, not
removing the normalization. Testable by zeroing the AC `alpha` blocks with
normalization on; not yet done.

## Everything else checks out

- **Factorized == explicit `[N,N]`** in every mode: 3.2e-15 to 1.8e-14.
- **Capacitance gradient non-zero** in every mode (8.4e7 to 8.2e8).
- **No head or frequency dominates at init.** Max head share 0.404–0.405
  out of 4 heads in *all three* modes — identical with and without
  normalization, so the physics initialization does not cause immediate
  domination. The mild concentration is the deliberate `logspace`
  symmetry-breaking across heads.
- **Size scaling**: with `frob` the block medians are comparable across
  anchors; without it they track absolute impedance, which grows with the
  grid.

## The caveat, reported separately as asked

Per-frequency scaling aligns the frequency *ranges*. It does **not** touch
the within-frequency tail, and cannot — dividing a block by a scalar leaves
every ratio inside that block unchanged. Measured at (7,13), identical in
all modes:

| block | max/median | p95/p5 |
|---|---|---|
| `dc^1` | 351 | 1 383 |
| `dc^2` | **123 463** | **1 913 827** |
| `re_w1^2` | 203 | 6 599 |
| `im_w2^2` | 177 | 2 831 |

The degree-2 blocks have a very heavy tail, which is expected — they are
squared transfer impedances, and distant node pairs genuinely couple by
orders of magnitude less than near ones. **No clipping and no extra
nonlinear transform has been added**: either would break the exact
factorization that keeps the layer O(N).

Note also that 8.2 % of sampled pairs are exactly zero at (3,7). That is
structural, not a tail — clamped pad nodes have zero impedance factors by
construction. Percentiles above are taken over the non-zero pairs.

## Physics initialization

`alpha` now starts at the physics instead of arbitrary scales:

- DC, degree 1 → 1.0 — ordinary superposition, `droop_i = Σ_j Z_ij I_j`
- Re/Im, degree 2 → 1.0, equal per frequency — that combination *is*
  `|Z(ω)|²`, the magnitude of the influence coefficient across frequencies
- everything else → 0.05

All heads start on the same pattern with different overall gains
(`logspace(-0.5, 0)`), so head symmetry is broken without assigning any head
a role. `alpha` is fully learnable and signed; this is a starting point, not
a constraint. Comparing a trained `alpha` against this pattern is how to
read out which frequency mixture the data actually wanted.

## Kept

One content term (degree 0), one unified dynamic kernel, DC initialized to
ordinary superposition, equal degree-2 Re/Im coefficients initialized as
`|Z(ω)|²`, all mixtures learnable, `freq_norm_mode="frob"` as the default.
No fourth score was built.
