# Review of the Sobolev results — what survives a noise control

Verification pass over `docs/SOBOLEV_RESULTS.md` (commit `7686ca6`), done
locally from the committed gate JSONs plus a new control the original run
did not have: **re-sampling the gate's designs for a fixed checkpoint.**

## The missing control

The gate ran at `--n-designs 3`. Designs — not perturbations — are the unit
of variance: the ~18 edges inside one design share a grid, a width draw and
a decap value, so they are strongly correlated. Per-edge binomial/Fisher
tests therefore overstate significance badly.

Measured spread over 6 design-sampling seeds, **same checkpoint, same
config** (`droop_v7_edgeconv.pt`, 7 layers):

| anchor | n_designs | sign: mean [min,max] (spread) | ρ: mean [min,max] (spread) |
|--------|-----------|-------------------------------|----------------------------|
| (3,7)  | 3         | 0.82 [0.61, 0.95] (**0.34**)  | +0.72 [+0.29, +0.88] (**0.60**) |
| (3,7)  | 12        | 0.85 [0.77, 0.89] (0.12)      | +0.80 [+0.67, +0.87] (0.20) |
| (4,7)  | 3         | 0.88 [0.77, 0.97] (**0.20**)  | +0.72 [+0.56, +0.87] (**0.32**) |
| (4,7)  | 12        | 0.83 [0.74, 0.87] (0.13)      | +0.62 [+0.48, +0.74] (0.26) |
| (7,13) | 3         | 0.41 [0.18, 0.67] (**0.49**)  | +0.14 [−0.20, +0.38] (**0.58**) |
| (7,13) | 12        | 0.37 [0.26, 0.46] (0.20)      | +0.13 [−0.10, +0.30] (0.40) |

**At n=3 a single fixed checkpoint spans ρ = −0.20 … +0.38 at (7,13).** The
entire λ sweep reported there (−0.109 … +0.385) fits inside that band.

## What survives

1. **Depth fixes big-die reach — the one large, robust result.**
   L20 λ=0 at (7,13): sign **0.94**, magnitude ratio **1.29**. The n=3 noise
   band for sign is [0.18, 0.67]; 0.94 is far outside it, and the magnitude
   ratio is a qualitative change (0.00 → 1.29 ≈ ideal 1.0), not a shift in a
   noisy statistic. The linearity check simultaneously goes from a degenerate
   0.0 % (0/0 on vanishing gradients) to a healthy 19 %. Confirmed.
2. **Depth breaks the small held-out anchor.** L20 at (4,7): sign 0.41,
   against an n=3 band of [0.77, 0.97] for the 7-layer model. Outside the
   band, and reproduced independently by both L20 runs. Confirmed.
3. **λ=3 also breaks (4,7)** (sign 0.44, same comparison). Confirmed.
4. **Decap direction is 1.00** at every anchor for every checkpoint, as in
   every previous run.

So the writeup's headline structural claim — *no single fixed depth serves
both die sizes; 7 hops is right for (4,7) and blind on (7,13), 20 hops the
reverse* — **holds**, and it is the finding that matters.

## What does not survive

| claim in SOBOLEV_RESULTS.md | status |
|-----------------------------|--------|
| "best site-ranking ever measured", ρ 0.893 @ λ=0.3, (3,7) | **noise** — control spans +0.29…+0.88 at n=3; λ=0 control itself is 0.823 and the *old* checkpoint scores 0.833 at n=12 |
| "every λ>0 lifts (7,13) off the control's negative ρ" | **noise** — band is −0.20…+0.38 for a fixed model |
| "λ=3 is best at (7,13) (ρ 0.385)" | **noise** — inside the band; its mag ratio is 0.08, i.e. sign measured on responses 12× too small |
| "λ trades near-anchor fidelity against far-anchor reach" | **unsupported** — the pattern is non-monotonic in λ (ρ at (4,7): 0.83 → 0.67 → −0.23 → 0.77) |
| L20 λ=0.3 "gives back most of the big-die gain" | **weak** — sign 0.71 vs 0.94, single draw at n=3; needs a re-gate |

**Verdict on step 1 of the objective: the Sobolev experiment is
inconclusive, not positive.** The gate cannot resolve effects of the size
gradient supervision produced. It is not evidence that λ fails either — it
is evidence the measurement was underpowered.

## Unverifiable from the repo

The forward-accuracy table (test R² 0.793 → 0.855 → 0.905) is the
better-powered measurement here — 2 000 held-out samples, not 3 designs —
and is plausibly the most informative number in the run. But the training
`*.history.json` files were not committed, so none of it can be checked.
**Please commit the histories for the six runs.** If test R² 0.905 at L20
holds up, depth is a win on both forward transfer *and* big-die reach, and
the only cost is (4,7).

## Fixes applied to the gate (this commit)

- `--n-designs` default 3 → **12** (~15 s for 3 anchors; still trivial).
- Every metric now reports a 95 % CI: Wilson for sign, **design-level
  bootstrap** for ρ. PASS now requires the CI *lower bound* to clear the bar.
- Output JSON records `hidden_dim`, `n_layers`, `n_designs`, `k_bot`,
  `k_top`, `seed` — the previous files recorded only the checkpoint path, so
  the "deep20" runs' depth was recoverable only from a filename.
- Warns when an anchor has < 30 live perturbations (at (7,13) only 17/54
  cleared the floor at n=3 — most single-edge moves there are below 1e-4
  relative, which is *why* that anchor is so noisy).

Re-gated baseline with the fixed script (`droop_v7_edgeconv.pt`, n=12):

| anchor | sign (95% CI)    | ρ (95% CI)            | mag  | live    |
|--------|------------------|-----------------------|------|---------|
| (3,7)  | 0.87 [0.81,0.91] | +0.833 [+0.77,+0.89]  | 0.88 | 179/216 |
| (4,7)  | 0.86 [0.79,0.91] | +0.654 [+0.50,+0.78]  | 0.65 | 121/216 |
| (7,13) | 0.37 [0.26,0.49] | +0.034 [−0.11,+0.17]  | 0.00 | 68/216  |

## Recommended next step

Unchanged in direction, tightened in method — go to **OBJECTIVE step 2
(global/attention term)**, because the depth/reach tradeoff is structural
and O(1)-reach is the principled fix. But:

1. **Re-gate the six existing checkpoints at `--n-designs 12`** first
   (~90 s total, no retraining). That alone may resolve whether λ=0.3 and
   L20 λ=0.3 differ at all, and it re-baselines the comparison target.
2. Commit the training histories so the forward-R² axis is auditable.
3. The comparison for the attention run is then **L20 λ=0 re-gated at
   n=12**, not the n=3 numbers (0.94 / 0.440 / 1.29), which are optimistic
   draws.
4. Skip the proposed λ ∈ {0.03, 0.1} sweep at depth 20 — at the current
   effect sizes it cannot produce a distinguishable result.
