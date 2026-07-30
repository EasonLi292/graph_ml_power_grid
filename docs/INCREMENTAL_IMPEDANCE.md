# Incremental impedance updates, and whether the weak action families are learnable

Two independent questions, answered before the full 20k-pair generation:

1. Can a proposal be scored without refactorizing? **Yes — exactly, and it is
   implemented.**
2. Do the decap and load-frequency families contain a rankable signal?
   **Partly, and not the parts the pilot sampled.**

---

## 1. The update (`tools/incremental_impedance.py`)

Conductance enters the nodal admittance as `Y = Σ_e w_e a_e a_eᵀ`, so
changing r branches is a rank-r update `Y' = Y + A D Aᵀ`. Block Woodbury
gives the updated inverse **action** from the base factorization:

```
U    = Z A                                    (r back-substitutions, cached)
Z' B = Z B − U (I + D Aᵀ U)⁻¹ D (Aᵀ Z B)
```

`Aᵀ Z B` is read off `Z B` by index arithmetic (two non-zeros per column), so
it costs O(r k) rather than O(N r k). No N×N object is ever formed.

The whole factor pipeline touches `Y` only through `torch.linalg.solve(Y, ·)`,
so the update drops in as a `solve` override — `impedance_factors(...,
solvers=...)` and `dc_symmetric_factor(..., solver=...)`. Invariant channels,
Frobenius frequency normalization, the physics initialization, the dynamic
kernel and multi-frequency operation are all untouched, because none of them
sees `Y` at all.

### Rank by action family

| action | rank |
|---|---|
| one wire width / via resistance | 1 |
| wire section, strap, rail | number of **segments** |
| add or remove an edge | 1 (old or new admittance is 0) |
| resize / add / remove one decap | 1 at ω>0, **0 at DC** |
| move a decap | 2 |
| **global decap scaling** | **number of sites** — not low rank |
| any load change | 0 — sources are not in `Y`; bypassed entirely |

A "rail change" is cheap because a rail is a few segments, not because of its
name. Global decap scaling is deliberately supported and deliberately trips
the fallback.

### Two defects found by the checks, both worth recording

**`torch.linalg.cond` is useless here.** The core is 1×1 for every
single-branch action, and `cond` of a 1×1 matrix is identically 1.0 however
near zero the entry is — so the guard that exists to catch a disconnecting
edge removal would never have fired. Replaced with `smin(core) / max(1, ‖M‖)`.
Measured, the relationship is exact:

| d / d\* | core rcond | mode | rel err | err × rcond |
|---|---|---|---|---|
| 0.5 | 5.0e-01 | update | 8.7e-16 | 4.3e-16 |
| 0.9 | 1.0e-01 | update | 3.2e-15 | 3.2e-16 |
| 0.99 | 1.0e-02 | update | 2.9e-14 | 2.9e-16 |
| 0.9999 | 1.0e-04 | update | 4.1e-12 | 4.1e-16 |
| 1−1e-8 | 1.0e-08 | update | 3.0e-08 | 3.0e-16 |
| 1.0 | 0 | refactor | — | — |

`err × rcond` is pinned at machine epsilon across five decades, so
`err ≈ eps / rcond` and `rcond_min = 1e-8` buys ~1e-8 accuracy. That is where
the threshold comes from — not from a theoretical claim.

**A disconnecting proposal has no answer at all.** Isolating a node makes `Y`
exactly singular, so refactorization cannot rescue it either. It now raises
`SingularCircuit` with the diagnosis rather than returning a plausible-looking
solver. The core rcond is what detects it: removing a bridge of conductance g
gives `aᵀZa = R_eff = 1/g`, so the core hits exactly zero.

### Correctness: 73/73

`scripts/probes/woodbury_checks.py`. Update vs full refactorization:

* inverse action `Z'B`, all 17 action families — **3e-15 … 8e-15**
* invariant Re/Im Z channels — 1e-12
* Frobenius-normalized channels + hidden state — 3e-12 … 1e-11
* model output and per-load **change**, identical weights — 6e-14 … 1e-11
* gradients w.r.t. width and capacitance vs central differences — 8e-8 … 2e-5,
  including **at zero change**, where `d = 0` but `dd/dwidth ≠ 0` (this is why
  the non-symmetric core form is the default; the symmetric `D⁻¹` form is
  undefined there)
* 3 circuits × 7 magnitudes × 3 families — 1e-14
* interacting adjacent edges (core off-diagonal/diagonal 0.39 vs 0.08 for
  scattered) — 7e-15
* add / remove / move — 8e-15; a decap action provably reuses the DC operator
  object rather than rebuilding it
* ZA cache: 6 solves for 6 re-scorings of the same 2 branches, 30 hits
* 5 chained accepted updates still exact (1.6e-15), refactorizing at depth 4

Downstream checks are held to 1e-9 rather than 1e-15 **and the reason is not
the update**: `n_power` subspace iteration deliberately collapses `X` toward
the dominant subspace, so its columns are near-dependent and `qr()` rotates
`Qr` by far more than the 1e-15 perturbation that produced it.

The rank-16 **sketch** error against the true `Z` is 2.5e-01 (max entry over
all node pairs). That is the approximation the base path already pays; the
update neither adds to it nor removes it, and it is reported separately so the
1e-15 figure is never quietly credited with it.

---

## 2. Timing (`scripts/probes/woodbury_crossover.py`)

One proposal, end to end, proposal → model prediction, ms, CPU, batch 1:

| anchor | n_free | refac (shipped) | refac (LU) | update | vs refac(LU) | simulator | vs sim |
|---|---|---|---|---|---|---|---|
| (3,7) | 55 | 12.4 | 10.4 | 13.0 | 0.80× | 12.8 | 0.99× |
| (7,13) | 211 | 61.0 | 42.3 | 40.0 | 1.06× | 20.4 | 0.51× |
| (13,13) | 325 | 136.8 | 72.2 | 53.0 | 1.36× | 23.5 | 0.44× |
| (19,19) | 703 | 685.5 | 228.1 | 122.8 | 1.86× | 44.9 | 0.37× |
| (25,25) | 1225 | 2429.6 | 648.9 | 255.4 | **2.54×** | 86.2 | **0.34×** |

**Faster than rebuilding the factors: yes** — 2.5× at (25,25) against the fair
LU-reusing baseline, 9.5× against the path actually shipped today.

**Faster than the transient simulator: no.** Still ~3× slower, and the ratio
*worsens* with size. Where the 255 ms goes at (25,25):

| woodbury core | impedance_factors | dc_factor | model forward | total |
|---|---|---|---|---|
| **0.4** | 107.0 | 36.0 | 115.2 | 258.6 |

The update did its job — its own cost is 0.4 ms. What remains is (a) dense
back-substitution, O(N²) per solve, which needs the sparse factorization
already on the books, and (b) **the model forward alone, 115 ms, which already
exceeds the entire 86 ms simulation**. No amount of incremental-update work
can fix (b). Model and simulator both scale ~linearly here, so this is a
constant factor of ~1.3×, not something that closes at scale on CPU at batch 1.
Where the surrogate still wins is what the simulator cannot do at all:
gradients, and batched GPU evaluation.

Measured crossover in rank: **64 at (13,13)**, beyond 256 at (25,25) — the
bigger the grid, the more rank the update tolerates. `recommended_max_rank =
32`, half the smallest measured crossover, so the guard trips before the
update becomes the slower option. Below ~200 free nodes the update *loses*
(0.80× at (3,7)); it is a large-grid tool.

### The workload this was actually built for

The paired dataset is one base circuit plus K perturbations, which is exactly
the shared-base structure Woodbury wants. Factor cache for one group of 1 base
+ 8 perturbations:

| anchor | refactor | update | speedup | per group |
|---|---|---|---|---|
| (7,13) | 124 ms | 96 ms | 1.29× | 0.12 → 0.10 s |
| (13,13) | 292 ms | 174 ms | 1.67× | 0.29 → 0.17 s |
| (19,19) | 1506 ms | 615 ms | 2.45× | 1.51 → 0.61 s |
| (25,25) | 4991 ms | 2152 ms | 2.32× | 4.99 → 2.15 s |

The ~5 h full factor cache becomes ~2–2.5 h.

---

## 3. Are the weak families learnable? (`scripts/probes/action_audit.py`)

Simulator ground truth only — no model, so "the target is unlearnable" and
"the model failed" can be told apart. 252 simulations per anchor, 12 bases.

| anchor | family | disp | ties | stab | rho y0 lin | rho y0 log |
|---|---|---|---|---|---|---|
| (7,13) | decap_global | **0.152** | 0.000 | 0.551 | **0.955** | 0.561 |
| (7,13) | decap_resize_one | 1.560 | 0.002 | **−0.046** | 0.322 | 0.229 |
| (7,13) | decap_add_one | 1.575 | 0.002 | **0.758** | 0.294 | 0.211 |
| (7,13) | decap_remove_one | 1.717 | 0.001 | — | 0.284 | 0.222 |
| (7,13) | decap_move_one | 1.538 | 0.001 | — | 0.269 | 0.246 |
| (7,13) | decap_redistribute | 0.952 | 0.000 | 0.907 | 0.815 | **0.817** |
| (7,13) | load_freq | 0.257 | 0.000 | 0.860 | 0.774 | 0.647 |
| (13,13) | decap_global | **0.204** | 0.000 | 0.682 | **0.955** | 0.594 |
| (13,13) | decap_resize_one | 1.891 | 0.029 | **0.105** | 0.249 | 0.249 |
| (13,13) | decap_add_one | 1.189 | 0.004 | **0.986** | 0.330 | 0.247 |
| (13,13) | decap_remove_one | 1.559 | 0.016 | — | 0.351 | 0.261 |
| (13,13) | decap_move_one | 1.322 | 0.001 | — | 0.282 | 0.261 |
| (13,13) | decap_redistribute | 0.675 | 0.000 | 0.984 | 0.882 | **0.801** |
| (13,13) | load_freq | 0.215 | 0.000 | 0.862 | 0.936 | 0.583 |

`disp` = coefficient of variation of the relative change across loads — how
much there is to rank. `stab` = Spearman between |Δlog| rankings at two
magnitudes of the same action on the same circuit. `rho y0` = Spearman of the
change against the **base droop**, i.e. what "rank by which load already
droops most" achieves without learning anything.

### What this says

**`decap_global` was the worst possible choice of decap family, and it is the
only one the pilot sampled.** Its dispersion is 0.15–0.20, 8× lower than any
localized decap action: every load moves by nearly the same relative amount,
so there is almost nothing to rank. And 96% of its *linear* ranking is already
given by the base droop — which is the direct confirmation of the correction
made earlier: the ρ 0.92–0.98 originally reported for decap was essentially
the base-droop baseline, not learned knowledge.

**Placement actions are the learnable ones.** `decap_add_one` has dispersion
1.19–1.58, stability **0.76–0.99**, and a base-droop baseline of only
0.21–0.25. High signal, stable ordering, weak trivial baseline — the opposite
profile to global scaling on every axis.

**`decap_resize_one` is not usable as sampled.** Dispersion is the highest of
all (1.56–1.89) but stability is ~0 (−0.05, +0.11): the ordering does not
survive a change of magnitude. The reason is visible in the same row — the
effect is ~0.5% relative, so most loads sit at the resolution floor and their
mutual ordering is discretization noise. It needs larger factors, or to be
restricted to well-populated sites.

**`decap_redistribute` is learnable but 80% trivial** (rho y0 log 0.80–0.82).

### Frequency coverage is *not* the load_freq problem

The impedance factors are built on a fixed ω grid derived from the constant
`FIXED_FREQ`, never from each sample's load frequency. Measured directly:
the factor difference between a base and its `load_freq ×2` perturbation is
**exactly 0.000e+00**. A load-frequency action moves one scalar node feature
and nothing else.

The obvious fix — sample ω at the modified frequency — was tested and **does
not help**:

| feature ranking the simulated load_freq change | ρ |
|---|---|
| \|Z\| at the **modified** frequency (oracle) | 0.237 |
| \|Z\| at the **sampled** grid ω₀ (what the model has) | 0.242 |
| base droop y₀ (trivial baseline) | **0.601** |

The oracle frequency is no better than the sampled one, and both are far worse
than the trivial baseline. Over the acted range the driving-point impedance
only moves 0.81–1.23× with a 4–11% spread across loads — too little to rank
by. So of the four candidate explanations, for `load_freq` it is (b), targets
with almost no distinguishable relative variation, **not** (c) inadequate
frequency coverage. Adding frequency samples would be wasted effort.

(The ω grid being independent of the load frequency is still a genuine design
gap worth closing for *forward* accuracy — it just will not move this ranking.)

---

## 4. Data separation (`tools/factor_cache.py`)

Labels are a property of the circuit; factor caches are derived and
disposable. The old cache key was `tag_m{m}_q{n_power}_f{n_freq}_c{n_ch}`,
which omitted the probe seed and the actual frequency **values** — so two
different 3-frequency grids hashed identically and a run could silently load
factors built for the other grid. `FactorSpec` now hashes every input and
carries a format version:

```
grid [0, 1e9,  5e9] -> v2_m16_q2_f3_s0_herm_fdc_6863744a4b
grid [0, 1e9, 25e9] -> v2_m16_q2_f3_s0_herm_fdc_34f762fe1d
same grid, seed 1   -> v2_m16_q2_f3_s1_herm_fdc_b4d444c3cb
```

`assert_labels_are_independent()` fails loudly if a label file ever picks up a
factor hyper-parameter; `datasets/paired_v1/pairs.h5` passes. `purge()` makes
disposal one call, so changing rank or normalization never implies
re-simulating.

Per-site decap capacitance is now supported end to end (`PDNGraph.decap_C`,
`build_regular_pdn(C_decap_sites=...)`, `transient_solver`), verified exactly
equal to the scalar path when uniform (0.000e+00). Since decap sites already
exist at every row × boundary, a site holding zero capacitance is an empty
slot — which is what makes add/remove/move ordinary rank-1 value changes
rather than topology edits.

---

## 5. Recommended action schema

| family | rank | keep? | why |
|---|---|---|---|
| `ww_edge` | 1 | yes, more of it | weakest wire family in the pilot (n=21) |
| `ww_section` | k | yes | ρ +0.59 |
| `ww_strap` | k | yes | ρ +0.68, lift +0.10 |
| `ww_multi` | k | yes | ρ +0.86, lift +0.13 |
| `ww_top_strap` | k | yes | ρ +0.57, lift +0.14 |
| `via_resistance` | 1 | **add** | supported, never sampled |
| `decap_add_one` | 1 | **add, primary** | disp 1.19–1.58, stab 0.76–0.99, baseline 0.21 |
| `decap_remove_one` | 1 | **add** | disp 1.56–1.72, baseline 0.22–0.26 |
| `decap_move_one` | 2 | **add** | disp 1.32–1.54, baseline 0.25–0.26 |
| `decap_redistribute` | k | add, low weight | stable but 80% trivial |
| `decap_resize_one` | 1 | only at ≥4× on populated sites | unstable at the sampled magnitudes |
| `decap_global` | S | **demote to calibration only** | disp 0.15, 96% trivial in linear space |
| `load_one` | 0 | yes | ρ +0.81 |
| `load_freq` | 0 | **demote to forward-only** | not rankable; frequency coverage is not the cause |

Requires storing per-site capacitance in `pairs.h5` (`C_decap_sites`), which
the current writer does not yet do — it stores a scalar `C_decap`.

## 6. Go / no-go

**Go, with the schema changed first.** The three things the audit was meant to
decide:

* Generation is now ~2.3× cheaper on the factor side at the large anchors, and
  correctness is not in question (73/73, 1e-15).
* The weak families are weak for a *measurable* reason, and for decap that
  reason is fixable by changing which action is sampled — placement, not
  global scaling. Generating 20k more pairs of `decap_global` would buy
  nothing.
* `load_freq` ranking is not fixable by frequency coverage and should not be
  sold as a ranking target.

**No-go on the standing assumption that a fast forward model makes
propose-and-score cheaper than simulating.** At batch 1 on CPU it does not,
and the model forward alone exceeds the simulator. That does not block dataset
generation — labels come from the simulator regardless — but it does mean the
"propose a change and see its effect immediately" argument needs either the
sparse factorization, GPU batching, or the gradient (which the simulator
cannot provide at all) to stand up.
