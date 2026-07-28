# Item 2 results — one-shot impedance attention does not clear the gate

Execution of `docs/ITEM2_RUNBOOK.md` (OBJECTIVE step 2). Verdict up front:

> **The hypothesis is not confirmed.** The combined model gets the O(1)-reach
> property it was designed for — non-zero distant sensitivity at (7,13),
> magnitude ratio **0.97** where the 7-layer stack scores **0.01** — but it
> does not convert that reach into usable derivatives: sign 0.63 and
> ρ +0.01 at (7,13), and it is *below* both local baselines at (4,7).
> All three ablations FAIL the gate at all three anchors.
>
> Secondary and more consequential: the runbook's premise —
> "no fixed depth serves both die sizes" — **did not reproduce**. A freshly
> trained 20-layer local baseline is the best model on this page at every
> anchor and shows none of the documented (4,7) collapse.

## Provenance — everything rebuilt here

The runbook's two prereqs (`dataset.h5`, `droop_v7_edgeconv*.pt`) are
gitignored and were absent, so both were rebuilt from scratch on this box
(64 cores, 4×RTX 3090; the prior results were produced on an H100).

- Dataset: `--per-edge`, 16k/2k/2k, seed 42, sweeps 50 pts — attrs confirm
  train anchors (3,7)(7,7)(5,13)(13,13), test (4,7)(7,13). Build: 4 min.
- Baselines retrained, `--conv-type edgeconv --epochs 50`, defaults
  otherwise. **The rebuild reproduces the original track**: L7 val R²
  **0.9973** vs the documented 0.9973; L20 val R² 0.9995 vs 0.998. Pooled
  per-load test R² came out slightly better than published (L7 0.849 vs
  0.793; L20 0.931 vs 0.905).
- Factor cache warmed first per runbook §0: 326 s for 16k train + 39 s for
  2k test, single job, then the three configs fanned out. No cache race.
- Epoch cost: combined/impedance ~100 s, content ~42 s, L7 30 s, L20 72 s.
  Confirmed the runbook's call that the GPU buys little — a 1000-sample
  timing test gave CPU 8 s/epoch vs CUDA 6.5 s/epoch. All three attention
  configs were run on CPU so the ablation shares a device exactly.

## 1. Forward accuracy — per-anchor worst-load R², identical metric

`--compare-baseline` puts the local baselines through the attention
harness's metric, so every column here is comparable (runbook §3).

| model | (4,7) R² | (7,13) R² | (4,7) worst-MAE | (7,13) worst-MAE |
|-------|----------|-----------|-----------------|------------------|
| L7 local (7 layers)     | −1.715 | −0.098 | 0.109 mV | 0.117 mV |
| L20 local (20 layers)   | −1.007 | **+0.930** | 0.093 mV | **0.026 mV** |
| imp-attn **combined**   | +0.598 | +0.008 | 0.033 mV | 0.105 mV |
| imp-attn **content**    | **+0.779** | +0.368 | **0.023 mV** | 0.087 mV |
| imp-attn **impedance**  | +0.551 | +0.242 | 0.035 mV | 0.092 mV |

The runbook's preliminary smoke signal **holds against L7**: every attention
config beats the 7-layer baseline at both held-out anchors simultaneously,
which is the combination no 7-layer model achieves. It does **not** hold
against L20, which wins (7,13) by a wide margin (+0.930 vs +0.008).

### All three configs peak early and then decay

Held-out R² is not monotone — it peaks between epoch 6 and 21 and decays
through epoch 50, while train loss keeps falling:

| config | best joint epoch | (4,7) | (7,13) | → at ep50 |
|--------|------------------|-------|--------|-----------|
| combined  | 9  | +0.675 | +0.703 | +0.598 / +0.008 |
| content   | 17 | +0.860 | +0.881 | +0.779 / +0.368 |
| impedance | 21 | +0.629 | +0.739 | +0.551 / +0.242 |

`train_impedance_attention.py` overwrites one checkpoint per epoch and does
no selection (the val split is loaded but never used to select), so the
gate as specified judges a checkpoint well past its held-out peak. **This
was checked, not assumed** — see §3.

## 2. The gate at 12 designs, seed 0 — every row FAILS

Bar: sign ≥ 0.95 and ρ ≥ 0.8 per anchor, CI *lower* bound must clear it.

| model | anchor | sign (95% CI) | ρ (95% CI) | mag | verdict |
|-------|--------|---------------|------------|-----|---------|
| L7  | (3,7)  | 0.84 [0.78,0.89] | +0.839 [+0.78,+0.89] | 0.86 | FAIL |
| L7  | (4,7)  | 0.59 [0.50,0.67] | +0.109 [−0.13,+0.36] | 0.55 | FAIL |
| L7  | (7,13) | 0.37 [0.26,0.49] | +0.099 [−0.09,+0.28] | **0.01** | FAIL |
| **L20** | (3,7)  | **0.94** [0.90,0.97] | **+0.947** [+0.93,+0.97] | 1.00 | FAIL |
| **L20** | (4,7)  | **0.88** [0.81,0.92] | **+0.709** [+0.62,+0.80] | 1.54 | FAIL |
| **L20** | (7,13) | **0.76** [0.65,0.85] | **+0.419** [+0.18,+0.64] | 0.60 | FAIL |
| combined  | (3,7)  | 0.63 [0.55,0.69] | +0.411 [+0.22,+0.56] | 1.17 | FAIL |
| combined  | (4,7)  | 0.56 [0.47,0.65] | +0.168 [−0.00,+0.32] | 1.82 | FAIL |
| combined  | (7,13) | 0.63 [0.51,0.74] | +0.009 [−0.17,+0.18] | **0.97** | FAIL |
| content   | (3,7)  | 0.46 [0.39,0.53] | +0.130 [−0.02,+0.29] | 1.32 | FAIL |
| content   | (4,7)  | 0.51 [0.42,0.60] | +0.068 [−0.14,+0.29] | 2.13 | FAIL |
| content   | (7,13) | 0.40 [0.29,0.52] | +0.025 [−0.15,+0.20] | 0.74 | FAIL |
| impedance | (3,7)  | 0.48 [0.41,0.55] | +0.227 [+0.10,+0.36] | 0.93 | FAIL |
| impedance | (4,7)  | 0.50 [0.42,0.59] | +0.007 [−0.16,+0.17] | 1.70 | FAIL |
| impedance | (7,13) | 0.56 [0.44,0.67] | +0.051 [−0.14,+0.23] | 0.82 | FAIL |

Decap-direction accuracy, 1.00 for every local checkpoint historically,
**degrades under the combined model**: 0.79 at (4,7), 0.92 at (3,7). The
attention configs are the first checkpoints on this track to get decap
direction wrong.

### Against the runbook's three decision criteria (§4), for `combined`

1. **(7,13) non-zero distant sensitivity — PARTIALLY met.** Magnitude ratio
   0.97 vs the 7-layer 0.01, so the reach claim is real and is the one thing
   the architecture demonstrably delivers. But the criterion also requires
   sign "well above the ~0.72 trivial rate", and sign is
   **0.63 [0.51, 0.74]** — below 0.72, with the CI upper bound barely
   touching it. Reach without direction. **Not met.**
2. **(4,7) no over-smoothing — NOT met.** Required sign/ρ at or above the
   7-layer level (0.86 / 0.65 as documented). Combined scores
   **0.56 / +0.168**. Against the *same-box* L7 (0.59 / +0.109) it is
   statistically indistinguishable, not above.
3. **Both simultaneously — NOT met**, since neither is met alone.

## 3. The early-peak confound, tested and closed

Since held-out R² peaks near epoch 6–9, the ep50 checkpoint the gate judged
is past peak. To rule this out, `combined` and `content` were retrained for
25 epochs with `--keep-epochs` (a new flag added for this) and the
best-joint-R² epoch was gated directly. Selecting on held-out anchors is
test-set selection, so these are an **upper bound**, not a legitimate model:

| checkpoint | (3,7) sign/ρ | (4,7) sign/ρ | (7,13) sign/ρ |
|------------|--------------|--------------|---------------|
| combined ep50 | 0.63 / +0.411 | 0.56 / +0.168 | 0.63 / +0.009 |
| combined **peak (ep6)** | 0.53 / +0.164 | 0.55 / +0.082 | 0.56 / +0.063 |
| content ep50 | 0.46 / +0.130 | 0.51 / +0.068 | 0.40 / +0.025 |
| content **peak (ep9)** | 0.45 / +0.221 | 0.79 / +0.516 | 0.32 / −0.035 |

The peak checkpoints are **not better** — combined at its forward peak is
slightly *worse* on the gate than at ep50. The failure is structural, not a
checkpoint-selection artifact. (Note also that better forward R² does not
track better derivatives: content's peak has the best forward numbers on
this page and ρ = −0.035 at (7,13).)

## 4. What the ablations say

- **The physics term is not decorative.** `combined` beats both ablations on
  gate quality at every anchor — ρ +0.411 at (3,7) vs +0.130 (content) and
  +0.227 (impedance); sign 0.63 vs 0.46 / 0.48. Neither term alone
  reproduces the combined derivative behaviour, so the design note's
  "if content ≈ combined the factors aren't paying rent" branch does not fire.
- **But it costs forward accuracy.** `content` is the *best* forward model of
  the three (+0.779 / +0.368 vs combined's +0.598 / +0.008). The impedance
  score term buys derivative sign/ranking and pays for it in fit.
- Caveat on the naming: `content` is not physics-free. The invariant
  self-impedance scalar `selfz` is still an encoder input
  ([impedance_attention_model.py:173](eason/impedance_attention_model.py#L173)),
  so the ablation removes the pairwise impedance *score*, not all knob
  dependence. That is why content still shows non-zero magnitude ratios.

## 5. The premise did not reproduce — and the variance is larger than the CIs

The runbook and design note rest on: 7 hops is right for (4,7) and blind on
(7,13); 20 hops is right for (7,13) and over-smooths (4,7). Retrained here
at n=12, **that trade-off is absent**:

| anchor | published L20 | L20 retrained here |
|--------|---------------|--------------------|
| (3,7)  | 0.88 / +0.860 | 0.94 / +0.947 |
| (4,7)  | **0.41 / −0.176** ("collapse") | **0.88 / +0.709** |
| (7,13) | 0.94 / +0.440 | 0.76 / +0.419 |

L20 is now the strongest model at *every* anchor, and the (4,7) collapse
that motivated this entire work item does not appear. The reverse happened
at L7, whose (4,7) row moved the other way (published re-gate 0.86 / +0.654
→ **0.59 / +0.109** here).

These gaps are far outside the reported 95 % CIs. That is not a
contradiction — the gate's CIs are a **design-level bootstrap for a fixed
checkpoint**, so they quantify perturbation sampling only. Retraining the
same config with the same hyperparameters moves (4,7) sign by ±0.3, an order
of magnitude more than the ±0.06 design-sampling interval. **Seed-to-seed
retraining variance is the dominant uncertainty on this track and is
currently unmeasured**, which means the depth conclusions in
`SOBOLEV_RESULTS.md` (already downgraded once by
`SOBOLEV_RESULTS_REVIEW.md` for the n=3 → n=12 issue) rest on single draws
of a quantity that moves this much.

## 6. Recommendation

1. **Do not proceed to OBJECTIVE items 3/5 on this architecture.** It fails
   its own falsification criteria, and its best-case (oracle-selected)
   checkpoint fails them too.
2. **Measure retraining variance before any further architecture work** —
   3–5 seeds of L7 and L20, gated at n=12. Everything on this track is
   currently being decided by differences smaller than the unmeasured
   seed spread. This is cheap (~30 min/seed for L7, ~60 for L20) and it
   determines whether there is a depth problem to solve at all.
3. **The reach claim survives and is worth keeping.** Magnitude ratio 0.97
   at (7,13) against L7's 0.01 is a real, large, structural effect and the
   only criterion the architecture met. If the seed study confirms a genuine
   depth trade-off, the productive combination is the impedance factors on
   top of a local stack (the design note assumed *replacement*; nothing
   tested *augmentation*), not one global layer alone — the one-layer model
   has reach but no directional accuracy, and the deep local model has
   direction but limited reach.
4. Add checkpoint selection to `train_impedance_attention.py`. Held-out R²
   decaying 0.70 → 0.01 at (7,13) while train loss falls means the final
   epoch is the wrong model to ship or to judge.

## Artifacts

`docs/analysis/sensgate_{n12_l7,n12_l20,imp_combined,imp_content,imp_impedance,imp_combined_peak,imp_content_peak}.json`,
`checkpoints/{droop_v7_edgeconv_l0,droop_v7_edgeconv_deep20_l0,imp_attn_*}.pt`
plus `.history.json` for every run, `logs/*.log`. Dataset and checkpoints
are gitignored; the JSONs and histories are the auditable record.

The per-epoch `imp_attn_*_peak.epNNN.pt` files from §3 were deleted after
gating; regenerate with
`scripts/train_impedance_attention.py --ablation combined --epochs 25 --keep-epochs`
(~45 min, deterministic at seed 0) if the peak checkpoints are needed again.
`--keep-epochs` is the only code change made during this run.
