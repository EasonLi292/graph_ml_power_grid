# Sobolev results — gradient supervision helps, reach is the wall

GPU execution of `docs/SOBOLEV_HANDOFF.md` (step 1 of `docs/OBJECTIVE.md`) on
the H100. Acceptance test is `scripts/sensitivity_gate.py` (bar: sign ≥ 0.95,
site-rank ρ ≥ 0.8, **per anchor**, including held-out).

Everything was rebuilt from scratch on this box — the dataset and the Jacobian
labels are gitignored, so this is an independent reproduction of the handoff's
local validation, not a reuse of its artifacts.

## Provenance

- `datasets/regular_v7_anchors/dataset.h5` rebuilt (`--per-edge`, 16k/2k/2k,
  seed 42, sweeps at 50 pts). Train anchors (3,7) (7,7) (5,13) (13,13);
  test anchors (4,7) (7,13).
- `jacobians.h5` generated for the **full** 16k train split (458 s on 96
  workers). Cross-check of the torch-sim droop against the stored dataset
  labels: **median rel 0.0, max 1.2e-7** — reproduces the handoff's
  ≤ 1.2e-7 claim exactly.
- All runs: `--conv-type edgeconv --epochs 50`, defaults otherwise
  (lr 1e-3, batch 32, log target, hidden 64).

## Forward accuracy — the Sobolev term is not a tax, it is a *gain* OOD

| run          | val R² | test R² (held-out anchors) | test worst-MAE mV |
|--------------|--------|----------------------------|-------------------|
| λ=0 control  | 0.9973 | 0.7927                     | 0.112             |
| **λ=0.3**    | —      | **0.8552**                 | **0.104**         |
| λ=3          | —      | 0.7472                     | 0.161             |
| λ=30         | —      | 0.8445                     | 0.107             |

Gradient matching costs nothing in-distribution (all runs reach val R² ≈ 0.995
by epoch ~16) and **improves held-out-anchor forward transfer** at λ=0.3
(0.793 → 0.855 test R²). That was not a predicted outcome — the handoff
expected forward MAE to get slightly *worse*.

## The gate — better, still failing

3 designs × (12 bot + 6 top) edges × +25 % width, decap ×2/×0.5, seed 0.
`sign / site-rank ρ`; T = train anchor, O = held out.

| checkpoint            | (3,7) T          | (4,7) O          | (7,13) O         | (7,13) mag |
|-----------------------|------------------|------------------|------------------|-----------|
| v7 edgeconv (previous)| 0.79 / 0.745     | 0.88 / 0.766     | 0.29 / 0.017     | 0.00      |
| λ=0 control           | 0.83 / 0.823     | 0.91 / 0.825     | 0.12 / −0.109    | 0.00      |
| **λ=0.3**             | 0.85 / **0.893** | 0.79 / 0.669     | 0.29 / 0.212     | 0.00      |
| λ=3                   | 0.83 / 0.827     | 0.44 / −0.234    | **0.59 / 0.385** | 0.08      |
| λ=30                  | 0.77 / 0.725     | 0.88 / 0.770     | 0.35 / 0.132     | 0.00      |

Decap-direction accuracy is 1.00 for every checkpoint at every anchor, as
before. All rows **FAIL** the gate.

### What gradient supervision bought

1. **Best site-ranking ever measured** at the train anchor: ρ = 0.893 at
   λ=0.3, clearing the ρ ≥ 0.8 half of the bar there (sign 0.85 does not).
2. **The big die stops being anti-correlated.** Every λ > 0 lifts (7,13) off
   the control's negative ρ (−0.109 → +0.13…+0.39). λ=3 is best there
   (0.385) — and worst at (4,7) (−0.234), so no single λ wins everywhere.
3. **Sign accuracy is still the binding constraint.** The trivial rule
   "widening always helps" scores ~0.72; the best model is 0.85. One edge
   move in six still goes the wrong way, so a verifier stays mandatory.

### What it did not buy: reach

At (7,13) the magnitude ratio is **0.00–0.08** across all four checkpoints —
the model's response to a distant single-edge change is essentially zero, so
the ranking it produces there is noise around zero, not a weak signal. With
7 message-passing hops a random edge on a 13×13 die is simply outside the
worst load's receptive field; no supervision on a quantity the architecture
cannot represent can fix that.

This is exactly the branch the handoff called: *"if (7,13) stays at ρ ≈ 0
while (3,7)/(4,7) improve, the bottleneck is the 7-hop receptive field, not
supervision."* Confirmed — with the refinement that (4,7) did **not**
improve, so λ trades near-anchor fidelity against far-anchor reach.

## Depth — the reach fix, confirmed and bounded

Crossing a (7,13) die needs ≈ 16 hops (≈ 24 on (13,13)); the stack has 7. Two
20-layer runs tested this directly (λ=0 isolates depth, λ=0.3 combines it with
the best gradient supervision), gated with `--n-layers 20`:

| checkpoint    | (3,7) T          | (4,7) O          | (7,13) O         | (7,13) mag | test R² |
|---------------|------------------|------------------|------------------|-----------|---------|
| L7  λ=0       | 0.83 / 0.823     | 0.91 / 0.825     | 0.12 / −0.109    | 0.00      | 0.793   |
| L7  λ=0.3     | 0.85 / 0.893     | 0.79 / 0.669     | 0.29 / 0.212     | 0.00      | 0.855   |
| **L20 λ=0**   | 0.88 / 0.860     | 0.41 / −0.176    | **0.94 / 0.440** | **1.29**  | **0.905** |
| L20 λ=0.3     | 0.88 / **0.889** | 0.41 / −0.068    | 0.71 / 0.199     | 0.12      | 0.815   |

**Reach was the binding constraint on the big die — confirmed.** At 20 hops the
(7,13) magnitude ratio goes 0.00 → **1.29** (ideal 1.0) and sign 0.12 → **0.94**,
essentially at the 0.95 bar. The autograd-linearity check also goes from a
degenerate 0.00 % to a healthy 18–24 %, i.e. the model finally has non-vanishing
sensitivities to perturb. Depth alone also gives the **best forward transfer
measured on this track** (test R² 0.905 vs 0.793 at 7 layers).

**But depth and gradient supervision do not compose.** Stacking λ=0.3 on the
20-layer model keeps the train-anchor ranking (ρ 0.889, the best recorded) yet
gives back most of the big-die gain (sign 0.94 → 0.71, mag 1.29 → 0.12) and
costs forward transfer (R² 0.905 → 0.815). The gradient term and the deep
stack are pulling the same weights in different directions.

**The binding constraint has moved to the small held-out anchor.** Both
20-layer runs collapse at (4,7) — sign 0.41, ρ ≈ 0 — where the 7-layer models
were their strongest (0.91 / 0.825). (4,7) is an *interpolation* between the
train anchors (3,7) and (7,7); 20 hops on a 7-row die is ~3× more propagation
than the die is wide, and it over-smooths.

Deep stacks train stably (val R² 0.998 at 20 layers, better than 7's 0.997),
so this is a cost question, not a stability one: ~2.9× per-epoch.

## Where this leaves the gate

Still **FAIL everywhere** — the repair harness (OBJECTIVE step 4) remains
blocked. But the failure mode has changed shape:

- *Was:* "the model has no derivative signal at all on big dies" (mag 0.00).
- *Now:* "no single fixed depth serves both die sizes" — 7 hops is right for
  (4,7) and blind on (7,13); 20 hops is right for (7,13) and over-smooths (4,7).

That is a direct argument for **OBJECTIVE step 2 (the global/attention term)**
over further depth tuning: an attention/impedance-sketch term gives O(1) reach
independent of die size, so it can serve both anchors at once, where a fixed
hop count structurally cannot. Recommended next experiment: port the
superposition-attention global term to the synthetic model at 7 local layers,
and re-gate — the comparison to beat is L20 λ=0 at (7,13) (0.94 / 0.440 / 1.29)
while holding (4,7) at the 7-layer level (0.91 / 0.825).

Secondary, cheap: since λ and depth trade off per anchor, a per-anchor-reported
λ sweep *at* depth 20 (λ ∈ {0.03, 0.1}) may recover the (3,7) ranking gain
without spoiling the big-die magnitude — but this is tuning, not a fix for the
structural reach/smoothing conflict.
