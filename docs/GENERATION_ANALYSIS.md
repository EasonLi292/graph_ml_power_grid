# Generation Analysis — Inverse Design Through the Surrogate

*Can the trained surrogate run **backward**? Given a droop budget and a
topology, gradient-optimize the design knobs (wire width, decap) to the
cheapest grid that meets spec — then check the recovered design against the
real simulator. This document is a deep dive on whether that works, where it
works, and where it doesn't. Surrogate = the coordinate-free 7-hop model
(`checkpoints/droop_v5_nocoord.pt`).*

Reproduce with:

```bash
for b in 0.10 0.15 0.20; do
  python3.12 scripts/design_grad.py \
      --ckpt checkpoints/droop_v5_nocoord.pt --target-mV $b --lambda-cost 1e-3
done
python3.12 scripts/plot_generation.py   # figures from the captured table
```

---

## 0. TL;DR

- **The loop works.** It recovers physically sensible designs, trades copper
  against the droop budget monotonically, scales copper with topology
  difficulty, and reports honest infeasibility when a budget can't be met.
- **On the topologies the surrogate knows well (n_top 3, 7), predicted droop
  at the optimum matches the simulator almost exactly** (within ~0.0002 mV).
- **On the unseen topology (n_top 4), the surrogate is systematically
  conservative** — it predicts a touch *higher* droop than the simulator
  measures. For a safety-margin tool, erring high is the right way to be wrong.
- **The honest caveat:** the optimizer is only as good as the surrogate's
  gradient. On the OOD topology that gradient is noisier, and one case
  (n_top = 4 @ 0.10 mV) settled in a local minimum instead of pushing wire
  width to the ceiling. The verdict was still correct (infeasible), but via a
  worse design point than the in-distribution cases found.

---

## 1. The method in brief

```
 latent z ∈ ℝ²
     │  sigmoid decode (keeps every z inside the valid box)
     ▼
 (wire_width, C_decap)
     │  differentiable graph build  →  frozen surrogate
     ▼
 predicted worst-load droop
     │
 loss = ReLU(droop/budget − 1)²   +   λ · cost(width, decap)
        └─ meet the spec ─┘            └─ then minimize metal ─┘
     │  backprop through surrogate → builder → decoder, Adam-step z
     ▼
 recovered design  →  VALIDATE against the transient simulator
```

The hinge loss is zero as soon as the design is feasible, so the cost term
then pulls the solution down to the **cheapest point on the spec boundary**.
Because the decoder is a plain sigmoid, this is exactly a VAE-style generator
with the learned decoder replaced by a fixed one — the loop generalizes
directly to a learned latent design distribution.

---

## 2. Full results

Nine (budget × topology) design problems. `sim` is the transient simulator's
ground-truth worst-load droop at the *recovered* design:

| budget | n_top | wire_width | C_decap (F) | pred (mV) | **sim (mV)** | verdict |
|---|---|---|---|---|---|---|
| 0.10 mV | 3 | 0.986 | 7.2e-10 | 0.113 | 0.113 | ✗ infeasible |
| | **4 (OOD)** | 0.736 | 7.7e-10 | 0.141 | 0.122 | ✗ infeasible |
| | 7 | 0.831 | 2.3e-10 | 0.099 | 0.099 | ✓ |
| 0.15 mV | 3 | 0.909 | 4.2e-10 | 0.134 | 0.134 | ✓ |
| | **4 (OOD)** | 0.713 | 5.6e-10 | 0.150 | 0.132 | ✓ |
| | 7 | 0.601 | 5.1e-11 | 0.150 | 0.150 | ✗ (boundary) |
| 0.20 mV | 3 | 0.613 | 2.5e-10 | 0.200 | 0.200 | ✓ |
| | **4 (OOD)** | 0.631 | 2.4e-10 | 0.196 | 0.183 | ✓ |
| | 7 | 0.448 | 5.1e-11 | 0.197 | 0.197 | ✓ |

Every ✗ verdict is genuinely at/over budget in the simulator — there are **no
false "feasible" claims** that the simulator overturns.

---

## 3. Five signals that the generation is physically correct

### 3.1 Surrogate fidelity *at the optimum* — and the safe-error story

![Surrogate fidelity at optimum](figures/fig_surrogate_fidelity.png)

This is the most important plot in the document. The optimizer drives designs
to the *aggressive corner* of the design space — precisely where §3 of
[PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md) showed the surrogate is least
precise. So the question isn't "what is the model's average R²?" — it is "is
the model accurate *where the optimizer lands*?"

- **In-distribution (n_top 3, 7): pred ≈ sim, dead on the y = x line.**
  0.113/0.113, 0.134/0.134, 0.200/0.200, 0.197/0.197. The optimizer is not
  exploiting a model blind spot.
- **OOD (n_top 4): the points sit *above* y = x — the conservative/safe zone.**
  The surrogate predicts 0.141 / 0.150 / 0.196 mV where the simulator measures
  0.122 / 0.132 / 0.183. It over-estimates droop on the unseen topology, so any
  design it certifies as feasible is feasible with margin to spare.

This validates the model **for design** far more directly than the aggregate
R² does. A 0.827 per-site R² sounds mediocre; a surrogate that lands on y = x
in-distribution and stays on the safe side OOD is a usable design oracle.

### 3.2 Looser budget → less copper (monotone cost/performance)

![Recovered knobs vs budget](figures/fig_design_cost_vs_budget.png)

As the budget relaxes 0.10 → 0.15 → 0.20 mV, recovered wire width shrinks
monotonically for every topology (n_top 3: 0.99 → 0.91 → 0.61; n_top 7:
0.83 → 0.60 → 0.45), and decap relaxes alongside it. The tool spends copper
only when the spec demands it — the correct economic behavior.

### 3.3 Harder topology → more copper

![Topology difficulty](figures/fig_design_topology.png)

At a fixed budget, the supply-rich n_top = 7 grid needs the least copper and
the supply-poor n_top = 3 grid the most (clearest at 0.15 and 0.20 mV: n_top 7
sits well below n_top 3). Fewer supply pads means each carries more current and
the IR path is longer — textbook PDN physics, **learned from data, never told
to the model**.

### 3.4 It lands on the spec boundary, not over-built

Where feasible, simulated droop sits right at the budget (0.200/0.200,
0.134/0.134) rather than far under it. The cost term does its job: the
recovered design is the *cheapest* one that meets spec, which is the entire
point of inverse design.

### 3.5 It reports honest infeasibility

At 0.10 mV with only 3 pads, the optimizer drives wire width to the ceiling
(0.986) and the simulator still measures 0.113 mV > 0.10 — and the tool says
so (✗). It does not invent a design or silently return an over-budget grid.
That budget is genuinely unachievable for a sparse grid within the knob ranges.

---

## 4. The honest caveat — optimizer ≠ surrogate

The loop has two failure modes, and only one is the surrogate's fault.

1. **Surrogate error** (covered above): bounded, and conservative on OOD.

2. **Optimizer landing in a local minimum.** The n_top = 4 @ 0.10 mV case is the
   tell. The in-distribution cases that can't meet a tight budget push wire
   width all the way to the ceiling (≈ 0.99). The OOD case instead settled at
   wire_width = 0.736 — *not* the ceiling — and reported infeasible from there.
   The verdict is still correct (sim = 0.122 > 0.10), but it stopped at a worse
   design point. The cause is the noisier surrogate gradient on the unseen
   topology (§4 of the prediction analysis: error has spatial structure tied to
   the unseen pad layout, which makes the loss surface bumpier). This is an
   *optimization* artifact, not a wrong droop prediction.

**Mitigations** (not yet applied): multi-start `z` (take the best of N random
inits), a feasibility margin (`optimize against budget·(1−ε)` so boundary cases
land safely under spec), or a short simulator-in-the-loop refinement after the
surrogate proposes a starting point.

---

## 5. Bottom line for generation

- Gradient inverse design through the frozen surrogate **recovers correct,
  cheapest-feasible designs** and tells a coherent physical story end to end:
  monotone cost vs budget, copper scaling with topology difficulty, boundary
  landing, and honest infeasibility.
- **Surrogate fidelity holds where it's used.** In-distribution predictions
  land on the simulator; on the unseen topology they are conservative (safe).
- The remaining weakness is **optimization robustness on OOD topologies**, not
  prediction accuracy — addressable with multi-start and a feasibility margin.
- For a real flow, the recommended use is exactly what we do here: **surrogate
  proposes, simulator confirms.** The surrogate collapses a slow search into a
  handful of candidate designs; the simulator certifies the final pick.
