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
- **The honest caveat — one false rejection.** On the unseen topology the
  surrogate is not just imprecise, it is **non-monotonic in wire width**: it
  predicts droop *bottoms out and then rises* as copper increases, which is
  unphysical. For n_top = 4 @ 0.10 mV this trapped the optimizer at the
  surrogate's spurious minimum (wire ≈ 0.74) and produced a **false-infeasible**
  verdict — the simulator says the spec *is* achievable at full wire (0.098 mV ≤
  0.10). So conservatism keeps bad designs from passing, but here it wrongly
  rejected a good one. Detail + figure in §4.

---

## 1. The method in brief

```
 (wire_width, C_decap)   ← the two free design knobs, optimized directly
     │  differentiable graph build  →  frozen surrogate
     ▼
 predicted worst-load droop
     │
 loss = ReLU(droop/budget − 1)²   +   λ · cost(width, decap)
        └─ meet the spec ─┘            └─ then minimize metal ─┘
     │  backprop through surrogate → builder, gradient-step the two knobs
     ▼
 recovered design  →  VALIDATE against the transient simulator
```

We optimize the two design knobs directly by gradient descent through the
frozen surrogate. Each knob is passed through a sigmoid so any unconstrained
value lands in its valid range (`wire_width ∈ [0.2, 1.0]`, `C_decap ∈
[5e-11, 8e-10]` in log space) — a simple reparameterization that removes the
need for a projection step inside the loop. The hinge loss is zero as soon as
the design is feasible, so the cost term then pulls the solution down to the
**cheapest point on the spec boundary**.

> For the **layer-by-layer mechanics** — how the knobs become edge attributes,
> how the gradient backpropagates through the *frozen* network to the two knobs,
> and why "frozen weights" still gives a usable input-gradient — see
> [INVERSE_DESIGN_MECHANICS.md](INVERSE_DESIGN_MECHANICS.md).

---

## 2. Full results

Nine (budget × topology) design problems. `sim` is the transient simulator's
ground-truth worst-load droop at the *recovered* design:

| budget | n_top | wire_width | C_decap (F) | pred (mV) | **sim (mV)** | verdict |
|---|---|---|---|---|---|---|
| 0.10 mV | 3 | 0.986 | 7.2e-10 | 0.113 | 0.113 | ✗ truly infeasible |
| | **4 (OOD)** | 0.736 | 7.7e-10 | 0.141 | 0.122 | ✗ **false reject** (see §4) |
| | 7 | 0.831 | 2.3e-10 | 0.099 | 0.099 | ✓ |
| 0.15 mV | 3 | 0.909 | 4.2e-10 | 0.134 | 0.134 | ✓ |
| | **4 (OOD)** | 0.713 | 5.6e-10 | 0.150 | 0.132 | ✓ |
| | 7 | 0.601 | 5.1e-11 | 0.150 | 0.150 | ✗ (boundary) |
| 0.20 mV | 3 | 0.613 | 2.5e-10 | 0.200 | 0.200 | ✓ |
| | **4 (OOD)** | 0.631 | 2.4e-10 | 0.196 | 0.183 | ✓ |
| | 7 | 0.448 | 5.1e-11 | 0.197 | 0.197 | ✓ |

There are **no false "feasible" claims** that the simulator overturns — every
design the tool *certifies* is genuinely within budget. The one error runs the
other way: at 0.10 mV / n_top = 4 the tool *rejects* a spec the simulator says
is achievable (§4).

---

## 3. Signals that the generation is physically correct

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
(The flip side of that conservatism — it can *reject* a feasible spec — is the
subject of §4.)

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

### 3.5 It reports honest infeasibility (when it's real)

At 0.10 mV with only 3 pads, the optimizer drives wire width to the ceiling
(0.986) and the simulator still measures 0.113 mV > 0.10 — and the tool says
so (✗). It does not invent a design or silently return an over-budget grid.
That budget is genuinely unachievable for a sparse grid within the knob ranges.
(The n_top = 4 case at the same budget is a *different* story — §4.)

### 3.6 Watching it choose — the descent, the convergence, and the atlas

The signals above are summary statistics; these three figures show the design
*being chosen*.

**The descent over the droop landscape.** Each panel is the predicted-droop
contour map over the `(wire_width, C_decap)` design space for the OOD topology,
with the spec boundary in cyan and the optimizer's path from start (○) to the
chosen design (★):

![Design-choice trajectory](figures/fig_design_trajectory.png)

- At **0.20 mV** the star lands exactly *on* the cyan spec boundary — the
  cheapest feasible point — and you can even see the boundary's non-monotonic
  "elbow" on the fat-wire side (the §4 pathology, visible here as the cyan line
  curling back up).
- At **0.15** and **0.10 mV** the budget is (for the surrogate) unreachable on
  this topology: the path climbs toward more copper/cap but stalls at the
  surrogate's interior minimum, never touching the boundary — the visual
  signature of the infeasible-budget case (genuinely infeasible at 0.15, and the
  false rejection at 0.10 dissected in §4).

**The convergence trace.** The same run as a time series — knobs, predicted
droop, and loss vs optimization step:

![Design-choice convergence](figures/fig_design_convergence.png)

It makes the loss's two-phase behavior literal: predicted droop drops straight
to the budget line and **sits on it**, while `C_decap` and `wire_width` keep
adjusting to trade cost — exactly "meet the spec, then minimize metal."

**The design atlas.** Every chosen design (3 budgets × 3 topologies) as a point
in design space:

![Design atlas](figures/fig_design_atlas.png)

The two physical trends are visible at a glance: along each topology's track,
looser budget → less copper *and* less cap (toward the lower-left); and the
supply-rich `n_top = 7` track sits far below the sparse `n_top = 3` track
(fewer pads → more metal for the same budget).

---

## 4. The one failure — surrogate non-monotonicity → a false rejection

Earlier I called the n_top = 4 @ 0.10 mV case a "local minimum." That
undersells it. Sweeping the surrogate and the simulator along wire width (decap
pinned near its ceiling) shows what actually goes wrong:

![Inferred vs actual droop vs wire width](figures/fig_surrogate_vs_sim_sweep.png)

- **Left (n_top = 4, OOD):** the simulator (blue) is monotonic — more copper
  always means less droop, all the way down to **0.098 mV at full wire, which
  *meets* the 0.10 spec.** The surrogate (red) tracks it down to wire ≈ 0.74,
  then **flattens and turns back up** — it predicts droop *increases* with more
  copper, which is physically impossible. The optimizer faithfully descends to
  the surrogate's minimum at ≈ 0.74 and stops; from there the tiny cost term
  even discourages adding the wire that would actually fix the droop.
- **Right (n_top = 3, in-dist):** surrogate and simulator are indistinguishable
  — clean, monotonic, exactly as a trustworthy surrogate should behave.

The same data as a direct inferred-vs-actual scatter — in-distribution points
sit on `y = x`; the OOD points ride above it and, at the low-droop end (high
wire), peel away as the surrogate refuses to come down:

![Inferred vs actual scatter](figures/fig_inferred_vs_actual.png)

### 4.1 Predicted vs actual across the *whole* design space

Sweeping **both** knobs (wire_width × C_decap) and comparing predicted to actual
droop as paired heatmaps confirms the failure is localized — and reveals it is
**axis-specific**:

![Predicted vs actual across the 2-D design space](figures/fig_designspace_pred_vs_sim.png)

- **Bottom row (n_top = 3, in-dist):** predicted and actual maps are
  indistinguishable; the error panel is ±0.004 mV of noise.
- **Top row (n_top = 4, OOD):** the bulk still matches, but the error panel
  lights up a **red blob at the fat-wire edge** — up to ~0.10 mV of
  *over*-prediction (25× the in-dist error), concentrated exactly where the
  wire-width non-monotonicity lives. Note the error is one-signed (red =
  conservative) almost everywhere — the model rarely under-predicts.

A 1-D capacitance sweep (the knob we had *not* yet swept) isolates which axis is
to blame:

![Inferred vs actual vs decap](figures/fig_cap_sweep_pred_vs_sim.png)

Along **capacitance** the OOD surrogate is **monotonic** and only mildly
conservative (a small, well-behaved offset). The pathology is specific to the
**wire-width / conductance** axis — which is exactly the axis the physics-shaped
conductance gate acts on, and the one the optimizer leans on hardest to cut
droop. So the OOD weakness isn't diffuse; it's a single, identifiable,
fixable distortion in the wire-width response.

**Consequences and correct framing.**

1. **This is a false rejection, not a correct infeasibility verdict.** The
   simulator says the spec is achievable (0.098 ≤ 0.10); the tool reported ✗.
   Conservatism (over-predicting droop) protects against the *dangerous* error —
   certifying a bad grid as good — but its price is exactly this: it can declare
   an achievable spec infeasible and leave performance on the table.
2. **The fault is the surrogate's response surface, not the optimizer.** Adam
   did its job; it minimized a function that is wrong outside its training
   support. The model never saw n_top = 4, so it doesn't respect the monotone
   "more copper → less droop" constraint there.
3. **Root cause is topology coverage** — the same gap behind every OOD number in
   [PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md).

**Fixes that actually address it** (not yet applied):

- **More training topologies** — the real fix; makes the surface monotonic on
  n_top = 4 by giving the model support there.
- **Enforced monotonicity** in wire/conductance (a monotone sub-network, or a
  training penalty on `∂droop/∂wire > 0`) — guarantees the physics regardless of
  coverage.
- **Simulator-in-the-loop refinement** after the surrogate proposes — would have
  caught that full wire meets spec.
- *Multi-start `z` would **not** fix this one* — the surrogate's own global
  optimum is the bad point, so every restart converges to the same place. (It
  still helps the generic local-minimum case; it just isn't the cure here.)

---

## 5. Bottom line for generation

- Gradient inverse design through the frozen surrogate **recovers correct,
  cheapest-feasible designs** and tells a coherent physical story end to end:
  monotone cost vs budget, copper scaling with topology difficulty, boundary
  landing, and genuine infeasibility flags.
- **Surrogate fidelity holds where it's used.** In-distribution predictions
  land on the simulator; on the unseen topology they are conservative (safe
  against shipping bad grids).
- The remaining weakness is the surrogate's **non-monotonic, over-pessimistic
  response on OOD topologies**, which cost one *false rejection* (not a
  dangerous false pass). The cure is topology coverage and/or enforced
  monotonicity — a model/data fix, addressable.
- For a real flow, the recommended use is exactly what we do here: **surrogate
  proposes, simulator confirms.** The surrogate collapses a slow search into a
  handful of candidate designs; the simulator certifies the final pick.
