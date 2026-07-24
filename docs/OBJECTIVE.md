# Objective — amortized power-grid repair, verified by physics

## The pain point

On a small, fully-specified grid, exact physics wins every forward
comparison — stages 0/1 on the IBM benchmarks proved this conclusively
(docs/STAGE1_PLAN.md, docs/STAGE1_RESULTS.md). That is not the pain point.
The pain point in real PDN design is the **price per question**:

- At industrial scale (10⁸–10⁹ nodes, chip+package+board) a single dynamic
  IR signoff run takes hours; designers get a handful of full analyses per
  week, late in the flow, when changes are most expensive.
- Design is a **discrete search under shared budgets** — decap sites, strap
  widths, vias compete with logic area, leakage, and routing tracks. Fixes
  redistribute current (~28 % of single-edge widenings make worst droop
  *worse* — measured, docs/SENSITIVITY_GATE.md), so the flow loops:
  fix → re-place/route → re-extract → re-simulate, a day per lap.
- The **workload is unknown**: droop depends on load waveform timing (the
  dominant conditioning variable — measured, stage 1), and real signoff can
  only enumerate a few scenarios.
- **Early in the flow there is nothing to solve** — no netlist yet, only
  partial/statistical descriptions physics cannot consume.

## The objective

Build a GNN surrogate + repair policy whose job is **amortization**, not
beating the solver on one instance: given a grid that violates a droop
budget, propose the discrete/continuous fixes in one shot (milliseconds),
with cheap exact physics (the QS/RC floors — seconds even at millions of
nodes) verifying every proposal.

**Success metric:** reach a *simulator-verified* fix in far fewer simulator
calls than adjoint-guided physics search — measured as (a) constraint
satisfaction rate of proposed fixes, (b) fix cost (decap area / wire
resources) vs the physics-search fix, (c) simulator calls consumed.
Never "beats physics on a fixed grid" — that comparison is settled.

## Division of labor (settled by experiment)

| job                                   | owner                | evidence |
|---------------------------------------|----------------------|----------|
| forward droop on one specified grid   | exact physics        | stage 0/1: physics floors 0.85–0.91 ρ, no model beats them |
| verification of any proposed fix      | exact physics (QS/RC solvers, seconds) | built, validated |
| which fix, where, under budgets       | learned model        | discrete moves have no gradient; search needs 10³⁺ queries |
| new workload / partial design queries | learned model        | physics cannot enumerate or consume them |
| training gradients for the model      | exact adjoint physics| linear system ⇒ exact Jacobian = one extra solve |

## The plan (gated, in order)

1. **Sobolev training** — the sensitivity gate (docs/SENSITIVITY_GATE.md)
   showed every checkpoint's learned Jacobian fails where it matters
   (derivative transfer across anchors; big-die reach). The synthetic
   system is linear, so the exact Jacobian ∂droop/∂(every R, every C) costs
   one adjoint/backprop solve per design. Generate Jacobian labels, add a
   gradient-matching term to the training loss.
2. **Global-reach term** — at (7,13) the 7-hop receptive field makes distant
   sensitivities architecturally zero; port the superposition-attention
   global term to the synthetic model.
3. **Re-run the sensitivity gate** as the acceptance test
   (bar: sign ≥ 0.95, site-rank ρ ≥ 0.8, per anchor including held-out).
4. **Repair harness** — surrogate proposes, physics verifies; score against
   adjoint-guided search on the success metric above.
5. **Timing as a data axis** — vary load delay/duty/phase in the next
   dataset build so learned sensitivities condition on workload.

Constraints that stand: GPU training is launched by the user (never from
here); per-anchor reporting always (v7 lesson); basis-invariant use of any
sketch geometry.
