# Inverse Design — Layer-by-Layer Mechanics

How the design loop optimizes `wire_width` and `C_decap` for a droop budget, and
what "optimizing through the network" means when the network's weights never
change. Code: [scripts/design_grad.py](../scripts/design_grad.py),
[eason/encoder.py](../eason/encoder.py), [eason/convs.py](../eason/convs.py).

## 0. The core idea

Three things could be called "the weights"; only one moves during design:

| | during training | during design |
|---|---|---|
| network weights (conv MLPs, head) | trained | **frozen (constant)** |
| edge attributes (`R`, `C`) | from data | **computed from the 2 knobs** |
| design knobs (`wire_width`, `C_decap`) | from data | **the variables optimized** |

We optimize **two scalars**. They determine every resistor/capacitor edge
attribute through fixed physics formulas, and the frozen network maps those to a
droop prediction. Every step is differentiable, so autograd gives
`∂droop/∂wire_width` and `∂droop/∂C_decap`, and we gradient-descend on the knobs.
This is the same chain rule as training — only the *leaf* differs: in training
the leaves are weight matrices; here they are the two knobs, weights are
constants. (Same pattern as an adversarial example: freeze a trained net,
gradient-descend on its input.)

## 1. Variables

The optimizer's variable is unconstrained `z ∈ ℝ²`, mapped into range by a
sigmoid ([`to_physical`](../scripts/design_grad.py)):

```
wire_width = sigmoid(z₀)·(1.0 − 0.2) + 0.2                                  ∈ [0.2, 1.0]
C_decap    = 10^(sigmoid(z₁)·(log₁₀ 8e-10 − log₁₀ 5e-11) + log₁₀ 5e-11)     ∈ [5e-11, 8e-10]
```

Any `z` decodes to a valid design (no clipping). `C_decap` is log-mapped because
it spans a decade.

## 2. Knobs → edge attributes

Topology (`edge_index`) is fixed. The knobs set the resistor/capacitor edge
attributes ([`build_diff_data`](../scripts/design_grad.py)):

```
strap: R = Rsheet·pitch / wire_width     ← depends on wire_width
via:   R = const
decap: C = C_decap                       ← is the knob
load:  (I_peak, freq, duty, phase) = const
```

`R = c/wire_width` is a tensor op, so `R` carries a gradient back to the knob
(`∂R/∂wire_width = −c/wire_width²`).

### 2.1 "It's just a scalar multiply — where's the gradient?"

Two facts, both needed:

1. **Scalar multiply/division is differentiable** — `R = c/w` has `dR/dw =
   −c/w²`; simple is not non-differentiable.
2. **It stays a tensor**, so the link survives into message passing. `wire_width`
   is a torch tensor (from `sigmoid(z)`) and `R = c/wire_width` is a tensor op, so
   `R` is a live autograd node, not a detached number. It is then used in the
   conv message (`gate = exp(−α·ẑ(R))`, `msg = gate · delta_mlp(…)`).

So the gradient path has two halves: **upstream** (knob → `R`, the division)
carries it from `R` to the knob; **downstream** (`R` used in the message) carries
it from droop to `R`. Both are intact.

What breaks it: leaving PyTorch (`float(...)`, numpy, `.detach()`) or a non-smooth
op (rounding, `argmax`). A single `wire_width` sets `R` on every strap edge, so
gradients from all of them sum back: `∂droop/∂wire_width = Σ_edges
(∂droop/∂R_edge)(∂R_edge/∂wire_width)`.

## 3. Forward pass

All weights below are frozen; only quantities built from the knobs carry a
gradient.

**Node projection.** `h_v^(0) = W_node[type(v)]·[is_vdd, is_pad]`. Knob-
independent at layer 0 — design info enters only through edges.

**Edge normalization.** Each attribute is z-scored
([`InputNormalizer`](../eason/normalizer.py)). Resistor relations pass the scalar
`ẑ(log₁₀ R)`; others project the 7-dim vector to 64-dim. `R` depends on
`wire_width` and `C` on `C_decap` — these two columns are the only entry points
for the knob gradients.

**One message-passing layer** ([conv](../eason/convs.py)), per relation:

```
conductance (strap, via):  gate = exp(−α·ẑ(R));  msg = gate · delta_mlp(h_j − h_i)
admittance  (decap):       msg = gate_mlp(edge_attr) · delta_mlp(h_j − h_i)
source      (load):        msg = msg_mlp([h_i ‖ h_j ‖ edge_attr])
agg_i = Σ_j msg_ij;  h_i^(ℓ) = LayerNorm(ReLU(upd_mlp([h_i ‖ agg_i])) + h_i^(ℓ-1))
```

`gate` depends on `R` (hence `wire_width`): thinner wire → larger R → smaller
gate → weaker coupling → more droop, all differentiable. Decap's `C_decap` enters
through `gate_mlp`. Load attributes are constant. After ℓ layers each node state
is a differentiable function of the knobs through the gates it absorbed; 7 layers
let a node feel edges 7 hops away (why droop needs depth).

**Head.** Per `load` edge: `droop_k = 10^head([h_vdd ‖ h_vss])`;
`worst = max_k droop_k` — a single scalar, differentiable in the knobs.

## 4. Loss

```
spec_violation = relu(worst/budget − 1)²     # one-sided hinge, 0 once feasible
cost           = ww_frac + cd_frac           # normalized spend, gradient > 0 in both knobs
loss           = spec_violation + λ·cost     # λ ≈ 1e-3
```

## 5. Backprop

`loss.backward()` walks the graph backward via the chain rule and deposits a
gradient only on `z` (weights have `requires_grad=False`). The `wire_width` path,
right-to-left:

```
∂loss/∂z₀ = ∂loss/∂worst · ∂worst/∂droop_k · ∂droop_k/∂h(endpoints)
          · [∂h^(7)/∂h^(6) ··· ∂h^(1)/∂h^(0)]   (7 frozen layers; each gate uses R)
          · ∂gate/∂ẑ · ∂ẑ/∂R · ∂R/∂wire_width · ∂wire_width/∂z₀
```

Frozen weights still appear in these factors (e.g. the conv matrices in
`∂h^(ℓ)/∂h^(ℓ-1)`); they just don't *receive* a gradient. Freezing stops weight
*updates*, not differentiation — `droop = f_θ(knobs)` with θ fixed is still a
differentiable function whose input-gradient we want. `C_decap`'s path is the same
but enters via the decap `gate_mlp`.

## 6. Update

Adam on the two knobs (~200 steps), no simulator in the loop:

```
loss, info = design_loss(model, z, n_top, budget, λ)
loss.backward(); opt.step(); opt.zero_grad()
```

- Infeasible: `spec_violation` dominates; since `∂droop/∂knob < 0`, Adam grows
  the knobs.
- Feasible: hinge gradient is zero; only `cost` remains, so Adam shrinks the
  knobs until droop rises back to the budget.

The fixed point is the cheapest design on the spec boundary. Because `cost`
treats wire and decap as substitutes, the gradients spend on whichever buys more
droop-reduction per unit cost.

## 7. Validate

`to_physical(z)` gives the final `(wire_width, C_decap)`, which the real simulator
re-checks — necessary because the loop optimized the surrogate, whose OOD response
can be wrong (the wire-axis non-monotonicity in
[GENERATION_ANALYSIS.md](GENERATION_ANALYSIS.md) §4).
