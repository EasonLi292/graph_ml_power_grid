# Inverse Design — Layer-by-Layer Mechanics

*How the design loop actually finds the optimal `wire_width` and `C_decap` for a
given droop budget — and, specifically, what "optimizing through the network"
means when **the network's weights never change**. This is the detailed
companion to [GENERATION_ANALYSIS.md](GENERATION_ANALYSIS.md); the code is
[scripts/design_grad.py](../scripts/design_grad.py),
[eason/encoder.py](../eason/encoder.py), and [eason/convs.py](../eason/convs.py).*

---

## 0. The one idea (read this first)

There are three different things one could call "the weights," and only one of
them moves during design. Keeping them straight removes all the confusion:

| thing | example | during **training** | during **design** |
|---|---|---|---|
| **network weights** | the MLP matrices in every conv, the head | **trained** (updated) | **frozen** (constant) |
| **edge attributes** | `R` on a strap, `C` on a decap | come from the data | **computed from the 2 knobs** |
| **design knobs** | `wire_width`, `C_decap` | come from the data | **the variables we optimize** |

**We never optimize edge weights directly, and we never touch the network
weights.** We optimize **two scalars** (`wire_width`, `C_decap`). Those two
scalars *determine* every resistor and capacitor edge attribute through fixed
physics formulas, and the frozen network turns those edge attributes into a
droop prediction. Because every step in that chain is differentiable,
`autograd` can compute

```
∂(predicted droop) / ∂(wire_width)      and      ∂(predicted droop) / ∂(C_decap)
```

and we do gradient descent on the two knobs. It is the **exact same chain rule
as backprop in training** — only the *leaf* of the graph is different: in
training the leaves are the weight matrices; in design the leaves are the two
design knobs, and the weight matrices are constants.

> **The closest analogy:** adversarial examples / DeepDream. There you freeze a
> trained image classifier and run gradient descent **on the input pixels** to
> change the output. Here we freeze a trained droop predictor and run gradient
> descent **on the design knobs** to hit a droop target. Same machinery,
> different leaf tensor.

---

## 1. The variables and the reparameterization

The optimizer's actual variable is an unconstrained 2-vector `z ∈ ℝ²`. A sigmoid
maps it into the valid physical ranges ([`to_physical`](../scripts/design_grad.py)):

```
wire_width = sigmoid(z₀) · (1.0 − 0.2) + 0.2                              ∈ [0.2, 1.0]
C_decap    = 10^( sigmoid(z₁) · (log₁₀ 8e-10 − log₁₀ 5e-11) + log₁₀ 5e-11 ) ∈ [5e-11, 8e-10]
```

Why this indirection? So that **any** real value of `z` decodes to a *valid*
design — no clipping, no constrained optimizer. `z` is a leaf tensor with
`requires_grad=True`; everything downstream is differentiable in `z`.

`C_decap` is mapped in log-space because it spans more than a decade, so a unit
step in `z₁` is a constant *ratio* change in capacitance.

---

## 2. From the 2 knobs to the edge attributes (this is the "edge weights" part)

The graph **topology is fixed** — the `edge_index` lists (who connects to whom)
never change during design. What changes is the **edge attribute tensor** on the
resistor and capacitor edges, and those are *differentiable functions of the two
knobs* ([`build_diff_data`](../scripts/design_grad.py)):

```
strap (top):  R_top = Rsheet_top · pitch_top / wire_width      ← depends on wire_width
strap (bot):  R_bot = Rsheet_bot · pitch_bot / wire_width      ← depends on wire_width
via:          R_via = const                                     (not a knob)
decap:        C     = C_decap                                   ← is the knob
load:         (I_peak, freq, duty, phase) = const
```

So when `wire_width` is a tensor, `R_top` and `R_bot` are tensors **with a
gradient connection back to `wire_width`** (`∂R/∂wire_width = −Rsheet·pitch /
wire_width²`). Likewise the decap edges carry `C_decap` directly. This is the
sense in which "edge weights are optimized": *they are not free parameters —
they are downstream of the two knobs, and the gradient flows through them.*

### 2.1 "But that's just a scalar multiply — where's the gradient?"

A natural objection: turning `wire_width` into `R` is *just division by a
scalar*, and that value is then *just used* by the message-passing convs — so
where does differentiability come from? Two halves, both required, and both
present here:

1. **Scalar multiply / division is differentiable.** "Differentiable" means
   "has a derivative," not "complicated" or "neural." `R = c / w` has derivative
   `dR/dw = −c/w²`; a plain multiply `y = a·x` has derivative `a`. These are the
   *cleanest* differentiable ops there are — being simple does not make them
   non-differentiable.

2. **It stays a tensor, so the link survives into message passing.** The only
   way this breaks is leaving PyTorch. In `build_diff_data`, `wire_width` is a
   torch tensor (from `sigmoid(z)`) and `R = Rsheet·pitch / wire_width` is a
   *tensor op*, so `R` carries a `grad_fn` pointing back to `wire_width` — it is
   an intermediate node in a live autograd graph, not a detached number. It is
   then **used inside the conv's message** (`gate = exp(−α·ẑ(R))`, `msg = gate ·
   delta_mlp(…)`), again through differentiable ops.

So your reading is correct — the edge attribute *is* just used in message
passing — with one precision: it's the **whole unbroken chain of tensor ops**
that makes it differentiable, and that chain has two halves:

```
   wire_width ──(÷, a tensor op)──► R ──(used in the conv message)──► droop
              └── upstream link ──┘   └──── downstream link ────────┘
```

- The **downstream** half ("used in message passing") carries the gradient
  *backward from droop to R*.
- The **upstream** half ("R built as a tensor op from the knob") carries it
  *from R back to wire_width*.

Both must be intact; here they are.

**What would break it** (for contrast): `R = float(Rsheet * pitch / wire_width)`
or `np.array(...)` or `R.detach()` anywhere on the path turns `R` into a dead
constant — the conv would still run, but `∂droop/∂wire_width` would be zero. A
non-smooth op (rounding `R`, an `argmax` over edges) would also sever it. The
"differentiable graph builder" name is exactly the promise that none of these
happen.

**Fan-out.** One `wire_width` sets the same `R` on *every* strap edge of a
layer, so at backward the gradients from all those edges sum back into the one
scalar — still just the chain rule, with addition:

```
∂droop/∂wire_width = Σ_edges (∂droop/∂R_edge) · (∂R_edge/∂wire_width)
```

---

## 3. The forward pass, layer by layer

The graph now goes through the frozen network. Every weight below is a
**constant** (we called `requires_grad_(False)` on all model parameters); the
only quantities carrying a gradient back to `z` are the ones built from the
knobs.

### 3.1 Node input projection

Node features are just `x_v = [is_vdd, is_pad]` (2-dim, **no coordinates**). A
per-node-type linear map lifts them to the 64-dim hidden width:

```
h_v^(0) = W_node[type(v)] · x_v + b          (W_node frozen)
```

Note `x_v` does **not** depend on the knobs — the design information enters only
through the *edges*. At layer 0 the node states are knob-independent; they
become knob-dependent the moment the first message (which uses edge attributes)
is mixed in.

### 3.2 Edge-attribute normalization

Each raw edge attribute is normalized analytically ([`InputNormalizer`](../eason/normalizer.py)):
log-scale columns (`R`, `C`, …) become `ẑ = (log₁₀ value − μ) / σ`. For the
**resistor** relations the encoder passes just that one scalar column into the
conv; for the others it linearly projects the 7-dim normalized vector to 64-dim
([`encoder._build_edge_attr_dict`](../eason/encoder.py)):

```
strap / via (conductance):  edge_attr = ẑ(log₁₀ R)              shape [E, 1]
decap (admittance):         edge_attr = W_edge · normalize(C…)   shape [E, 64]
load (source):              edge_attr = W_edge · normalize(I…)   shape [E, 64]
```

`R` depends on `wire_width`, so `ẑ(log₁₀ R)` depends on `wire_width`
(differentiably: `log₁₀` and the affine z-score are smooth). `C` depends on
`C_decap`. **These two columns are the only entry points for the knob
gradients.**

### 3.3 One message-passing layer

Each of the 7 layers runs, per edge relation, a [`MessagePassing`](../eason/convs.py)
conv. The three relation types differ in how the **edge attribute** enters the
message — and that is exactly where `wire_width` / `C_decap` influence the
computation:

**Resistor edges (`strap`, `via`) — `AdmittanceConv(kind="conductance")`:**

```
gate_ij = exp(−α · ẑ(log₁₀ R_ij))                  # α is a frozen learned scalar
msg_ij  = gate_ij  ⊙  delta_mlp( h_j − h_i )       # delta_mlp frozen
```

The message is "how strongly are `i` and `j` coupled" (`gate`, set by the
resistance) times "what differential am I propagating" (`delta_mlp` of the
node-state difference, mirroring Ohm's `i = (1/R)(v_j − v_i)`). **`gate_ij`
depends on `R_ij`, hence on `wire_width`.** A thinner wire → larger `R` → larger
`ẑ` → smaller `gate` → weaker coupling → (eventually) more droop. That entire
sentence is differentiable, so `∂msg / ∂wire_width` exists.

**Decap edges — `AdmittanceConv(kind="admittance")`:**

```
msg_ij = gate_mlp(edge_attr) ⊙ delta_mlp( h_j − h_i )     # both MLPs frozen
```

Here `edge_attr` carries the normalized `C_decap`, so `∂msg / ∂C_decap` flows
through the (frozen) `gate_mlp`.

**Load edges — `AdmittanceConv(kind="source")`:** generic
`msg = msg_mlp([h_i ‖ h_j ‖ edge_attr])`; the load attributes are constants, so
no knob gradient enters here (the load is the fixed excitation).

**Aggregate + update + residual** (shared, all weights frozen):

```
agg_i = Σ_j msg_ij                       # sum over neighbors of i
u_i   = upd_mlp( [ h_i ‖ agg_i ] )
h_i^(ℓ) = LayerNorm( ReLU(u_i) + h_i^(ℓ-1) )   # residual connection
```

After layer ℓ, every node state `h_i^(ℓ)` is a differentiable function of the
two knobs (through all the `gate`s it has absorbed). Stacking 7 layers lets a
node "feel" the resistance/capacitance of edges up to 7 hops away — which is why
droop, a long-range IR phenomenon, needs depth.

### 3.4 The droop head

For each `load` edge `k`, read its two endpoint states (the Vdd-side and
Vss-side `mesh_bot` nodes), concatenate (2 × 64 = 128-dim), and map to a scalar
([`PDNDroopRegressor.forward`](../eason/encoder.py)):

```
droop_log_k = head( [ h_vddside(k) ‖ h_vssside(k) ] )      # head frozen
droop_k     = 10 ^ droop_log_k
worst       = max_k droop_k                                 # the spec-binding number
```

`worst` is now a single scalar that is a differentiable function of
`(wire_width, C_decap)` — i.e. of `z`.

---

## 4. The loss

Two terms ([`design_loss`](../scripts/design_grad.py)):

```
rel_over       = relu( worst / budget − 1 )      # 0 if feasible, >0 if over budget
spec_violation = rel_over²                        # quadratic hinge
cost           = ww_frac + cd_frac               # normalized spend, each ∈ [0,1]
loss           = spec_violation + λ · cost        # λ ≈ 1e-3
```

- The **hinge** is one-sided: it punishes *exceeding* the budget and is exactly
  zero once feasible (no reward for over-design).
- The **cost** is the normalized wire + decap spend; its gradient is positive in
  both knobs (more copper / cap costs more).

---

## 5. Backprop — the chain rule, end to end

This is the part the table in §0 is really about. One `loss.backward()` call
walks the graph **backwards**, multiplying local derivatives (chain rule), and
deposits a gradient on every leaf with `requires_grad=True`. The *only* such
leaf is `z` (the network weights have `requires_grad=False`, so they're treated
as constants and receive nothing).

The path for `wire_width`, read right-to-left:

```
∂loss/∂z₀  =  ∂loss/∂worst
            · ∂worst/∂droop_k                (max picks the worst load)
            · ∂droop_k/∂h(endpoints)         (through the frozen head)
            · ∂h^(7)/∂h^(6) ··· ∂h^(1)/∂h^(0) (through 7 frozen conv layers —
                                               each layer's gate uses R)
            · ∂gate/∂ẑ(R) · ∂ẑ/∂R            (the normalizer)
            · ∂R/∂wire_width                 (= −Rsheet·pitch / wire_width²)
            · ∂wire_width/∂z₀                (the sigmoid)
```

Every factor is a number `autograd` already knows how to produce, because each
op (`exp`, `*`, `Linear`, `LayerNorm`, `max`, `10^x`, `sigmoid`, the division
for `R`) registered its derivative on the forward pass. The frozen weights still
participate in these factors (e.g. `∂h^(ℓ)/∂h^(ℓ-1)` involves the conv matrices)
— they just don't *receive* a gradient themselves. `C_decap`'s path is identical
but enters through the decap `gate_mlp` instead of the conductance gate.

**Why "frozen network" doesn't mean "no gradient":** freezing only stops the
weights from being *updated*; they are still part of the differentiable
function. `droop = f_θ(wire_width, C_decap)` with θ fixed is still a perfectly
good differentiable function of its inputs, and we want its input-gradient.

---

## 6. The update step

Plain Adam on the two knobs ([`optimize`](../scripts/design_grad.py)), ~200
steps:

```
for step in range(n_steps):
    loss, info = design_loss(model, z, n_top, budget, λ)
    loss.backward()          # fills z.grad via §5
    opt.step()               # z ← z − Adam(∂loss/∂z)
    opt.zero_grad()
```

No simulator runs inside this loop — that's the whole point. Each step is one
cheap forward + backward through the surrogate.

**What the two gradient terms do, intuitively:**

- *Infeasible* (`worst > budget`): `spec_violation` dominates. Its gradient says
  "increase whatever reduces droop" — and since `∂droop/∂wire_width < 0` and
  `∂droop/∂C_decap < 0`, Adam **grows** the knobs.
- *Feasible* (`worst ≤ budget`): `spec_violation` and its gradient are zero; only
  `λ·cost` is left, whose gradient is positive, so Adam **shrinks** the knobs —
  until droop rises back to the budget and the hinge switches on again.

The fixed point of those opposing pushes is the **cheapest design that just
meets spec** — it settles on the spec boundary. Because `cost = ww_frac +
cd_frac` treats wire and decap as substitutes, the gradients automatically spend
on whichever knob buys more droop-reduction per unit cost, so the wire/decap mix
is whatever is locally cheapest — not a fixed ratio.

---

## 7. Decode and validate

After the loop, `to_physical(z)` gives the final `(wire_width, C_decap)`. That
design is handed to the **real transient simulator** (`run_one`) and the spec is
re-checked. This matters because §1–§6 optimized the *surrogate's* prediction,
and on an unseen topology the surrogate's response surface can be wrong (the
wire-axis non-monotonicity in [GENERATION_ANALYSIS.md](GENERATION_ANALYSIS.md)
§4). So the design flow is **surrogate proposes, simulator confirms.**

---

## 8. One-paragraph summary

We hold two design scalars as the only trainable leaves, map them through a
sigmoid into valid ranges, and build the PDN graph so that every resistor/decap
edge attribute is a differentiable function of those scalars. The frozen droop
network turns that graph into a worst-load droop prediction. `autograd`
backpropagates a "hinge-spec + cost" loss through the network's fixed weights all
the way to the two scalars, and Adam steps them — pushing the design to the
cheapest point that just meets the droop budget. Nothing about the network
changes; we are using its input-gradient, exactly like an adversarial example,
to search the design space.
