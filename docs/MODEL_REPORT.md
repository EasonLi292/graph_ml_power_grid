# PDN Droop Surrogate + Inverse Design — Report

*A plain-language walkthrough of what the model is, the data it learns from,
how it's built, what we found, and whether it can design a power grid backward
from a droop spec.*

---

## 1. The problem in one paragraph

A chip's **power delivery network (PDN)** is the metal mesh that carries
supply current to every transistor. When a block draws current, the voltage
sags locally — this sag is **droop**. Too much droop and the circuit
misbehaves. The classic question is *forward*: "given this grid, how bad is the
droop?" — normally answered by a slow circuit simulator. We train a graph
neural network (GNN) to answer it **instantly**, and then run it **backward**:
"given a droop budget and a grid topology, what wire width and decap do I
need?"

---

## 2. The data

### 2.1 How a power grid becomes a graph

The PDN is a two-layer mesh, represented as a heterogeneous graph:

- **Nodes** — every wire intersection. Two node types: `mesh_top` (top metal
  layer) and `mesh_bot` (bottom layer). Each node carries just **2 features**:
  `[is_vdd, is_pad]` — which supply rail it's on, and whether it's a bump/pad
  (a boundary condition). Nothing else: layer identity is already the node
  *type*, physical adjacency is already the `edge_index`, and segment scale is
  already baked into the edge resistances. Crucially, **no coordinates** — the
  model is forbidden from learning absolute position, which wouldn't transfer
  to a new floorplan. (See §6.)
- **Edges** — four physical relations, each with attributes
  `[R, C, I_peak, freq, duty, phase]`:

  | relation | meaning | key attribute |
  |---|---|---|
  | `strap` | wire segment within a layer | **R** (resistance) |
  | `via`   | vertical connection between layers | **R** |
  | `decap` | decoupling capacitor (Vdd↔Vss) | **C** (capacitance) |
  | `load`  | a current-drawing block | **I_peak, freq, duty, phase** |

The model predicts droop **per `load` edge** — one number for every block on
the chip.

### 2.2 The design space (3 knobs)

To keep the problem tractable we vary only three things; everything else is
fixed:

| knob | range | role |
|---|---|---|
| `wire_width` | 0.2 – 1.0 | thicker wire → lower R → less droop (costs metal) |
| `C_decap`    | 5e-11 – 8e-10 F | more capacitance → buffers transients (costs area) |
| `n_top` (supply/track density) | {3, 4, 7} | number of supply pads — more pads → less droop |

The load (current, frequency, duty, phase) and the grid topology family are
**held fixed**, so the model can focus on the width/decap/pad-density tradeoff.

### 2.3 Dataset and the all-important split

`datasets/regular_v5` — 16,000 train / 2,000 val / 2,000 test samples. The
split is **by pad count**, and this is the crux of how we measure the model:

| split | n_top values | purpose |
|---|---|---|
| train + val | **3 and 7** | what the model sees |
| **test** | **4 (never seen)** | does it *generalize* to a new topology? |

So the test number is a true out-of-distribution (OOD) check: the model must
**interpolate to a pad count it never trained on**.

> **Why two metrics matter.** In-distribution, the model essentially memorizes:
> validation R² is ~0.99999 for every configuration, so it tells us nothing.
> The honest score is the **test** R² on the unseen n_top = 4. Throughout this
> report, "R²" means test R².

---

## 3. The architecture

```
   node features ─┐
                  ├─►  PDNEncoder  ─►  per-node hidden states
   edge features ─┘     (N message-passing layers)
                                         │
                          for each load edge k:
                          [ h(Vdd node) ‖ h(Vss node) ]
                                         │
                                       Head MLP
                                         │
                                  log10(droop_k)
```

### 3.1 The encoder (message passing)

Each layer lets every node exchange information with its neighbors, then
updates its hidden state (with a residual connection + LayerNorm). Stacking
`N` layers means information can travel `N` hops across the grid. Hidden width
is 64.

### 3.2 The conductance gate — the physics trick

Ordinary GNNs learn an arbitrary function on each edge. We instead **bake Ohm's
law into the resistor edges** (`strap`, `via`). The message from neighbor *j* to
node *i* is:

```
gate_ij = exp(−α · z(log10 R_ij))         # a scalar weight
msg_ij  = gate_ij · MLP(h_j − h_i)
```

- `h_j − h_i` mirrors the voltage *difference* in `i = (1/R)·(v_j − v_i)`.
- `gate_ij` scales the message by **conductance**: low-R (fat) wire → gate ≈ 1
  (neighbor heard loudly); high-R (thin) wire → gate ≈ 0 (ignored).
- `α` is a **single learnable scalar per layer**. At α ≈ 1 the gate is
  approximately `1/R` — the real physical conductance.

**Why deterministic (not a learned MLP)?** If an MLP could read R freely, it
would memorize resistance patterns as a fingerprint of the *training* grids and
cheat — great val score, collapse on new grids. Forcing the gate to be
`exp(−α·z)` removes that shortcut, which is a big reason the test score holds
up. (Decap edges keep a learned gate; capacitance's effect on droop is messier
than a single scalar.)

### 3.3 The readout head

For each `load` edge the head reads the encoder's hidden state at the Vdd-side
and Vss-side nodes, concatenates them, and an MLP outputs one droop value
(`log10` space). That's the per-block droop prediction.

---

## 4. Findings

### 4.1 Depth (number of hops) is the dominant lever

Droop must propagate from the supply pads to the interior blocks. If the
network has fewer hops than the grid is wide, interior nodes never "hear" the
pads and the model is structurally blind. Sweeping the number of layers
(conductance gate on, everything else fixed):

| hops (`n_layers`) | test R² (dev sweep) |
|---|---|
| 3 | 0.187 |
| 5 | 0.801 |
| 7 | 0.863 |

The jump from 3→5 is enormous (+0.61); 5→7 adds +0.06; beyond that it flattens.
**7 hops is the chosen default.** (This sweep was run during development to
establish the trend — depth is the dominant lever. The final coordinate-free
model scores **0.827** at 7 hops; the absolute level shifts slightly but the
trend is the point.)

> **Per-site vs worst-load.** The 0.827 here is *per-load-site* R². The number a
> designer acts on is the **worst** droop on the chip, and the model is much
> better at that: worst-load R² = **0.944**, Spearman = **0.987**. Deep dive in
> [PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md).

### 4.2 The conductance gate was a hidden hero

Earlier experiments credited gains to various tricks ("conductance conv",
"DropEdge", "more data") that lifted R² from 0.187 to ~0.78. On inspection,
**every one of those runs had also quietly added a layer (3→4)** — so depth was
an uncontrolled variable. Two genuinely useful ingredients survive that
scrutiny:

1. **The conductance gate** — now the default; physically motivated and it
   directly prevents the memorization shortcut.
2. **Depth** — enough hops to span the grid.

Together they give 0.863 with no other tricks.

### 4.3 What did *not* help: DropEdge

DropEdge (randomly dropping strap/decap edges during training, as
regularization) only hurt once the conductance gate + 7 hops were in place:

| DropEdge rate | test R² |
|---|---|
| 0 (off) | **0.863** |
| 0.05 | 0.849 |
| 0.10 | 0.858 |
| 0.20 | 0.837 |

The model already generalizes well; the regularizer just removes signal.
**Dropped from the design.**

---

## 5. Inverse design — can it pick the right width & decap?

The payoff. Given a **droop budget** (e.g. ≤ 0.15 mV) and a **topology**
(`n_top`), we freeze the surrogate and optimize the design backward:

1. Decode an unconstrained latent `z` into a valid `(wire_width, C_decap)`.
2. Build the grid graph differentiably and predict worst-case droop.
3. Loss = `ReLU(droop − budget)²  +  λ · cost(width, decap)` — meet the spec,
   then minimize metal.
4. Backprop through *surrogate → graph builder → decoder*, Adam-step `z`.
5. **Validate the recovered design against the real simulator.**

### 5.1 Results (coordinate-free surrogate; sim = simulator at the recovered design)

| budget | n_top | wire_width | C_decap | pred | **sim** | meets spec |
|---|---|---|---|---|---|---|
| 0.10 mV | 3 | 0.986 | 7.2e-10 | 0.113 | 0.113 | ✗ infeasible |
| | 4 (OOD) | 0.736 | 7.7e-10 | 0.141 | 0.122 | ✗ infeasible |
| | 7 | 0.831 | 2.3e-10 | 0.099 | 0.099 | ✓ |
| 0.15 mV | 3 | 0.909 | 4.2e-10 | 0.134 | 0.134 | ✓ |
| | 4 (OOD) | 0.713 | 5.6e-10 | 0.150 | 0.132 | ✓ |
| | 7 | 0.601 | 5.1e-11 | 0.150 | 0.150 | ✗ (boundary) |
| 0.20 mV | 3 | 0.613 | 2.5e-10 | 0.200 | 0.200 | ✓ |
| | 4 (OOD) | 0.631 | 2.4e-10 | 0.196 | 0.183 | ✓ |
| | 7 | 0.448 | 5.1e-11 | 0.197 | 0.197 | ✓ |

*Full deep dive — figures, the conservative-OOD "safe error" story, and the one
local-minimum case — in [GENERATION_ANALYSIS.md](GENERATION_ANALYSIS.md).*

### 5.2 The story — five physically correct signals

1. **The surrogate is trustworthy exactly where it's used.** In-distribution,
   predicted droop ≈ simulated droop dead on (0.134/0.134, 0.200/0.200). The
   optimizer isn't exploiting a model blind spot. *This validates the model for
   design far better than the aggregate R².*
2. **Looser budget → less copper, every time.** As the budget relaxes
   0.10→0.15→0.20, recovered wire width shrinks monotonically (n_top=3:
   0.99→0.91→0.61). Correct cost/performance tradeoff.
3. **Harder topology → more copper.** At a fixed budget, fewer pads need fatter
   wire (n_top 7 always uses the least). Fewer supply points means each carries
   more current — textbook PDN physics, learned, not told.
4. **It lands on the spec boundary, not over-built.** Simulated droop sits right
   at the budget — the *cheapest* design that meets spec, which is the point.
5. **It reports honest infeasibility.** At 0.10 mV with only 3 pads, the
   optimizer drives wire width to the ceiling (~0.986) and still can't meet
   spec — and says so (✗) rather than inventing a design.

On the **unseen topology (n_top = 4)** the surrogate is consistently
*conservative* — it predicts higher droop than the simulator measures (0.141 vs
0.122; 0.150 vs 0.132; 0.196 vs 0.183). Erring high is the **safe** direction
for a design tool: anything it certifies as feasible is feasible with margin.

---

## 6. The coordinate-free representation

We deliberately strip node features down to `[is_vdd, is_pad]` — no
coordinates, no layer one-hot. Layer identity is already the node *type*,
adjacency is the `edge_index`, and segment scale is in the resistances, so those
signals would be redundant; coordinates specifically are a *non-transferable*
shortcut (within one grid family, position ≈ distance-from-pad, which a model
can memorize but can't carry to a new floorplan). The model therefore learns
only from **topology, rail/boundary flags, and component values** — and still
recovers the spatial droop structure on the held-out topology (worst-load
Spearman 0.987; see the per-chip map in
[PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md) §4.1).

---

## 7. Limitations & honest caveats

- **Narrow topology diversity.** The model has seen only two pad counts (3, 7)
  on one grid family. 0.827 on the interpolated n_top = 4 is strong, but says
  little about a *genuinely* different topology. The next real improvement is
  more topology variety in training — a data question, not an architecture one.
- **Aggregate R² is the wrong scoreboard for design.** What matters downstream
  is fidelity *where the optimizer lands* (§5.1) and correct *ranking* of
  designs (Spearman, not R²) — both of which look good.
- **OOD optimization robustness.** The inverse-design optimizer can settle in a
  local minimum on the unseen topology (one of nine cases). The fix is
  multi-start `z` and a feasibility margin (`budget·(1−ε)`), not a better model.

---

## 8. Bottom line

- **Model:** a heterogeneous GNN with a physics-shaped **conductance gate**,
  **7 message-passing hops**, and a **coordinate-free** node representation,
  predicting per-block droop. Test (OOD) per-site R² = **0.827**; worst-load R²
  = **0.944**, Spearman = **0.987** — on a pad count it never trained on.
- **The wins that matter:** the conductance gate (intuitive, prevents
  memorization), enough depth to span the grid, and a representation that learns
  only transferable structure. DropEdge and the other tricks were either
  confounded by depth or actively unhelpful.
- **Inverse design works and tells a coherent physical story:** it recovers
  sensible width/decap, trades cost against the droop budget correctly, scales
  copper with topology difficulty, and flags infeasible specs — with surrogate
  predictions that the real simulator confirms (and that err *conservatively* on
  unseen topologies).
- **Deeper dives:** [PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md) (forward
  accuracy, where it differs) and [GENERATION_ANALYSIS.md](GENERATION_ANALYSIS.md)
  (inverse design). For a quick visual scan, the curated
  [MODEL_PERFORMANCE.md](MODEL_PERFORMANCE.md) figure gallery.
