# PDN Droop Surrogate + Inverse Design

A GNN that predicts power-grid voltage droop from the grid graph (replacing a
slow circuit solve), and the inverse: given a droop budget, recover the wire
width and decap that meet it.

Deep dives: [PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md) (forward accuracy),
[GENERATION_ANALYSIS.md](GENERATION_ANALYSIS.md) (inverse design),
[INVERSE_DESIGN_MECHANICS.md](INVERSE_DESIGN_MECHANICS.md) (how the backprop
loop works), [MODEL_SIZE.md](MODEL_SIZE.md) (parameter count).

## 1. The data

The PDN is a two-layer mesh encoded as a heterogeneous graph.

- **Nodes** — wire intersections, two types (`mesh_top`, `mesh_bot`). Each
  carries 2 features `[is_vdd, is_pad]` (which rail; whether it's a supply
  bump). No coordinates and no layer one-hot: layer identity is the node type,
  adjacency is the `edge_index`, and segment scale is in the edge resistances.
  Coordinates would be a non-transferable shortcut (§6).
- **Edges** — four relations with attributes `[R, C, I_peak, freq, duty, phase]`:

  | relation | meaning | key attr |
  |---|---|---|
  | `strap` | wire segment within a layer | R |
  | `via` | vertical layer-to-layer link | R |
  | `decap` | decoupling cap (Vdd↔Vss) | C |
  | `load` | current-drawing block | I_peak, freq, duty, phase |

Droop is predicted **per `load` edge** — one value per block.

**Design space (3 knobs):** `wire_width` (0.2–1.0), `C_decap` (5e-11–8e-10 F),
`n_top` (supply-pad count, {3,4,7}). Load and grid family are fixed.

**Dataset / split** (`datasets/regular_v5`, 16k/2k/2k): split by pad count —
train+val on n_top {3,7}, **test on the never-seen n_top 4** (OOD). Validation R²
is ~0.99999 for every config (in-distribution memorization), so it is
uninformative; the honest metric is **test R²**, used throughout.

## 2. Architecture

```
node features ─┐
               ├─►  PDNEncoder (7 message-passing layers) ─► per-node states
edge features ─┘                                              │
                              for each load edge k: [ h(Vdd) ‖ h(Vss) ]
                                                             │  Head MLP
                                                       log10(droop_k)
```

**Encoder:** 7 message-passing layers, residual + LayerNorm, hidden width 64.
Depth = hops information travels across the grid. ~1.0M params, 98.6% in the
conv stack ([MODEL_SIZE.md](MODEL_SIZE.md)).

**Conductance gate (physics bias on resistor edges):**

```
gate_ij = exp(−α · z(log10 R_ij))      msg_ij = gate_ij · MLP(h_j − h_i)
```

`h_j − h_i` mirrors Ohm's `i = (1/R)(v_j − v_i)`; the gate scales the message by
conductance (fat wire → gate ≈ 1, thin wire → gate ≈ 0). `α` is one learnable
scalar per layer. Keeping the gate a deterministic `exp(−α·z)` rather than a free
MLP blocks the model from memorizing R-patterns as a per-grid fingerprint — a
shortcut that scores well in-distribution and collapses OOD. Decap edges keep a
learned gate (C's effect on droop is not a single scalar).

**Head:** per `load` edge, concatenate the Vdd-side and Vss-side `mesh_bot`
states → MLP → `log10(droop)`.

## 3. Findings

**Depth is the dominant lever.** Droop must propagate from pads to interior
blocks; too few hops and interior nodes never hear the pads. Layer sweep
(development, conductance gate on):

| hops | test R² |
|---|---|
| 3 | 0.187 |
| 5 | 0.801 |
| 7 | 0.863 |

7 hops is the default. The final coordinate-free model scores **0.827** per-site
at 7 hops; the trend is what matters.

The number a designer acts on is the **worst** droop on the chip, where the model
is much stronger: worst-load R² = **0.944**, Spearman = **0.987** (vs per-site
0.827 / 0.918). See [PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md).

**Two ingredients carry the result:** the conductance gate (prevents the
memorization shortcut) and depth. Earlier "wins" credited to other tricks were
confounded — every one had also quietly added a layer.

**DropEdge does not help** once the gate + 7 hops are in place: p=0 → 0.863,
0.05 → 0.849, 0.10 → 0.858, 0.20 → 0.837. Dropped.

## 4. Inverse design

Freeze the surrogate; optimize the two knobs by gradient descent to the cheapest
design meeting a droop budget (mechanics: [INVERSE_DESIGN_MECHANICS.md](INVERSE_DESIGN_MECHANICS.md)):

1. Knobs `(wire_width, C_decap)`, sigmoid-reparameterized to stay in range.
2. Build the graph differentiably; predict worst-load droop.
3. Loss = `ReLU(droop − budget)² + λ·cost(width, decap)` — meet spec, then
   minimize metal.
4. Backprop to the two knobs; Adam step.
5. Validate the recovered design against the real simulator.

Results (`sim` = simulator at the recovered design):

| budget | n_top | wire_width | C_decap | pred | sim | verdict |
|---|---|---|---|---|---|---|
| 0.10 mV | 3 | 0.986 | 7.2e-10 | 0.113 | 0.113 | ✗ infeasible |
| | 4 (OOD) | 0.736 | 7.7e-10 | 0.141 | 0.122 | ✗ false reject (§ gen-analysis 4) |
| | 7 | 0.831 | 2.3e-10 | 0.099 | 0.099 | ✓ |
| 0.15 mV | 3 | 0.909 | 4.2e-10 | 0.134 | 0.134 | ✓ |
| | 4 (OOD) | 0.713 | 5.6e-10 | 0.150 | 0.132 | ✓ |
| | 7 | 0.601 | 5.1e-11 | 0.150 | 0.150 | ✗ boundary |
| 0.20 mV | 3 | 0.613 | 2.5e-10 | 0.200 | 0.200 | ✓ |
| | 4 (OOD) | 0.631 | 2.4e-10 | 0.196 | 0.183 | ✓ |
| | 7 | 0.448 | 5.1e-11 | 0.197 | 0.197 | ✓ |

The loop behaves correctly: in-distribution it lands the requested droop to a few
µV (pred ≈ sim ≈ target); looser budget → less metal; fewer pads → more metal; it
sits on the spec boundary rather than over-building; and it flags genuine
infeasibility. On the unseen n_top 4 the surrogate is conservative (predicts
higher droop than the simulator) — safe, except in one case where that pessimism
plus a non-monotonic wire response caused a false rejection of an achievable
spec. Full analysis in [GENERATION_ANALYSIS.md](GENERATION_ANALYSIS.md).

## 5. Coordinate-free representation

Node features are `[is_vdd, is_pad]` only. Within one grid family, absolute
position correlates with distance-from-pad and would help in-family, but it does
not transfer to a new floorplan. The model therefore learns from topology,
rail/boundary flags, and component values, and still recovers the spatial droop
pattern on the held-out topology (worst-load Spearman 0.987;
[PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md) §4.1).

## 6. Limitations

- **Topology diversity.** Only two pad counts seen on one grid family. 0.827 on
  the interpolated n_top 4 is strong but says little about a genuinely different
  topology — the next improvement is more topology variety (data, not
  architecture).
- **R² is the wrong scoreboard for design.** What matters is fidelity where the
  optimizer lands and correct ranking (Spearman) — both good.
- **OOD optimization robustness.** The surrogate's response is non-monotonic in
  wire width on the unseen topology, which caused one false rejection. Fix with
  more topologies and/or enforced monotonicity; for now, surrogate proposes and
  the simulator confirms.
