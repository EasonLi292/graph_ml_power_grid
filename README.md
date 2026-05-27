# graph_ml_power_grid

Graph machine learning for analyzing on-chip power delivery networks (PDNs). The model ingests an IC's power grid as a graph and predicts electrical behavior — primarily static IR drop and dynamic voltage drop — without running a full SPICE simulation.

PDNs are the **intermediate problem** here. The longer-term goal is a generative model over transistor-level analog circuits; PDN regression exercises the same encoder / VAE / generation scaffolding on a much smaller, regular design space first. See [docs/GOAL.md](docs/GOAL.md) for the full scope and stop-conditions.

## Motivation

As process nodes shrink and current densities climb, IR drop on the on-die power grid has become a first-order signoff problem. Commercial signoff tools (Voltus, RedHawk-SC) solve very large sparse linear systems for every PDN snapshot, which is slow inside a place-and-route loop. A learned surrogate that operates directly on the PDN graph can give designers a fast inner-loop signal for grid health, decap placement, and current-source planning.

## Architecture

This project reuses the encoder/decoder VAE skeleton from [Z-GED](https://github.com/EasonLi292/Z-GED), retargeted from RLC filter synthesis to power-grid analysis.

- **Encoder — impedance-aware GNN.** Three message-passing layers operating over the PDN graph. Nodes represent grid taps, vias, and instance pins; edges carry per-segment resistance (and inductance/capacitance for dynamic analysis). Hierarchical latent branches separate topology, per-segment electrical values, and load/source conditions, mirroring the Z-GED `[z_topology | z_values | z_pz]` split — here repurposed as `[z_topology | z_RLC | z_loads]`.
- **Decoder (planned) — autoregressive transformer.** A GPT-style decoder that, depending on the head, either (a) reconstructs the PDN as an Eulerian walk over the grid for self-supervised pretraining, or (b) emits per-node voltage / IR-drop predictions conditioned on the latent code and the load configuration. Not yet implemented — only the encoder + regression head currently exist.
- **Training as a VAE (planned).** The model is to be pretrained as a variational autoencoder on PDN graphs so that the latent space captures grid topology and impedance structure, with a supervised head then fine-tuned against the MNA-solver ground truth. Currently only the supervised regression head is wired up.

## Graph representation

A PDN is encoded as a heterogeneous graph with three node types and five logical bidirectional edge relations. **All electrical elements live on edges** — including the load, which is a two-terminal current source between a `mesh_bot` node and `gnd`. Voltages are predicted per `mesh_bot` node.

**Node types** — uniform 6-dim feature `[one_hot_type(3), payload(3)]`. The type one-hot is part of the feature itself (not just an implicit signal carried by per-type weights), so message passing can't forget what kind of node it's looking at.

- `mesh_top` — junction on the coarse top metal layer. Payload `(x, y, is_pad)`; `is_pad=1` carries a Dirichlet condition `V = Vdd` in the solver.
- `mesh_bot` — junction on the fine bottom metal layer. Payload `(x, y, 0)`.
- `gnd` — single ideal-zero reference node. Payload `(0, 0, 0)`.

**Edge relations**, all bidirectional, all carrying the same 6-dim attribute `[R, C, I_peak, freq, duty, phase]` (zero in columns not relevant to that element type):

- `mesh_top ↔ mesh_top` strap — `R` = derived top-segment resistance.
- `mesh_bot ↔ mesh_bot` strap — `R` = derived bot-segment resistance.
- `mesh_top ↔ mesh_bot` via — `R` = `R_via`.
- `mesh_bot ↔ gnd` decap — `C` = `C_decap`.
- `mesh_bot ↔ gnd` load — `(I_peak, freq, duty, phase)` columns; per-edge values. **The load is a current source, not a resistor** — its current is dictated by switching activity, not by Ohm's law, so an `R_load = V/I` model would create a spurious voltage-current feedback under droop.

Same-type bidirectionality (e.g. `mesh_top ↔ mesh_top`) is one PyG relation with both directions packed into `edge_index`. Cross-type bidirectionality (e.g. via, decap, load) is expressed as two PyG relations sharing the same relation name. In the normalizer, the raw 6-dim edge attribute is mapped to 7-dim by replacing `phase` with `(sin 2πφ, cos 2πφ)` for circular continuity on load edges.

**Canonical regular instance** ([tools/grid_construction.py](tools/grid_construction.py), `build_regular_pdn`): `n_top × n_top` M_top stacked on 7×7 M_bot, vias at every M_top node aligned to the corresponding bot positions. Four supply-pad patterns are supported by the builder (`corner` / `checker` / `edge_strip` / `distributed`), but **this dataset pins `pad_pattern="corner"`** so the four pad-via bot positions are independent of `n_top` and the surviving load set stays at a constant 12 cells across every sample. Loads sit on the even M_bot sub-grid *minus* nodes directly under a Vdd-pad via; 9 decap sites sit on the odd sub-grid. Per-segment `R_top`, `R_bot` are derived from sheet resistance × pitch / wire width: `R_seg = Rsheet × (pitch / wire_width)`. Every load draws the same `(I_peak, freq, duty, phase)` waveform — see "Dataset" below for the design rationale.

Ground truth: backward-Euler MNA transient simulation in [tools/transient_solver.py](tools/transient_solver.py). Smoke test: `python scripts/smoke_test.py` builds the canonical instance, runs a 5 ns / 10 ps transient, and prints peak droop per M_bot node.

## Dataset

**Reduced 3-knob design space** (see [tools/sampler.py](tools/sampler.py) for the full rationale). The prior 9-continuous-knob LHS made it impossible to evaluate the eventual generative model cleanly; this version cuts to three knobs, pins everything else to physically realistic constants, and produces samples whose load/decap *placement* is identical across the entire dataset — only component values change.

### Varying knobs

| param        | range / set                | scale            | meaning |
|--------------|----------------------------|------------------|---------|
| `wire_width` | 0.2 – 1.0                  | log-uniform      | strap width (× `pitch_bot`) |
| `C_decap`    | 50 pF – 800 pF             | log-uniform      | per-decap-site capacitance (single MIM macro) |
| `n_top`      | `{3, 4}` train; `{7}` OOD  | uniform discrete | M_top track density (= via-stub density to M_bot) |

`wire_width` and `C_decap` are LHS-sampled jointly; `n_top` is uniform discrete and bucketed for the topology-OOD split (model trains on `{3, 4}`, evaluates extrapolation on `{7}`).

### Fixed constants (every sample)

| topology        | value     | electrical / workload | value         |
|-----------------|-----------|-----------------------|---------------|
| `n_bot`         | 7         | `Rsheet_top`          | ≈ 0.0316 Ω/sq |
| `pad_pattern`   | `corner`  | `Rsheet_bot`          | ≈ 0.158 Ω/sq  |
| `n_loads`       | 12        | `R_via`               | ≈ 0.0632 Ω    |
| `n_decaps`      | 9         | `freq`                | ≈ 0.894 GHz   |
|                 |           | `I_peak`              | ≈ 4.47 mA (broadcast to every load) |
|                 |           | `duty`                | 0.4 |
|                 |           | `phase`               | 0.0 (all loads in-phase — worst case) |

All electrical constants are geometric medians of the prior LHS box, so this dataset's operating point sits in the middle of the prior design space.

**Per-segment R derivation:** the solver stamps `R_top = Rsheet_top × (pitch_top / wire_width)` and `R_bot = Rsheet_bot × (pitch_bot / wire_width)`. `pitch_top` varies with `n_top` (`pitch_top ∈ {3.0, 2.0, 1.0}` for `n_top ∈ {3, 4, 7}` at `n_bot = 7`), so `R_top` actually spans ~15× across the dataset even though `Rsheet_top` is fixed.

**Per-sample sim:** warmup of `max(2 periods, ceil(5·R_bot·C_decap / period))` periods so the initial-condition transient is ≲1% residual regardless of the freq/RC ratio, then 8 measurement periods × 100 steps/period. Each sample yields **peak droop** (Vdd − Vmin over the measurement window) and **static IR drop** (DC solve under the time-averaged load current `I_peak × duty` per load).

### H5 layout (v4)

```
/                                       attrs: version=4, fixed_constants, train_n_top, ood_n_top,
                                               load_attr_row, topology (per n_top), sim_config, ...
├── bulk/{train, val, test}/            LHS over (wire_width, C_decap) × uniform(TRAIN_N_TOP)
│     ├── global_params      [N, 2]    (wire_width, C_decap)
│     ├── n_top              [N]
│     ├── peak_droop_bot     [N, 49]
│     ├── static_droop_bot   [N, 49]
│     ├── worst_node_idx, worst_node_droop
│     └── V_subset/ (train only — first 200 samples' full V(t) on M_bot)
├── ood/n_top_<N>/                      same continuous LHS at a held-out n_top
└── analysis/sweeps/<axis>/n_top_<N>/   1-D sweep along a continuous axis at fixed n_top
```

The H5 stores no per-sample `load_x` or `n_loads` — those are constants and live as root attributes (`load_attr_row`, `n_loads`). The PyG loader pulls them from the sampler constants directly.

Build with:

```
python scripts/build_dataset.py --out datasets/regular_v4/dataset.h5 \
    --n-train 16000 --n-val 2000 --n-test 2000 --n-ood 2000 \
    --sweep-points 50 --seed 42
python scripts/inspect_dataset.py datasets/regular_v4/dataset.h5
```

[tools/pyg_dataset.py](tools/pyg_dataset.py) wraps the H5 as a PyG-compatible `Dataset`. `split` accepts `"train" | "val" | "test"` for the bulk loaders, `"ood_n_top_<N>"` for the held-out topology, or `"sweep:<axis>/n_top_<N>"` for the analysis grids. `droop_kind="peak"` (default) trains against transient droop; `droop_kind="static"` trains against DC IR drop. `target="linear"` returns droop in volts; `target="log"` returns `log10(droop)` — peak droop is heavy-tailed, so the log target trains more cleanly.

## Encoder + droop regressor

[tools/encoder.py](tools/encoder.py) implements the heterogeneous GNN encoder and a droop-regression head:

- `InputNormalizer` — log10 + z-score for log-scale columns (R, C, I_peak, freq), plain z-score for linear columns; statistics derived analytically from the parameter ranges so no fit-on-data is needed. Columns that are constant across the dataset (`I_peak`, `freq`, `duty`, `phase`, `R_via`, `Rsheet_*`) register with a bounded-sigma stat so they normalize to a stable zero — the GNN still sees them but they carry no per-sample signal. Edge attributes are the uniform 6-dim `[R, C, I_peak, freq, duty, phase]`; the normalizer outputs a 7-dim vector with `phase` replaced by `(sin 2πφ, cos 2πφ)`.
- `EdgeAwareConv` — generic message passing with `msg = MLP([x_i || x_j || edge_attr])`, sum aggregation, `update = MLP([x_i || agg])`. One instance per edge relation, wrapped in `HeteroConv` and stacked 3 deep with LayerNorm + residual.
- `PDNEncoder` — emits per-node hidden representations across the three node types (`mesh_top`, `mesh_bot`, `gnd`).
- `PDNDroopRegressor` — encoder + 2-layer MLP head over `mesh_bot` nodes; predicts log10(droop) by default. ~180k parameters at `hidden_dim=32` and 3 layers (~600k at the heavier `hidden_dim=64`).

Train with:

```
python scripts/train_droop.py --data datasets/regular_v4/dataset.h5 \
    --epochs 60 --batch-size 128 --lr 2e-3 --target log \
    --ckpt checkpoints/droop_v1.pt
python scripts/eval_droop.py --ckpt checkpoints/droop_v1.pt --split test
```

[tools/training.py](tools/training.py) reports MAE / RMSE in mV, R² across all (sample, node) pairs, and the per-sample worst-node-droop MAE — all in linear-volt space, regardless of training target.

## Targets

- Static IR drop per node (mV) for a given current-load configuration.
- Dynamic voltage-drop waveforms / peak droop, conditioned on switching activity.
- Sensitivity of worst-case droop to decap placement and grid-strap density (via gradients through the latent code, once the VAE side is wired up).

## Data

All PDN graphs and reference labels are synthetically generated. Topology is the regular `n_top × 7` mesh from [tools/grid_construction.py](tools/grid_construction.py); per-sample design parameters are LHS-sampled within the ranges in [tools/sampler.py](tools/sampler.py); ground truth is produced by the MNA solver in [tools/transient_solver.py](tools/transient_solver.py) (transient peak droop + DC IR drop per sample). No external benchmarks at this stage.

## Repository layout

```
docs/        - design notes (goal scope, stop conditions)
scripts/     - dataset build, training, evaluation, smoke / inspect helpers
tools/       - graph construction, MNA solver, sampler, PyG dataset, encoder
checkpoints/ - trained model artifacts (created on first train run)
datasets/    - synthetic dataset HDF5s (created on first build)
```

## Status

Forward pipeline (graph construction → MNA ground truth → PyG dataset → heterogeneous GNN encoder → droop regression head) is wired up end to end against the 3-knob v4 design space. Next milestone is fitting peak-droop and static IR-drop regression to the target accuracy in [docs/GOAL.md](docs/GOAL.md), then adding the VAE / generative head.
