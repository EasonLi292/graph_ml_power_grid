# graph_ml_power_grid

Graph machine learning for analyzing on-chip power delivery networks (PDNs). The model ingests an IC's power grid as a graph and predicts electrical behavior — primarily static IR drop and dynamic voltage drop — without running a full SPICE simulation.

PDNs are the **intermediate problem** here. The longer-term goal is a generative model over transistor-level analog circuits; PDN regression exercises the same encoder / VAE / generation scaffolding on a much smaller, regular design space first. See [docs/GOAL.md](docs/GOAL.md) for the full scope and stop-conditions.

## Motivation

As process nodes shrink and current densities climb, IR drop on the on-die power grid has become a first-order signoff problem. Commercial signoff tools (Voltus, RedHawk-SC) solve very large sparse linear systems for every PDN snapshot, which is slow inside a place-and-route loop. A learned surrogate that operates directly on the PDN graph can give designers a fast inner-loop signal for grid health, decap placement, and current-source planning.

## Architecture

This project reuses the encoder/decoder VAE skeleton from [Z-GED](https://github.com/EasonLi292/Z-GED), retargeted from RLC filter synthesis to power-grid analysis.

- **Encoder — impedance-aware GNN.** Three message-passing layers operating over the PDN graph. Nodes represent grid taps, vias, and instance pins; edges carry per-segment resistance (and inductance/capacitance for dynamic analysis). Hierarchical latent branches separate topology, per-segment electrical values, and load/source conditions, mirroring the Z-GED `[z_topology | z_values | z_pz]` split — here repurposed as `[z_topology | z_RLC | z_loads]`.
- **Decoder — autoregressive transformer.** A GPT-style decoder (4 layers, 256 dims) that, depending on the head, either (a) reconstructs the PDN as an Eulerian walk over the grid for self-supervised pretraining, or (b) emits per-node voltage / IR-drop predictions conditioned on the latent code and the load vector.
- **Training as a VAE.** The model is pretrained as a variational autoencoder on PDN graphs so that the latent space captures grid topology and impedance structure. A supervised head is then fine-tuned against ground-truth SPICE / signoff IR-drop solutions.

## Graph representation

A PDN is encoded as a heterogeneous graph with four node types and six edge types. Voltages are predicted per node; lumped R/C live on edges; load waveforms live on nodes.

**Node types**
- `mesh_top` — junction on the coarse top metal layer (M_top). Corners flagged `is_vdd_pad` carry a Dirichlet condition V = Vdd.
- `mesh_bot` — junction on the fine bottom metal layer (M_bot).
- `load` — current source. Features: `(I_peak, freq, duty, phase)` of a square-wave current draw.
- `gnd` — single ideal-zero reference node.

**Edge types** (each carries one scalar feature)
- `R_seg_top` — strap segment between adjacent `mesh_top` nodes (Ω).
- `R_seg_bot` — strap segment between adjacent `mesh_bot` nodes (Ω).
- `R_via` — via between co-located `mesh_top` and `mesh_bot` (Ω).
- `C_decap` — decoupling capacitor between `mesh_bot` and `gnd` (F).
- `I_in` — directed `mesh_bot → load`; current magnitude lives on the load node.
- `I_out` — directed `load → gnd`; same magnitude as `I_in` (KCL closes through the load).

**Canonical regular instance** ([tools/grid_construction.py](tools/grid_construction.py), `build_regular_pdn`): 4×4 M_top stacked on 7×7 M_bot, vias at every M_top node aligned to even M_bot positions, two supply-pad patterns (`corner` = 4 corners only, `checker` = alternating M_top nodes), loads on the even M_bot sub-grid *minus* any node directly under a Vdd-pad via (so a load can never sit on an ideal supply tap), 9 decaps offset onto the odd sub-grid. Per-segment `R_top`, `R_bot` are derived from sheet resistance × pitch / wire width: `R_seg = Rsheet × (pitch / wire_width)`. Loads share `(I_peak, freq, duty, phase)` (single global clock, in-phase). The pad pattern is drawn uniformly per sample, exposing the model to two distinct supply geometries.

Ground truth: backward-Euler MNA transient simulation in [tools/transient_solver.py](tools/transient_solver.py). Smoke test: `python scripts/smoke_test.py` builds the canonical instance, runs a 5 ns / 10 ps transient, and prints peak droop per M_bot node.

## Dataset

Grid size is fixed at (4, 7). Each sample has three parameter buckets:

**Global continuous** (one value per sample, LHS-sampled from `GLOBAL_RANGES`):

| param         | range            | scale       | meaning |
|---------------|------------------|-------------|---------|
| `Rsheet_top`  | 0.01 – 0.1 Ω/sq  | log-uniform | top-metal sheet resistance |
| `Rsheet_bot`  | 0.05 – 0.5 Ω/sq  | log-uniform | bottom-metal sheet resistance |
| `wire_width`  | 0.2 – 1.0        | log-uniform | strap width (in units of `pitch_bot`) |
| `R_via`       | 0.02 – 0.2 Ω     | log-uniform | per-via-stack resistance |
| `C_decap`     | 10 pF – 1 nF     | log-uniform | per-decap capacitance |
| `freq`        | 200 MHz – 4 GHz  | log-uniform | clock frequency (single domain) |

**Per-load continuous** (independently sampled for *each* load instance within a sample, from `PER_LOAD_RANGES`):

| param      | range          | scale       |
|------------|----------------|-------------|
| `I_peak`   | 1 mA – 20 mA   | log-uniform |
| `duty`     | 0.2 – 0.6      | uniform     |
| `phase`    | 0 – 1          | uniform     |

Each load has its own activity factor (closer to real "different gate types per instance"). The encoder maps the raw phase into `(sin 2πφ, cos 2πφ)` so it gets the circular continuity for free.

**Topology** (discrete, uniform per sample over the chosen pattern pool): one of four pad patterns — `corner` (4 pads), `checker` (8), `edge_strip` (12), `distributed` (8, corners + interior 2×2). Three patterns (`corner`, `checker`, `edge_strip`) are used for training/val/test; `distributed` is held out as an OOD topology probe.

Per-segment R the solver actually stamps is derived: `R_top = Rsheet_top × (pitch_top / wire_width)` and similarly for `R_bot`. At the default topology `pitch_top = 2 × pitch_bot`, so the top straps pick up a 2× geometric factor even at equal sheet R.

Per-sample sim: warmup of `max(2 periods, ceil(5·R_bot·C_decap / period))` periods so the initial-condition transient is ≲1% residual regardless of the freq/RC ratio, then 8 measurement periods × 100 steps/period. Each sample yields two labels: **peak droop** (Vdd − Vmin over the measurement window) and **static IR drop** (DC solve under the time-averaged load current, `I_peak × duty` per load).

### H5 layout

```
/                                       attrs: version=3, ranges, topology, sim_config, ...
├── bulk/{train, val, test}/            LHS over the 3 training patterns
│     ├── global_params [N, 6]
│     ├── pad_pattern_idx [N]
│     ├── load_x [N, max_n_loads=12, 4]   per-load (I_peak, freq, duty, phase), zero-padded
│     ├── n_loads [N]
│     ├── peak_droop_bot   [N, 49]
│     ├── static_droop_bot [N, 49]
│     ├── peak_droop_top,  static_droop_top, worst_node_*
│     └── V_subset/ (train only — first 200 samples' full V(t))
├── ood/distributed/                    held-out pad pattern; same fields
└── analysis/sweeps/<axis>/<pattern>/   1-D sweep with all-others-at-median; 50 pts default
```

Build with:

```
python scripts/build_dataset.py --out datasets/regular_v2/dataset.h5 \
    --n-train 16000 --n-val 2000 --n-test 2000 --n-ood 2000 \
    --sweep-points 50 --seed 42
python scripts/inspect_dataset.py datasets/regular_v2/dataset.h5
```

[tools/pyg_dataset.py](tools/pyg_dataset.py) wraps the H5 as a PyG-compatible `Dataset`. `split` accepts `"train" | "val" | "test"` for the bulk loaders, `"ood_distributed"` for the held-out pattern, or `"sweep:<axis>/<pattern>"` for the analysis grids. `droop_kind="peak"` (default) trains against transient droop; `droop_kind="static"` trains against DC IR drop. `target="linear"` returns droop in volts; `target="log"` returns `log10(droop)` — peak droop is heavy-tailed (linear skew ≈ 1.8, log skew ≈ 0), so the log target trains more cleanly in practice.

## Encoder + droop regressor

[tools/encoder.py](tools/encoder.py) implements the heterogeneous GNN encoder and a droop-regression head:

- `InputNormalizer` — log10 + z-score for log-scale params (R/C, I_peak, freq), plain z-score for linear params; statistics derived analytically from the parameter ranges so no fit-on-data is needed. Phase is encoded as `(sin 2πφ, cos 2πφ)` so `load.x → load_proj_input` has dim 5.
- `EdgeAwareConv` — generic message passing with `msg = MLP([x_i || x_j || edge_attr])`, sum aggregation, `update = MLP([x_i || agg])`. One instance per edge type, wrapped in `HeteroConv` and stacked 3 deep with LayerNorm + residual.
- `PDNEncoder` — emits per-node hidden representations across all four node types.
- `PDNDroopRegressor` — encoder + 2-layer MLP head over `mesh_bot` nodes; predicts log10(droop) by default. ~615k parameters at the default `hidden_dim=64`.

Train with:

```
python scripts/train_droop.py --data datasets/regular_v2/dataset.h5 \
    --epochs 60 --batch-size 128 --lr 2e-3 --target log \
    --ckpt checkpoints/droop_v1.pt
python scripts/eval_droop.py --ckpt checkpoints/droop_v1.pt --split test
```

[tools/training.py](tools/training.py) reports MAE / RMSE in mV, R² across all (sample, node) pairs, and the per-sample worst-node-droop MAE — all in linear-volt space, regardless of training target.

## Targets

- Static IR drop per node (mV) for a given current-load vector.
- Dynamic voltage-drop waveforms / peak droop, conditioned on switching activity.
- Sensitivity of worst-case droop to decap placement and grid-strap density (via gradients through the latent code).

## Data

PDN graphs and reference solutions are sourced from open benchmarks (e.g. IBM power-grid benchmarks, ICCAD contest grids) plus synthetically generated grids with randomized strap pitch, via density, and load maps. Ground truth comes from a SPICE-style nodal solver run offline.

## Repository layout

```
ml/          - data loaders, GNN encoder, transformer decoder, VAE wiring
scripts/     - training, evaluation, IR-drop inference
tests/       - unit tests and grid-spec checks
tools/       - PDN parsing, graph construction, SPICE ground-truth runners
checkpoints/ - trained model artifacts
datasets/    - benchmark and synthetic PDN graphs
```

## Status

Early scaffolding. Architecture and data pipeline are being ported from Z-GED; first milestone is static IR-drop regression on the IBM PG benchmarks.
