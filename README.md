# graph_ml_power_grid

Graph machine learning for analyzing on-chip power delivery networks (PDNs). The model ingests an IC's power grid as a graph and predicts electrical behavior — primarily static IR drop and dynamic voltage drop — without running a full SPICE simulation.

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

**Canonical regular instance** ([tools/grid_construction.py](tools/grid_construction.py), `build_regular_pdn`): 4×4 M_top stacked on 7×7 M_bot, vias at every M_top node aligned to even M_bot positions, four `is_vdd_pad` corners, 16 identical loads on the even M_bot sub-grid, 9 decaps offset onto the odd sub-grid. All `R_seg_top`, `R_seg_bot`, `R_via`, `C_decap` are equal across edges; all loads share `(I_peak, freq, duty, phase)` (single global clock, in-phase). This worst-case simultaneous-draw setup is the first benchmark for the encoder — once it learns the Vdd-drop map here, randomized variants (per-segment R, per-load phase) follow.

Ground truth: backward-Euler MNA transient simulation in [tools/transient_solver.py](tools/transient_solver.py). Smoke test: `python scripts/smoke_test.py` builds the canonical instance, runs a 5 ns / 10 ps transient, and prints peak droop per M_bot node.

## Dataset

Topology stays fixed; seven scalar parameters vary per sample, drawn via Latin Hypercube from [tools/sampler.py](tools/sampler.py)'s `DEFAULT_RANGES`:

| param      | range            | scale       |
|------------|------------------|-------------|
| `R_top`    | 0.05 – 0.5 Ω     | log-uniform |
| `R_bot`    | 0.2 – 5 Ω        | log-uniform |
| `R_via`    | 0.02 – 0.2 Ω     | log-uniform |
| `C_decap`  | 10 pF – 1 nF     | log-uniform |
| `I_peak`   | 1 mA – 20 mA     | log-uniform |
| `freq`     | 200 MHz – 4 GHz  | log-uniform |
| `duty`     | 0.2 – 0.6        | uniform     |

`Vdd = 1 V` and `phase = 0` are fixed (phase doesn't affect periodic-steady-state peak droop). Per-sample sim: warmup of `max(2 periods, ceil(5·R_bot·C_decap / period))` periods so the initial-condition transient is ≲1% residual regardless of the freq/RC ratio, then 8 measurement periods × 100 steps/period; peak droop is taken over the measurement window.

Default split: 8000 / 1000 / 1000 train/val/test, seeded independently. An optional `--n-extrapolation` flag draws from `EXTRAPOLATION_RANGES` (parameter ranges shifted just outside the training box) — off by default. Targets stored per sample:

- `peak_droop_bot[49]` — primary regression target.
- `peak_droop_top[16]` — secondary.
- `worst_node_idx`, `worst_node_droop` — convenience scalars.
- For 100 train samples only: full `V_bot[T, 49]` and `V_top[T, 16]` waveforms (diagnostics; basis for a future transient head).

The H5 layout, parameter ranges, sim config, topology, and git SHA are persisted in file/group attrs. Build with:

```
python scripts/build_dataset.py --out datasets/regular_v1/dataset.h5 \
    --n-train 8000 --n-val 1000 --n-test 1000 --seed 42
python scripts/inspect_dataset.py datasets/regular_v1/dataset.h5
```

[tools/pyg_dataset.py](tools/pyg_dataset.py) wraps the H5 file as a PyG-compatible `Dataset`. The full topology lives in a single canonical `HeteroData` template; per-sample edge_attr (R/C) and `load.x` (I_peak/freq/duty/phase) are overwritten on the fly. `target="linear"` returns droop in volts; `target="log"` returns `log10(droop)` — peak-droop is heavy-tailed (linear skew ≈ 1.8, log skew ≈ 0), so the log target trains more cleanly in practice.

## Encoder + droop regressor

[tools/encoder.py](tools/encoder.py) implements the heterogeneous GNN encoder and a droop-regression head:

- `InputNormalizer` — log10 + z-score for log-scale params (R/C, I_peak, freq), plain z-score for linear params; statistics derived analytically from the parameter ranges so no fit-on-data is needed.
- `EdgeAwareConv` — generic message passing with `msg = MLP([x_i || x_j || edge_attr])`, sum aggregation, `update = MLP([x_i || agg])`. One instance per edge type, wrapped in `HeteroConv` and stacked 3 deep with LayerNorm + residual.
- `PDNEncoder` — emits per-node hidden representations across all four node types.
- `PDNDroopRegressor` — encoder + 2-layer MLP head over `mesh_bot` nodes; predicts log10(droop) by default. ~615k parameters at the default `hidden_dim=64`.

Train with:

```
python scripts/train_droop.py --data datasets/regular_v1/dataset.h5 \
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
