# graph_ml_power_grid

Graph machine learning for analyzing on-chip power delivery networks (PDNs). The model ingests an IC's power grid as a graph and predicts electrical behavior — primarily static IR drop and dynamic voltage drop — without running a full SPICE simulation.

## Motivation

As process nodes shrink and current densities climb, IR drop on the on-die power grid has become a first-order signoff problem. Commercial signoff tools (Voltus, RedHawk-SC) solve very large sparse linear systems for every PDN snapshot, which is slow inside a place-and-route loop. A learned surrogate that operates directly on the PDN graph can give designers a fast inner-loop signal for grid health, decap placement, and current-source planning.

## Architecture

This project reuses the encoder/decoder VAE skeleton from [Z-GED](https://github.com/EasonLi292/Z-GED), retargeted from RLC filter synthesis to power-grid analysis.

- **Encoder — impedance-aware GNN.** Three message-passing layers operating over the PDN graph. Nodes represent grid taps, vias, and instance pins; edges carry per-segment resistance (and inductance/capacitance for dynamic analysis). Hierarchical latent branches separate topology, per-segment electrical values, and load/source conditions, mirroring the Z-GED `[z_topology | z_values | z_pz]` split — here repurposed as `[z_topology | z_RLC | z_loads]`.
- **Decoder — autoregressive transformer.** A GPT-style decoder (4 layers, 256 dims) that, depending on the head, either (a) reconstructs the PDN as an Eulerian walk over the grid for self-supervised pretraining, or (b) emits per-node voltage / IR-drop predictions conditioned on the latent code and the load vector.
- **Training as a VAE.** The model is pretrained as a variational autoencoder on PDN graphs so that the latent space captures grid topology and impedance structure. A supervised head is then fine-tuned against ground-truth SPICE / signoff IR-drop solutions.

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
