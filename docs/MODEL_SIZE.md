# Model Size & Parameter Count

Exact size of `PDNDroopRegressor` at the default config (`hidden_dim = 64`,
`n_layers = 7`, `conv_type = "admittance"`). Reproduce: `python3.12
scripts/model_params.py`.

## Headline

| | value |
|---|---|
| **Trainable parameters** | **1,000,093** |
| float32 footprint (params only) | 4.00 MB (4,000,372 bytes) |
| on-disk checkpoint (`droop_v5_nocoord.pt`) | 4.20 MB |
| state-dict tensors | 462 |

The on-disk checkpoint is a bit larger than the raw 4.00 MB because it also
stores the normalizer's (non-trainable) analytic mean/std **buffers** and a
little metadata: it is a dict `{"model": state_dict, "epoch": int,
"val_metrics": {...}}`. No optimizer state is saved.

---

## Top-level breakdown

| component | params | share |
|---|---:|---:|
| `encoder.node_proj` | 384 | 0.0% |
| `encoder.edge_proj` | 3,072 | 0.3% |
| **`encoder.convs`** (7 message-passing layers) | **986,524** | **98.6%** |
| `encoder.norms` (LayerNorms) | 1,792 | 0.2% |
| `head` (per-load readout MLP) | 8,321 | 0.8% |
| `encoder.normalizer` | 0 | — |
| **total** | **1,000,093** | 100% |

The message-passing stack is essentially the entire model; everything around it
(input projections, norms, head) is < 1.5% combined.

### The small parts

- **`node_proj`** — one `Linear(2 → 64)` per node type (`mesh_top`, `mesh_bot`):
  `2 × (2·64 + 64) = 384`. (Node features are just `[is_vdd, is_pad]`, hence the
  `2`.)
- **`edge_proj`** — one `Linear(7 → 64)` per edge relation: `6 × 512 = 3,072`.
  ⚠️ Note: the `strap` and `via` relations use the deterministic conductance
  gate and **bypass `edge_proj`** at forward time, so 4 of these 6 projections
  (`4 × 512 = 2,048` params) are *allocated but never used* — dead weight that
  could be pruned.
- **`norms`** — `LayerNorm(64)` (weight + bias = 128) per node type per layer:
  `7 × 2 × 128 = 1,792`.
- **`head`** — `Linear(128 → 64)` + `Linear(64 → 1)` = `8,256 + 65 = 8,321`. The
  input is 128-dim because it concatenates the two load-edge endpoint states
  (64 + 64).
- **`normalizer`** — analytic z-score (stats derived from the parameter ranges),
  so **0 learnable parameters** (it holds non-trainable buffers instead).

---

## Inside one message-passing layer (140,932 params)

Each of the 7 layers is a `HeteroConv` holding one conv per edge relation. The
conv *kind* depends on the relation, and that sets its parameter count:

| relation | conv kind | params (per layer) |
|---|---|---:|
| `mesh_top → strap → mesh_top` | conductance | 20,737 |
| `mesh_bot → strap → mesh_bot` | conductance | 20,737 |
| `mesh_top → via → mesh_bot` | conductance | 20,737 |
| `mesh_bot → via → mesh_top` | conductance | 20,737 |
| `mesh_bot → decap → mesh_bot` | admittance | 29,056 |
| `mesh_bot → load → mesh_bot` | source | 28,928 |
| **one layer total** | | **140,932** |
| **× 7 layers** | | **986,524** |

### Per-submodule (where the params live inside a conv)

All MLPs are 2-layer (`Linear → ReLU → Linear`); `h = 64`.

| submodule | shape | params |
|---|---|---:|
| `delta_mlp` | `Linear(64→64)` ×2 | 8,320 |
| `gate_mlp` (admittance only) | `Linear(64→64)` ×2 | 8,320 |
| `msg_mlp` (source only) | `Linear(192→64)` + `Linear(64→64)` | 16,512 |
| `upd_mlp` (every conv) | `Linear(128→64)` + `Linear(64→64)` | 12,416 |
| `alpha` (conductance gate) | scalar | **1** |

So a **conductance** conv = `alpha (1) + delta_mlp (8,320) + upd_mlp (12,416) =
20,737`. An **admittance** conv adds a `gate_mlp` (8,320) → 29,056. A **source**
conv replaces the delta path with a wider `msg_mlp` (concatenates `[h_i ‖ h_j ‖
edge_attr]`, 192-dim in) → 28,928.

**The conductance gate is almost free.** Its entire learnable footprint is the
*single scalar* `α` per conv — 4 of them per layer, 28 in the whole model. The
physics shaping costs 28 parameters total; the rest is the generic `delta_mlp` /
`upd_mlp` machinery shared with any GNN.

---

## How size scales with width and depth

Exact closed form (verified against the built model for every entry):

```
N(h, L) = (34·L + 2)·h²  +  (30·L + 56)·h  +  (4·L + 1)
```

where `h = hidden_dim`, `L = n_layers`. The leading term `34·L·h²` is the conv
stack — quadratic in width, linear in depth. At the default `h=64, L=7`:
`(34·7+2)·4096 + (30·7+56)·64 + 29 = 983,040 + 17,024 + 29 = 1,000,093`.

| hidden_dim ＼ n_layers | 3 | 5 | **7** |
|---|---:|---:|---:|
| 32 | 111,181 | 182,741 | 254,301 |
| **64** | 435,341 | 717,717 | **1,000,093** |
| 128 | 1,722,637 | 2,844,437 | 3,966,237 |

**Depth is cheap, width is expensive.** 3 → 7 layers at `h=64` roughly doubles
the model (435k → 1.00M); 64 → 128 width nearly quadruples it (1.00M → 3.97M).
Depth is also the dominant accuracy lever ([MODEL_REPORT.md](MODEL_REPORT.md)
§3), so performance is bought on the cheaper axis.

---

## Reproduce

```bash
python3.12 scripts/model_params.py
```
