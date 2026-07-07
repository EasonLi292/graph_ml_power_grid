"""Per-node droop regressor for the IBM (real-grid) graphs.

Same conv core as the synthetic model — :class:`EdgeConvGated` with a
MonotoneGate on branch admittance — but on a single homogeneous node
type with three passive relations (R, C, L) and a per-node readout:

* R edges → ``kind="resistor"``  (gate increases with conductance)
* C edges → ``kind="capacitor"`` (gate increases with capacitance)
* L edges → ``kind="resistor"``  on z(log L): higher inductance =
  higher transient impedance, same monotone direction as R.

Loads and ground decaps are node *features* here (IBM loads go
node→ground, not Vdd→Vss pairs), so there is no load relation; the
injection enters through the input projection.

``transfer_from_synthetic`` copies the conv stack (delta/gate/update
MLPs — all shaped [hidden] only) from a trained synthetic
``PDNDroopRegressor`` checkpoint: bot-strap conv → R conv, decap conv →
C conv, top-strap conv → L conv (an R-like init beats random). Input
projection and head are new (different feature spaces).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.ibmpg_patches import NODE_FEATURE_DIM

from .convs import EdgeConvGated

RELS = ("R", "C", "L")
_KIND = {"R": "resistor", "C": "capacitor", "L": "resistor"}


class IBMNodeRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 64, n_layers: int = 7) -> None:
        super().__init__()
        h = hidden_dim
        self.n_layers = n_layers
        self.proj = nn.Linear(NODE_FEATURE_DIM, h)
        self.convs = nn.ModuleList([
            nn.ModuleDict({r: EdgeConvGated(h, kind=_KIND[r]) for r in RELS})
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, data) -> torch.Tensor:
        x = self.proj(data["node"].x)
        for layer, norm in zip(self.convs, self.norms):
            out = torch.zeros_like(x)
            for r in RELS:
                et = ("node", r, "node")
                ei = data[et].edge_index
                if ei.numel():
                    out = out + layer[r](x, ei, data[et].edge_attr)
            x = norm(self.dropout_free_relu(out) + x)
        return self.head(x).squeeze(-1)          # log10 droop per node

    @staticmethod
    def dropout_free_relu(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


# Synthetic-relation → IBM-relation conv mapping for the transfer.
# PyG's HeteroConv ModuleDict keys look like '<mesh_bot___strap___mesh_bot>'.
_TRANSFER_MAP = {
    "R": "<mesh_bot___strap___mesh_bot>",
    "C": "<mesh_bot___decap___mesh_bot>",
    "L": "<mesh_top___strap___mesh_top>",
}


def transfer_from_synthetic(model: IBMNodeRegressor, ckpt_path: str) -> int:
    """Copy conv-stack weights from a synthetic edgeconv checkpoint.

    Returns the number of tensors copied. Layer i of the IBM model takes
    layer i of the synthetic encoder (same depth expected; extra IBM
    layers keep their fresh init).
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    copied = 0
    own = model.state_dict()
    for i in range(model.n_layers):
        for rel, syn_key in _TRANSFER_MAP.items():
            src_prefix = f"encoder.convs.{i}.convs.{syn_key}."
            dst_prefix = f"convs.{i}.{rel}."
            for k in list(state.keys()):
                if k.startswith(src_prefix):
                    dst = dst_prefix + k[len(src_prefix):]
                    if dst in own and own[dst].shape == state[k].shape:
                        own[dst] = state[k]
                        copied += 1
    model.load_state_dict(own)
    return copied
