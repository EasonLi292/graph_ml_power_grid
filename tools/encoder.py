"""Heterogeneous GNN encoder for the regular PDN.

Architecture
------------
* ``InputNormalizer`` — log10 + z-score on log-scale params, plain z-score
  otherwise. Statistics are derived analytically from the parameter ranges
  in [tools/sampler.py], so no fit-on-data is needed.
* ``EdgeAwareConv`` — message ``MLP([x_i, x_j, edge_attr])``, sum aggregator,
  update ``MLP([x_i, agg])``. Used as the per-relation conv inside
  ``HeteroConv``.
* ``PDNEncoder`` — 3 stacked HeteroConv layers with LayerNorm + residual.
  Returns hidden representations per node type.
* ``PDNDroopRegressor`` — wraps the encoder with a per-``mesh_bot`` MLP head
  predicting one scalar (peak droop, in either linear or log10 space).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing

from .grid_construction import build_regular_pdn
from .sampler import DEFAULT_RANGES, ParamRanges, derived_R_ranges


# ----- node / edge type constants (kept in one place) -----

NODE_TYPES = ("mesh_top", "mesh_bot", "load", "gnd")

EDGE_TYPES = (
    ("mesh_top", "R_top", "mesh_top"),
    ("mesh_bot", "R_bot", "mesh_bot"),
    ("mesh_top", "R_via", "mesh_bot"),
    ("mesh_bot", "R_via", "mesh_top"),
    ("mesh_bot", "C_decap", "gnd"),
    ("gnd", "C_decap_rev", "mesh_bot"),
    ("mesh_bot", "I_in", "load"),
    ("load", "I_in_rev", "mesh_bot"),
    ("load", "I_out", "gnd"),
    ("gnd", "I_out_rev", "load"),
)

# Edges with a physical scalar (rest are "wire" edges with no edge_attr in the dataset).
EDGES_WITH_SCALAR = {
    ("mesh_top", "R_top", "mesh_top"): "R_top",
    ("mesh_bot", "R_bot", "mesh_bot"): "R_bot",
    ("mesh_top", "R_via", "mesh_bot"): "R_via",
    ("mesh_bot", "R_via", "mesh_top"): "R_via",
    ("mesh_bot", "C_decap", "gnd"): "C_decap",
    ("gnd", "C_decap_rev", "mesh_bot"): "C_decap",
}


# ----- input normalization -----

class InputNormalizer(nn.Module):
    """Normalizes load.x and edge_attr using a priori parameter range stats.

    Sampled-param stats (``Rsheet_top``, ``wire_width``, ``I_peak``, ...)
    come straight from ``ranges``. Edge attributes the model actually sees
    are the derived per-segment resistances ``R_top`` / ``R_bot``; we
    register stats for those analytically from
    ``derived_R_ranges(ranges, pitch_top, pitch_bot)``.
    """

    def __init__(self, ranges: ParamRanges = DEFAULT_RANGES) -> None:
        super().__init__()
        log_params: set[str] = set()

        def _register(name: str, lo: float, hi: float, scale: str) -> None:
            if scale == "log":
                lo_t, hi_t = math.log10(lo), math.log10(hi)
                log_params.add(name)
            else:
                lo_t, hi_t = lo, hi
            mu = 0.5 * (lo_t + hi_t)
            sigma = (hi_t - lo_t) / math.sqrt(12) + 1e-8
            self.register_buffer(f"mu_{name}", torch.tensor(mu, dtype=torch.float32))
            self.register_buffer(f"sigma_{name}", torch.tensor(sigma, dtype=torch.float32))

        for p in ranges.params:
            _register(p.name, p.lo, p.hi, p.scale)

        # Derived per-segment R_top / R_bot for the canonical topology.
        # Pad pattern doesn't affect pitch, only pad placement.
        proto = build_regular_pdn()
        for name, (lo, hi, scale) in derived_R_ranges(
            ranges, proto.pitch_top, proto.pitch_bot
        ).items():
            _register(name, lo, hi, scale)

        self._log_params = log_params

    def _norm_scalar(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name in self._log_params:
            x = torch.log10(x.clamp_min(1e-15))
        mu = getattr(self, f"mu_{name}")
        sigma = getattr(self, f"sigma_{name}")
        return (x - mu) / sigma

    def normalize_load(self, load_x: torch.Tensor) -> torch.Tensor:
        # load_x: [N, 4] = (I_peak, freq, duty, phase ∈ [0, 1])
        # Output: [N, 5] — phase encoded as (sin 2πφ, cos 2πφ) so the model
        # gets the natural circular continuity (φ=0 ≡ φ=1) for free.
        I = self._norm_scalar(load_x[:, 0:1], "I_peak")
        f = self._norm_scalar(load_x[:, 1:2], "freq")
        d = self._norm_scalar(load_x[:, 2:3], "duty")
        ph = load_x[:, 3:4]
        sin_ph = torch.sin(2 * math.pi * ph)
        cos_ph = torch.cos(2 * math.pi * ph)
        return torch.cat([I, f, d, sin_ph, cos_ph], dim=1)

    def normalize_edge_attr(self, attr: torch.Tensor, name: str) -> torch.Tensor:
        return self._norm_scalar(attr, name)


# ----- conv -----

class EdgeAwareConv(MessagePassing):
    """Generic edge-conditioned message passing.

    msg_ij = MLP([x_i || x_j || edge_attr]); aggr = sum;
    upd_i  = MLP([x_i || agg_i]).
    Works on bipartite (src ≠ dst) edges via the standard PyG convention.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__(aggr="sum")
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # x is either Tensor (homogeneous) or (src, dst) tuple (bipartite)
        if isinstance(x, torch.Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.upd_mlp(torch.cat([x[1], out], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


# ----- encoder -----

@dataclass
class EncoderConfig:
    hidden_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.0


class PDNEncoder(nn.Module):
    """3-layer heterogeneous GNN backbone over the canonical PDN topology."""

    NODE_IN_DIM = {"mesh_top": 3, "mesh_bot": 2, "load": 5, "gnd": 1}

    def __init__(
        self,
        cfg: EncoderConfig | None = None,
        ranges: ParamRanges = DEFAULT_RANGES,
    ) -> None:
        super().__init__()
        cfg = cfg or EncoderConfig()
        self.cfg = cfg
        self.normalizer = InputNormalizer(ranges)

        h = cfg.hidden_dim
        self.node_proj = nn.ModuleDict(
            {nt: nn.Linear(self.NODE_IN_DIM[nt], h) for nt in NODE_TYPES}
        )

        # One projection per edge type (edge_attr is a single scalar — either the
        # physical R/C value, or a placeholder constant for wire edges).
        self.edge_proj = nn.ModuleDict(
            {self._et_key(et): nn.Linear(1, h) for et in EDGE_TYPES}
        )

        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {et: EdgeAwareConv(h) for et in EDGE_TYPES},
                    aggr="sum",
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict({nt: nn.LayerNorm(h) for nt in NODE_TYPES})
                for _ in range(cfg.n_layers)
            ]
        )
        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _et_key(et: tuple[str, str, str]) -> str:
        return "__".join(et)

    def _build_edge_attr_dict(self, data: HeteroData) -> dict:
        out = {}
        device = data["mesh_bot"].x.device
        for et in EDGE_TYPES:
            if et in EDGES_WITH_SCALAR:
                scalar_name = EDGES_WITH_SCALAR[et]
                raw = data[et].edge_attr  # [E, 1]
                normed = self.normalizer.normalize_edge_attr(raw, scalar_name)
            else:
                # Wire edges (I_in, I_out): give them a constant 1, the per-edge
                # Linear lets the model learn a relation-specific bias.
                num_edges = data[et].edge_index.shape[1]
                normed = torch.ones((num_edges, 1), device=device)
            out[et] = self.edge_proj[self._et_key(et)](normed)
        return out

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        # Per-node-type input projection (with normalization on load.x)
        x_dict = {
            "mesh_top": data["mesh_top"].x,
            "mesh_bot": data["mesh_bot"].x,
            "load": self.normalizer.normalize_load(data["load"].x),
            "gnd": data["gnd"].x,
        }
        x_dict = {nt: self.node_proj[nt](x) for nt, x in x_dict.items()}

        edge_attr_dict = self._build_edge_attr_dict(data)

        for conv, norm in zip(self.convs, self.norms):
            out = conv(x_dict, data.edge_index_dict, edge_attr_dict)
            x_dict = {
                nt: norm[nt](self.dropout(F.relu(out[nt])) + x_dict[nt])
                for nt in NODE_TYPES
                if nt in out
            }
        return x_dict


class PDNDroopRegressor(nn.Module):
    """Encoder + per-``mesh_bot`` scalar head.

    ``target_space="log"`` predicts log10(droop); ``"linear"`` predicts droop
    directly. The model itself is identical — only training loss / inference
    inverse-transform changes.
    """

    def __init__(
        self,
        cfg: EncoderConfig | None = None,
        ranges: ParamRanges = DEFAULT_RANGES,
        target_space: str = "log",
    ) -> None:
        super().__init__()
        self.encoder = PDNEncoder(cfg, ranges)
        h = self.encoder.cfg.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        self.target_space = target_space

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = self.encoder(data)["mesh_bot"]
        return self.head(x).squeeze(-1)
