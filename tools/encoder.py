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

from .grid_construction import (
    EDGE_ATTR_COLS,
    EDGE_ATTR_DIM,
    NODE_FEATURE_DIM,
    build_regular_pdn,
)
from .sampler import ALL_N_TOP, DEFAULT_RANGES, FIXED_PAD_PATTERN, ParamRanges, derived_R_ranges


# ----- node / edge type constants (kept in one place) -----

NODE_TYPES = ("mesh_top", "mesh_bot", "gnd")

# Five logical bidirectional relations: strap (within mesh_top), strap
# (within mesh_bot), via (across types), decap (across types), load
# (across types). Same-type bidir lives in one relation with edge_index
# packed; cross-type bidir is expressed as two PyG relations sharing the
# same relation name.
EDGE_TYPES = (
    ("mesh_top", "strap", "mesh_top"),
    ("mesh_bot", "strap", "mesh_bot"),
    ("mesh_top", "via", "mesh_bot"),
    ("mesh_bot", "via", "mesh_top"),
    ("mesh_bot", "decap", "gnd"),
    ("gnd", "decap", "mesh_bot"),
    ("mesh_bot", "load", "gnd"),
    ("gnd", "load", "mesh_bot"),
)

# Normalized edge-attribute dimension: 6 raw columns
# [R, C, I_peak, freq, duty, phase] become 7 after phase → (sin, cos).
EDGE_ATTR_DIM_NORMALIZED = EDGE_ATTR_DIM + 1


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
            if lo == hi:
                # Fixed param: sigma must be bounded well away from zero so
                # any float32 jitter at the value doesn't blow up the
                # normalized column. The normalized value is ~ 0 by design.
                sigma = 1.0
            else:
                sigma = (hi_t - lo_t) / math.sqrt(12) + 1e-8
            self.register_buffer(f"mu_{name}", torch.tensor(mu, dtype=torch.float32))
            self.register_buffer(f"sigma_{name}", torch.tensor(sigma, dtype=torch.float32))

        for p in ranges.params:
            _register(p.name, p.lo, p.hi, p.scale)

        # Derived per-segment R_top / R_bot. Pitch_top changes with n_top
        # (coarser top mesh → longer segment), so the analytic range needs
        # to span the union over every n_top this dataset emits. Pitch_bot
        # is invariant (n_bot fixed).
        pitch_tops, pitch_bots = [], []
        for nt in ALL_N_TOP:
            proto_nt = build_regular_pdn(n_top=nt, pad_pattern=FIXED_PAD_PATTERN)
            pitch_tops.append(proto_nt.pitch_top)
            pitch_bots.append(proto_nt.pitch_bot)
        agg = {}
        for pt, pb in zip(pitch_tops, pitch_bots):
            for name, (lo, hi, scale) in derived_R_ranges(ranges, pt, pb).items():
                if name not in agg:
                    agg[name] = [lo, hi, scale]
                else:
                    agg[name][0] = min(agg[name][0], lo)
                    agg[name][1] = max(agg[name][1], hi)
        for name, (lo, hi, scale) in agg.items():
            _register(name, lo, hi, scale)

        self._log_params = log_params

    def _norm_scalar(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name in self._log_params:
            x = torch.log10(x.clamp_min(1e-15))
        mu = getattr(self, f"mu_{name}")
        sigma = getattr(self, f"sigma_{name}")
        return (x - mu) / sigma

    def normalize_edge_attr(
        self, attr: torch.Tensor, relation: tuple[str, str, str]
    ) -> torch.Tensor:
        """Normalize a 6-dim raw edge attribute to 7-dim.

        Layout in: ``[R, C, I_peak, freq, duty, phase]``.
        Layout out: ``[R_n, C_n, I_n, f_n, d_n, sin 2πφ, cos 2πφ]``.

        Each relation only populates the columns relevant to its physical
        role; the rest stay zero. For load edges, phase becomes the usual
        circular (sin, cos) encoding so φ=0 ≡ φ=1.

        For resistor edges, the R column uses a relation-specific stat:
        top-strap → ``R_top`` (derived range), bot-strap → ``R_bot``
        (derived range), via → ``R_via`` (sampled range).
        """
        E = attr.shape[0]
        out = torch.zeros((E, EDGE_ATTR_DIM_NORMALIZED), device=attr.device, dtype=attr.dtype)
        rel_name = relation[1]

        if rel_name == "strap":
            stat = "R_top" if relation[0] == "mesh_top" else "R_bot"
            out[:, 0:1] = self._norm_scalar(attr[:, 0:1], stat)
        elif rel_name == "via":
            out[:, 0:1] = self._norm_scalar(attr[:, 0:1], "R_via")
        elif rel_name == "decap":
            out[:, 1:2] = self._norm_scalar(attr[:, 1:2], "C_decap")
        elif rel_name == "load":
            out[:, 2:3] = self._norm_scalar(attr[:, 2:3], "I_peak")
            out[:, 3:4] = self._norm_scalar(attr[:, 3:4], "freq")
            out[:, 4:5] = self._norm_scalar(attr[:, 4:5], "duty")
            ph = attr[:, 5:6]
            out[:, 5:6] = torch.sin(2 * math.pi * ph)
            out[:, 6:7] = torch.cos(2 * math.pi * ph)
        else:
            raise ValueError(f"unknown relation: {relation!r}")

        return out


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
    """Heterogeneous GNN backbone over the redesigned PDN graph.

    Node features are uniform 6-dim ``[one_hot_type(3), payload(3)]``. The
    explicit type signal in the input means message passing can never
    "forget" what kind of node it's processing, even though we still keep
    per-node-type ``Linear`` projections for inductive bias.

    Edge features are uniform 7-dim after normalization (the raw 6-dim
    ``[R, C, I_peak, freq, duty, phase]`` with phase replaced by
    ``(sin 2πφ, cos 2πφ)``). Per-relation ``Linear`` projects to the
    hidden dim.
    """

    NODE_IN_DIM = {nt: NODE_FEATURE_DIM for nt in NODE_TYPES}

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

        # Per-relation projection from the 7-dim normalized edge attribute.
        self.edge_proj = nn.ModuleDict(
            {self._et_key(et): nn.Linear(EDGE_ATTR_DIM_NORMALIZED, h) for et in EDGE_TYPES}
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
        for et in EDGE_TYPES:
            raw = data[et].edge_attr  # [E, EDGE_ATTR_DIM]
            normed = self.normalizer.normalize_edge_attr(raw, et)
            out[et] = self.edge_proj[self._et_key(et)](normed)
        return out

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {nt: self.node_proj[nt](data[nt].x) for nt in NODE_TYPES}
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
