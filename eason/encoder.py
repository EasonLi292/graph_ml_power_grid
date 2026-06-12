"""Encoder backbone + load-edge readout head.

* ``EncoderConfig`` — dataclass of the encoder knobs: hidden width,
  depth, node dropout, per-relation conv type, and DropEdge rate.
* ``PDNEncoder`` — stacked ``HeteroConv`` layers with LayerNorm +
  residual. Node features are uniform 6-dim ``[one_hot_type(2),
  is_vdd, is_pad, x, y]``; edge features are uniform 7-dim
  post-normalization. Optional **DropEdge** (drops a fraction of
  ``strap`` and ``decap`` edges per layer during training, never via /
  load — those are the supply path and the input signal respectively).
* ``PDNDroopRegressor`` — encoder + per-load-edge scalar head. The head
  reads the two endpoint hidden states for each load edge (the Vdd-side
  mesh_bot node and the Vss-side mesh_bot node), concatenates them, and
  maps to a single droop scalar per load.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv

from tools.grid_construction import NODE_FEATURE_DIM
from tools.sampler import DEFAULT_RANGES, ParamRanges

from .convs import _make_conv
from .normalizer import InputNormalizer
from .schema import EDGE_ATTR_DIM_NORMALIZED, EDGE_TYPES, NODE_TYPES


@dataclass
class EncoderConfig:
    hidden_dim: int = 64
    n_layers: int = 7
    dropout: float = 0.0
    conv_type: str = "admittance"  # "admittance" | "edge_aware"
    drop_edge_p: float = 0.0       # fraction of strap/decap edges dropped per layer (train only)


class PDNEncoder(nn.Module):
    """Heterogeneous GNN backbone over the two-rail PDN graph."""

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
        self.edge_proj = nn.ModuleDict(
            {self._et_key(et): nn.Linear(EDGE_ATTR_DIM_NORMALIZED, h) for et in EDGE_TYPES}
        )
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {et: _make_conv(et, h, cfg.conv_type) for et in EDGE_TYPES},
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

    # Relations DropEdge is allowed to touch. Via is the only supply path
    # between layers (dropping it cuts the model off from the pads); load
    # carries the input signal that defines the droop target. So both
    # are protected — DropEdge applies only to ``strap`` and ``decap``.
    _DROPPABLE_RELS = ("strap", "decap")

    # Relations whose conv consumes the raw normalized R column directly
    # (the conductance-kind conv). For these we skip ``edge_proj`` and
    # pass a [E, 1] scalar — the z-scored log10(R) — into the conv,
    # which folds it into ``exp(−α · z)`` deterministically.
    _CONDUCTANCE_RELS = ("strap", "via")

    def _build_edge_attr_dict(self, data: HeteroData) -> dict:
        out = {}
        for et in EDGE_TYPES:
            raw = data[et].edge_attr  # [E, EDGE_ATTR_DIM]
            normed = self.normalizer.normalize_edge_attr(raw, et)  # [E, 7]
            if self.cfg.conv_type == "admittance" and et[1] in self._CONDUCTANCE_RELS:
                # Hand-coded conductance gate: pass the R column only.
                out[et] = normed[:, 0:1]
            else:
                out[et] = self.edge_proj[self._et_key(et)](normed)
        return out

    def _drop_edges(
        self,
        edge_index_dict: dict,
        edge_attr_dict: dict,
    ) -> tuple[dict, dict]:
        """DropEdge sampler — fresh mask per call. Keeps ``via`` and
        ``load`` edges intact, masks ``strap`` and ``decap`` at rate
        ``cfg.drop_edge_p``.
        """
        p_keep = 1.0 - self.cfg.drop_edge_p
        new_ei = dict(edge_index_dict)
        new_ea = dict(edge_attr_dict)
        for et in EDGE_TYPES:
            if et[1] not in self._DROPPABLE_RELS:
                continue
            ei = edge_index_dict[et]
            E = ei.size(1)
            if E == 0:
                continue
            mask = torch.rand(E, device=ei.device) < p_keep
            new_ei[et] = ei[:, mask]
            new_ea[et] = edge_attr_dict[et][mask]
        return new_ei, new_ea

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {nt: self.node_proj[nt](data[nt].x) for nt in NODE_TYPES}
        edge_attr_dict = self._build_edge_attr_dict(data)
        edge_index_dict = dict(data.edge_index_dict)

        do_drop = self.training and self.cfg.drop_edge_p > 0
        for conv, norm in zip(self.convs, self.norms):
            ei_use, ea_use = (
                self._drop_edges(edge_index_dict, edge_attr_dict)
                if do_drop
                else (edge_index_dict, edge_attr_dict)
            )
            out = conv(x_dict, ei_use, ea_use)
            x_dict = {
                nt: norm[nt](self.dropout(F.relu(out[nt])) + x_dict[nt])
                for nt in NODE_TYPES
                if nt in out
            }
        return x_dict


class PDNDroopRegressor(nn.Module):
    """Encoder + per-load-edge scalar head.

    For each load edge ``k``, reads the encoder's hidden state at the
    Vdd-side mesh_bot endpoint and the Vss-side mesh_bot endpoint,
    concatenates, and predicts one droop scalar. Output shape is
    ``[total_load_edges_in_batch]``.

    ``target_space="log"`` predicts log10(droop); ``"linear"`` predicts
    droop directly. The model itself is identical — only the training
    loss / inverse transform changes.
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
        # Head reads (h_vdd, h_vss) for each load — two hidden states
        # concat → scalar droop.
        self.head = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        self.target_space = target_space

    def forward(self, data: HeteroData) -> torch.Tensor:
        # The load relation is directed Vdd→Vss; edge_index[0] is the
        # Vdd-side mesh_bot index, edge_index[1] the Vss-side. PyG
        # batches edge_index with per-graph offsets, so this works for
        # any batch size.
        x_bot = self.encoder(data)["mesh_bot"]
        ei = data["mesh_bot", "load", "mesh_bot"].edge_index   # [2, E]
        h_vdd = x_bot[ei[0]]
        h_vss = x_bot[ei[1]]
        return self.head(torch.cat([h_vdd, h_vss], dim=-1)).squeeze(-1)
