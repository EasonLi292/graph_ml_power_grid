"""PyG ``Dataset`` wrapper around the dataset HDF5.

Per-sample workflow: look up a ``HeteroData`` template by ``n_top``,
clone it, fill in the edge attribute columns that vary, and attach the
target. With the new design only two continuous knobs actually vary
(``wire_width`` and ``C_decap``); the rest comes from the sampler
constants stored once at the HDF5 root.

Targets
-------
``y`` is a per-load-site supply droop vector of length ``n_loads`` (12 by
default at ``n_bot=7``), not the old per-bot-node vector. Each entry is
``Vdd − (V_bot_vdd[k] − V_bot_vss[k])`` for load ``k``. It lands on the
``HeteroData`` as ``data["y"]`` (a global tensor, not pinned to any
node type) since it is one scalar per load *edge*, not per node.

The (Vdd, Vss) endpoint indices for the readout are also attached as
``data["load_endpoints"]`` and ``data["decap_endpoints"]`` by
``to_hetero_data`` upstream.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from .grid_construction import (
    EDGE_ATTR_COLS,
    EDGE_ATTR_DIM,
    build_regular_pdn,
    to_hetero_data,
)
from .sampler import (
    ALL_ANCHORS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    FIXED_R_VIA,
    FIXED_RSHEET_BOT,
    FIXED_RSHEET_TOP,
)


Target = Literal["linear", "log"]
DroopKind = Literal["peak", "static"]
LOG_FLOOR = 1e-7  # volts; clip to avoid log(0) at near-zero droop sites


class RegularPDNDataset:
    """Loads one section of the dataset HDF5.

    Args:
        h5_path: path to the dataset HDF5 file.
        split: ``"train" | "val" | "test"`` (under ``/bulk``), or
            ``"ood_n_top_<N>"`` (under ``/ood``), or
            ``"sweep:<axis>/n_top_<N>"`` (under ``/analysis/sweeps``).
        target: ``"linear"`` returns droop in volts; ``"log"`` returns
            ``log10(droop)``.
        droop_kind: ``"peak"`` (transient) or ``"static"`` (DC IR drop).
    """

    GLOBAL_KEYS = ("wire_width", "C_decap")

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        target: Target = "linear",
        droop_kind: DroopKind = "peak",
        jac_path: str | Path | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.split = split
        self.target = target
        self.droop_kind = droop_kind

        # Optional exact-Jacobian labels (scripts/gen_jacobian_labels.py)
        # for Sobolev training: ∂(worst droop)/∂(ln ww_e), ∂/∂(ln C_site).
        self._jac = None
        if jac_path is not None:
            with h5py.File(jac_path, "r") as jf:
                if split in jf and "done" in jf[split]:
                    g = jf[split]
                    self._jac = {
                        "top": g["jac_lnww_top"][:],
                        "bot": g["jac_lnww_bot"][:],
                        "dec": g["jac_lnC"][:],
                        "done": g["done"][:],
                    }

        with h5py.File(self.h5_path, "r") as f:
            grp = self._resolve_group(f, split)
            self._global = grp["global_params"][:]            # [N, 2]
            self._n_top  = grp["n_top"][:]                    # [N] int16
            # Older (single-die-size) files have no n_bot column — all 7.
            self._n_bot = (
                grp["n_bot"][:] if "n_bot" in grp
                else np.full_like(self._n_top, 7)
            )
            key = "peak_droop_loads" if droop_kind == "peak" else "static_droop_loads"
            self._target_y = grp[key][:]                      # [N, MAX_LOADS] nan-padded
            # Per-edge (heterogeneous) wire widths, if this dataset has them.
            self._per_edge = "ww_top_edges" in grp
            if self._per_edge:
                self._ww_top_edges = grp["ww_top_edges"][:]   # [N, MAX_TOP] nan-padded
                self._ww_bot_edges = grp["ww_bot_edges"][:]   # [N, MAX_BOT] nan-padded

        # Per-anchor template + pitch + strap-edge/load-count lookup.
        self._templates: dict[tuple[int, int], object] = {}
        self._pitch_by_anchor: dict[tuple[int, int], tuple[float, float]] = {}
        self._n_strap_by_anchor: dict[tuple[int, int], tuple[int, int]] = {}
        self._n_loads_by_anchor: dict[tuple[int, int], int] = {}
        for nt, nb in ALL_ANCHORS:
            g = build_regular_pdn(n_top=int(nt), n_bot=int(nb))
            a = (int(nt), int(nb))
            self._templates[a] = to_hetero_data(g)
            self._pitch_by_anchor[a] = (g.pitch_top, g.pitch_bot)
            self._n_strap_by_anchor[a] = (
                int(g.top_edges.shape[0]), int(g.bot_edges.shape[0])
            )
            self._n_loads_by_anchor[a] = int(g.n_loads)

        self._ww_col = list(self.GLOBAL_KEYS).index("wire_width")
        self._cd_col = list(self.GLOBAL_KEYS).index("C_decap")

    # ----- group resolution -------------------------------------------------

    @staticmethod
    def _resolve_group(f: h5py.File, split: str) -> h5py.Group:
        if split in ("train", "val", "test"):
            return f["bulk"][split]
        if split.startswith("ood_n_top_"):
            return f["ood"][split[4:]]
        if split.startswith("sweep:"):
            axis_pattern = split[len("sweep:"):]
            axis, n_top_key = axis_pattern.split("/")
            return f["analysis"]["sweeps"][axis][n_top_key]
        raise KeyError(
            f"unknown split {split!r}; expected train/val/test, "
            f"ood_n_top_<N>, or sweep:<axis>/n_top_<N>"
        )

    # ----- dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return self._global.shape[0]

    _R_COL = EDGE_ATTR_COLS.index("R")
    _C_COL = EDGE_ATTR_COLS.index("C")
    _I_COL = EDGE_ATTR_COLS.index("I_peak")
    _F_COL = EDGE_ATTR_COLS.index("freq")
    _D_COL = EDGE_ATTR_COLS.index("duty")
    _P_COL = EDGE_ATTR_COLS.index("phase")

    def _build_load_attr(self, n_loads: int):
        import torch

        a = np.zeros((n_loads, EDGE_ATTR_DIM), dtype=np.float32)
        a[:, self._I_COL] = FIXED_I_PEAK
        a[:, self._F_COL] = FIXED_FREQ
        a[:, self._D_COL] = FIXED_DUTY
        a[:, self._P_COL] = FIXED_PHASE
        return torch.from_numpy(a)

    def __getitem__(self, idx: int):
        import torch

        anchor = (int(self._n_top[idx]), int(self._n_bot[idx]))
        data = deepcopy(self._templates[anchor])
        pitch_top, pitch_bot = self._pitch_by_anchor[anchor]
        n_loads = self._n_loads_by_anchor[anchor]

        wire_width = float(self._global[idx, self._ww_col])
        C_decap    = float(self._global[idx, self._cd_col])

        if self._per_edge:
            # Heterogeneous per-edge R = Rsheet × pitch / width_edge, bidir-
            # tiled to match the template's [u→v ; v→u] strap packing.
            n_t, n_b = self._n_strap_by_anchor[anchor]
            wt = self._ww_top_edges[idx, :n_t].astype(np.float64)
            wb = self._ww_bot_edges[idx, :n_b].astype(np.float64)
            R_top_e = (FIXED_RSHEET_TOP * (pitch_top / wt)).astype(np.float32)
            R_bot_e = (FIXED_RSHEET_BOT * (pitch_bot / wb)).astype(np.float32)
            data["mesh_top", "strap", "mesh_top"].edge_attr[:, self._R_COL] = \
                torch.from_numpy(np.concatenate([R_top_e, R_top_e]))
            data["mesh_bot", "strap", "mesh_bot"].edge_attr[:, self._R_COL] = \
                torch.from_numpy(np.concatenate([R_bot_e, R_bot_e]))
        else:
            # Uniform strap R: Rsheet × pitch / wire_width.
            R_top = FIXED_RSHEET_TOP * (pitch_top / wire_width)
            R_bot = FIXED_RSHEET_BOT * (pitch_bot / wire_width)
            data["mesh_top", "strap", "mesh_top"].edge_attr[:, self._R_COL] = R_top
            data["mesh_bot", "strap", "mesh_bot"].edge_attr[:, self._R_COL] = R_bot
        # Via R: fixed.
        data["mesh_top", "via", "mesh_bot"].edge_attr[:, self._R_COL] = FIXED_R_VIA
        data["mesh_bot", "via", "mesh_top"].edge_attr[:, self._R_COL] = FIXED_R_VIA
        # Decap C (single mesh_bot-internal relation, bidir packed).
        data["mesh_bot", "decap", "mesh_bot"].edge_attr[:, self._C_COL] = C_decap
        # Load attr — directed (Vdd→Vss) load relation, one row per load.
        data["mesh_bot", "load", "mesh_bot"].edge_attr = self._build_load_attr(n_loads)

        droop = self._target_y[idx, :n_loads]   # strip the NaN pad
        if self.target == "log":
            y = np.log10(np.maximum(droop, LOG_FLOOR))
        else:
            y = droop
        data["y"] = torch.from_numpy(y.astype(np.float32))

        # Sobolev labels: per-physical-edge Jacobians + segment sizes so the
        # loss can fold the batched bidir edge rows back to physical edges.
        if self._jac is not None and bool(self._jac["done"][idx]):
            n_t, n_b = self._n_strap_by_anchor[anchor]
            n_d = data["mesh_bot", "decap", "mesh_bot"].edge_index.size(1) // 2
            data["jac_top"] = torch.from_numpy(
                self._jac["top"][idx, :n_t].astype(np.float32))
            data["jac_bot"] = torch.from_numpy(
                self._jac["bot"][idx, :n_b].astype(np.float32))
            data["jac_dec"] = torch.from_numpy(
                self._jac["dec"][idx, :n_d].astype(np.float32))
            data["jac_seg"] = torch.tensor([n_t, n_b, n_d], dtype=torch.long)
            data["has_jac"] = torch.tensor([True])
        else:
            data["jac_top"] = torch.zeros(0)
            data["jac_bot"] = torch.zeros(0)
            data["jac_dec"] = torch.zeros(0)
            data["jac_seg"] = torch.zeros(3, dtype=torch.long)
            data["has_jac"] = torch.tensor([False])

        return data
