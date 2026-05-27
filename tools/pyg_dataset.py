"""PyG ``Dataset`` wrapper around the v4 dataset HDF5.

Each sample looks up a per-``n_top`` ``HeteroData`` template, clones it,
then fills in the per-sample edge-attribute columns:

* strap edge ``R`` (top, bot) — derived from the fixed ``Rsheet_*`` and
  the sample's ``wire_width``;
* via edge ``R`` — fixed (``R_via``);
* decap edge ``C`` — the sample's ``C_decap``;
* load edge ``(I_peak, freq, duty, phase)`` — all fixed; broadcast from
  the sampler constants (the HDF5 stores them once as a root attribute,
  not per-sample, since they are invariant by design);
* ``mesh_bot.y`` — the chosen droop target.

Only two continuous quantities actually vary across samples
(``wire_width`` and ``C_decap``); ``n_top`` selects which template to
clone. Everything else is constant by construction.
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
    ALL_N_TOP,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PAD_PATTERN,
    FIXED_PHASE,
    FIXED_R_VIA,
    FIXED_RSHEET_BOT,
    FIXED_RSHEET_TOP,
)


Target = Literal["linear", "log"]
DroopKind = Literal["peak", "static"]
LOG_FLOOR = 1e-7  # volts; clip to avoid log(0) at the corner pads


class RegularPDNDataset:
    """Loads one section of the v4 dataset HDF5.

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
    ) -> None:
        self.h5_path = Path(h5_path)
        self.split = split
        self.target = target
        self.droop_kind = droop_kind

        with h5py.File(self.h5_path, "r") as f:
            grp = self._resolve_group(f, split)
            self._global = grp["global_params"][:]            # [N, 2]
            self._n_top  = grp["n_top"][:]                    # [N] int16
            key = "peak_droop_bot" if droop_kind == "peak" else "static_droop_bot"
            self._target_y = grp[key][:]                      # [N, n_bot^2]

        # Per-n_top template + pitch lookup, computed once.
        self._templates: dict[int, object] = {}
        self._pitch_by_n_top: dict[int, tuple[float, float]] = {}
        for nt in ALL_N_TOP:
            g = build_regular_pdn(n_top=int(nt), pad_pattern=FIXED_PAD_PATTERN)
            self._templates[int(nt)] = to_hetero_data(g)
            self._pitch_by_n_top[int(nt)] = (g.pitch_top, g.pitch_bot)
        self._n_loads = int(g.n_loads)  # n_loads is invariant across n_top here

        # Column positions in the global_params row.
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

    # Column indices into the 6-dim edge attribute, cached for speed.
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

        n_top = int(self._n_top[idx])
        data = deepcopy(self._templates[n_top])
        pitch_top, pitch_bot = self._pitch_by_n_top[n_top]

        wire_width = float(self._global[idx, self._ww_col])
        C_decap    = float(self._global[idx, self._cd_col])

        # Strap and via R columns: Rsheet_* fixed, wire_width varies.
        R_top = FIXED_RSHEET_TOP * (pitch_top / wire_width)
        R_bot = FIXED_RSHEET_BOT * (pitch_bot / wire_width)
        data["mesh_top", "strap", "mesh_top"].edge_attr[:, self._R_COL] = R_top
        data["mesh_bot", "strap", "mesh_bot"].edge_attr[:, self._R_COL] = R_bot
        data["mesh_top", "via", "mesh_bot"].edge_attr[:, self._R_COL] = FIXED_R_VIA
        data["mesh_bot", "via", "mesh_top"].edge_attr[:, self._R_COL] = FIXED_R_VIA

        # Decap C column.
        data["mesh_bot", "decap", "gnd"].edge_attr[:, self._C_COL] = C_decap
        data["gnd", "decap", "mesh_bot"].edge_attr[:, self._C_COL] = C_decap

        # Load (I/freq/duty/phase) — all fixed; built from sampler constants.
        load_attr = self._build_load_attr(self._n_loads)
        data["mesh_bot", "load", "gnd"].edge_attr = load_attr
        data["gnd", "load", "mesh_bot"].edge_attr = load_attr.clone()

        droop = self._target_y[idx]
        if self.target == "log":
            y = np.log10(np.maximum(droop, LOG_FLOOR))
        else:
            y = droop
        data["mesh_bot"].y = torch.from_numpy(y.astype(np.float32))

        return data
