"""PyG ``Dataset`` wrapper around the v3 dataset HDF5.

For every sample we look up the right ``HeteroData`` template by
``pad_pattern_idx`` (one template per pad pattern, cached at init),
clone, then fill in:

* edge_attr for ``R_top`` / ``R_bot`` / ``R_via`` / ``C_decap``, derived
  from the sample's global params (``R_top = Rsheet × pitch / width``);
* ``load.x`` from the per-load slice of ``load_x`` (first ``n_loads`` rows);
* ``mesh_bot.y`` from the chosen droop target (``peak`` or ``static``).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from .grid_construction import PAD_PATTERNS, build_regular_pdn, to_hetero_data


Target = Literal["linear", "log"]
DroopKind = Literal["peak", "static"]
LOG_FLOOR = 1e-7  # volts; clip to avoid log(0) at the corner pads


class RegularPDNDataset:
    """Loads one section of the v3 dataset HDF5.

    Args:
        h5_path: path to the dataset HDF5 file.
        split: ``"train" | "val" | "test"`` (under ``/bulk``), or
            ``"ood_<pattern>"`` (under ``/ood``), or a sweep selector
            ``"sweep:<axis>/<pattern>"`` (under ``/analysis/sweeps``).
        target: ``"linear"`` returns droop in volts; ``"log"`` returns
            ``log10(droop)``.
        droop_kind: ``"peak"`` (transient) or ``"static"`` (DC IR drop).
    """

    GLOBAL_KEYS = ("Rsheet_top", "Rsheet_bot", "wire_width", "R_via", "C_decap", "freq")

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
            self._global = grp["global_params"][:]            # [N, 6]
            self._pad_idx = grp["pad_pattern_idx"][:]         # [N] int8
            self._load_x = grp["load_x"][:]                   # [N, max_n_loads, 4]
            self._n_loads = grp["n_loads"][:]                 # [N] int16
            key = "peak_droop_bot" if droop_kind == "peak" else "static_droop_bot"
            self._target_y = grp[key][:]                      # [N, n_bot^2]

        # One template per pad pattern (cheap — built once at init).
        self._templates = {
            i: to_hetero_data(build_regular_pdn(pad_pattern=pp))
            for i, pp in enumerate(PAD_PATTERNS)
        }
        proto = build_regular_pdn()
        self._pitch_top = proto.pitch_top
        self._pitch_bot = proto.pitch_bot

    # ----- group resolution -------------------------------------------------

    @staticmethod
    def _resolve_group(f: h5py.File, split: str) -> h5py.Group:
        if split in ("train", "val", "test"):
            return f["bulk"][split]
        if split.startswith("ood_"):
            return f["ood"][split[4:]]
        if split.startswith("sweep:"):
            axis_pattern = split[len("sweep:") :]
            axis, pattern = axis_pattern.split("/")
            return f["analysis"]["sweeps"][axis][pattern]
        raise KeyError(
            f"unknown split {split!r}; expected train/val/test, ood_<pattern>, "
            f"or sweep:<axis>/<pattern>"
        )

    # ----- dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return self._global.shape[0]

    def __getitem__(self, idx: int):
        import torch

        pp_idx = int(self._pad_idx[idx])
        data = deepcopy(self._templates[pp_idx])
        g = self._global[idx]
        Rsheet_top = float(g[0]); Rsheet_bot = float(g[1]); wire_width = float(g[2])
        R_via = float(g[3]); C_decap = float(g[4])

        R_top = Rsheet_top * (self._pitch_top / wire_width)
        R_bot = Rsheet_bot * (self._pitch_bot / wire_width)

        data["mesh_top", "R_top", "mesh_top"].edge_attr.fill_(R_top)
        data["mesh_bot", "R_bot", "mesh_bot"].edge_attr.fill_(R_bot)
        data["mesh_top", "R_via", "mesh_bot"].edge_attr.fill_(R_via)
        data["mesh_bot", "R_via", "mesh_top"].edge_attr.fill_(R_via)
        data["mesh_bot", "C_decap", "gnd"].edge_attr.fill_(C_decap)
        data["gnd", "C_decap_rev", "mesh_bot"].edge_attr.fill_(C_decap)

        n_loads = int(self._n_loads[idx])
        data["load"].x = torch.from_numpy(self._load_x[idx, :n_loads, :].astype(np.float32))

        droop = self._target_y[idx]
        if self.target == "log":
            y = np.log10(np.maximum(droop, LOG_FLOOR))
        else:
            y = droop
        data["mesh_bot"].y = torch.from_numpy(y.astype(np.float32))

        return data
