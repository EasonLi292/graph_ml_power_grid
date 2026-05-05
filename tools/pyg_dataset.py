"""PyG ``Dataset`` wrapper around the H5 dataset.

Topology is identical for every sample, so we keep a single canonical
``HeteroData`` template and overwrite per-sample edge_attr / load.x at
``__getitem__`` time. Switch ``target`` between ``"linear"`` and ``"log"``
when constructing the dataset (decision deferred to inspection time).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from .grid_construction import build_regular_pdn, to_hetero_data


Target = Literal["linear", "log"]
LOG_FLOOR = 1e-7  # volts; clip to avoid log(0) at the corner pads


class RegularPDNDataset:
    """Loads a single split (train/val/test) from a dataset H5 file.

    Returns ``torch_geometric.data.HeteroData`` per ``__getitem__``.
    Compatible with PyG's ``DataLoader``.
    """

    PARAM_KEYS = ("R_top", "R_bot", "R_via", "C_decap", "I_peak", "freq", "duty")

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        target: Target = "linear",
    ) -> None:
        self.h5_path = Path(h5_path)
        self.split = split
        self.target = target

        with h5py.File(self.h5_path, "r") as f:
            g = f[split]
            self._params = g["params"][:]  # [N, 7] in raw units
            self._droop_bot = g["peak_droop_bot"][:]  # [N, 49]

        self._template = to_hetero_data(build_regular_pdn())

    def __len__(self) -> int:
        return self._params.shape[0]

    def __getitem__(self, idx: int):
        import torch

        data = deepcopy(self._template)
        p = self._params[idx]
        R_top, R_bot, R_via, C_decap, I_peak, freq, duty = (float(x) for x in p)

        data["mesh_top", "R_top", "mesh_top"].edge_attr.fill_(R_top)
        data["mesh_bot", "R_bot", "mesh_bot"].edge_attr.fill_(R_bot)
        data["mesh_top", "R_via", "mesh_bot"].edge_attr.fill_(R_via)
        data["mesh_bot", "R_via", "mesh_top"].edge_attr.fill_(R_via)
        data["mesh_bot", "C_decap", "gnd"].edge_attr.fill_(C_decap)
        data["gnd", "C_decap_rev", "mesh_bot"].edge_attr.fill_(C_decap)

        n_loads = data["load"].x.shape[0]
        data["load"].x = torch.tensor(
            [[I_peak, freq, duty, 0.0]] * n_loads, dtype=torch.float32
        )

        droop = self._droop_bot[idx]
        if self.target == "log":
            y = np.log10(np.maximum(droop, LOG_FLOOR))
        else:
            y = droop
        data["mesh_bot"].y = torch.from_numpy(y.astype(np.float32))

        return data
