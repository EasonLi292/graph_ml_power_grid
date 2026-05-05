"""Print summary statistics for a dataset HDF5 and dump per-split histograms.

Usage:
    python scripts/inspect_dataset.py datasets/regular_v1/dataset.h5

Helps decide whether ``peak_droop`` should be predicted in linear or log space:
look at the printed quantiles and the linear/log skewness — if the linear
distribution is heavy-tailed (skew > 2), predicting in log10 space tends to
train more cleanly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


def _quantiles(x: np.ndarray, qs=(0, 0.05, 0.5, 0.95, 1.0)) -> str:
    qv = np.quantile(x, qs)
    return "  ".join(f"q{int(q*100)}={v:.3g}" for q, v in zip(qs, qv))


def _skew(x: np.ndarray) -> float:
    x = x.ravel()
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: inspect_dataset.py <dataset.h5>")
        sys.exit(1)
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as f:
        print(f"== {path.name}  (created {f.attrs['created_at']}, git {str(f.attrs['git_sha'])[:8]})")
        print(f"   topology: {f.attrs['topology']}")
        print(f"   sim_config: {f.attrs['sim_config']}")
        param_names = json.loads(f.attrs["param_names"])

        for split in f.keys():
            grp = f[split]
            params = grp["params"][:]
            droop = grp["peak_droop_bot"][:]
            worst = grp["worst_node_droop"][:]

            print(f"\n-- split: {split}  N={params.shape[0]}")
            for j, name in enumerate(param_names):
                print(f"   {name:8s}  {_quantiles(params[:, j])}")

            print(f"   droop_bot (mV, all nodes):  {_quantiles(droop * 1e3)}")
            print(f"   worst_node_droop (mV):       {_quantiles(worst * 1e3)}")
            print(f"   skew(droop_linear)= {_skew(droop):.2f}   "
                  f"skew(log10 droop)= {_skew(np.log10(np.maximum(droop, 1e-7))):.2f}")

            if "V_subset" in grp:
                sub = grp["V_subset"]
                V_bot = sub["V_bot"][:]
                print(f"   V_subset: {V_bot.shape[0]} samples × {V_bot.shape[1]} steps × {V_bot.shape[2]} bot nodes")


if __name__ == "__main__":
    main()
