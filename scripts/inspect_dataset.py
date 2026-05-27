"""Print summary statistics for a v4 dataset HDF5.

Usage:
    python scripts/inspect_dataset.py datasets/regular_v4/dataset.h5

Shows per-section quantiles for the global continuous knobs, n_top
counts, and droop targets (peak / static). Useful for sanity-checking
sample coverage before training.
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
    m, s = x.mean(), x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _describe(grp: h5py.Group, label: str, global_names: list[str]) -> None:
    n = grp["global_params"].shape[0]
    print(f"\n-- {label}  N={n}")
    g = grp["global_params"][:]
    for j, name in enumerate(global_names):
        print(f"   global  {name:11s}  {_quantiles(g[:, j])}")
    if "n_top" in grp:
        n_top = grp["n_top"][:]
        uniq, cnt = np.unique(n_top, return_counts=True)
        print(f"   n_top counts: {dict(zip(uniq.tolist(), cnt.tolist()))}")
    peak = grp["peak_droop_bot"][:]
    static = grp["static_droop_bot"][:]
    worst = grp["worst_node_droop"][:]
    print(f"   peak_droop_bot   (mV): {_quantiles(peak * 1e3)}")
    print(f"   static_droop_bot (mV): {_quantiles(static * 1e3)}")
    print(f"   worst_node_droop (mV): {_quantiles(worst * 1e3)}")
    print(f"   skew(peak linear)={_skew(peak):.2f}   "
          f"skew(log10 peak)={_skew(np.log10(np.maximum(peak, 1e-7))):.2f}")
    if "V_subset" in grp:
        V = grp["V_subset"]["V_bot"]
        print(f"   V_subset: {V.shape[0]} samples × {V.shape[1]} steps × {V.shape[2]} bot nodes")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: inspect_dataset.py <dataset.h5>")
        sys.exit(1)
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as f:
        print(f"== {path.name}  (v{int(f.attrs['version'])}, "
              f"created {f.attrs['created_at']}, git {str(f.attrs['git_sha'])[:8]})")
        gn = json.loads(f.attrs["global_param_names"])
        print(f"   global params: {gn}")
        print(f"   train n_top:   {json.loads(f.attrs['train_n_top'])}")
        print(f"   ood n_top:     {json.loads(f.attrs['ood_n_top'])}")
        print(f"   n_loads:       {int(f.attrs['n_loads'])}")
        print(f"   fixed_constants: {f.attrs['fixed_constants']}")
        print(f"   sim_config:    {f.attrs['sim_config']}")

        for split in ("train", "val", "test"):
            if split in f["bulk"]:
                _describe(f["bulk"][split], f"bulk/{split}", gn)
        if "ood" in f:
            for key in f["ood"]:
                _describe(f["ood"][key], f"ood/{key}", gn)
        if "analysis" in f and "sweeps" in f["analysis"]:
            sw = f["analysis"]["sweeps"]
            axes = list(sw.keys())
            n_top_keys_first = list(sw[axes[0]].keys()) if axes else []
            counts = {
                ax: {k: int(sw[ax][k]["global_params"].shape[0]) for k in sw[ax]}
                for ax in axes
            }
            print(f"\n-- analysis/sweeps: {len(axes)} axes × {len(n_top_keys_first)} n_top")
            print(f"   axes: {axes}")
            print(f"   counts per (axis, n_top): {counts[axes[0]] if axes else {}}")


if __name__ == "__main__":
    main()
