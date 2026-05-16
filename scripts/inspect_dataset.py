"""Print summary statistics for a v3 dataset HDF5.

Usage:
    python scripts/inspect_dataset.py datasets/regular_v2/dataset.h5

Shows per-section quantiles for global params, per-load params, and both
droop targets (peak / static). Useful for sanity-checking sample coverage
before training.
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


def _describe(grp: h5py.Group, label: str, global_names: list[str], per_load_names: list[str]) -> None:
    n = grp["global_params"].shape[0]
    print(f"\n-- {label}  N={n}")
    g = grp["global_params"][:]
    for j, name in enumerate(global_names):
        print(f"   global  {name:11s}  {_quantiles(g[:, j])}")
    load_x = grp["load_x"][:]                 # [N, K, 4]
    n_loads = grp["n_loads"][:]
    mask = (np.arange(load_x.shape[1])[None, :] < n_loads[:, None])  # [N, K]
    flat = load_x[mask]                       # [sum_loads, 4]
    # columns: I_peak, freq, duty, phase  (freq is constant per sample)
    for j, name in zip([0, 2, 3], per_load_names):
        print(f"   per-load {name:8s}  {_quantiles(flat[:, j])}")
    pad_idx = grp["pad_pattern_idx"][:]
    uniq, cnt = np.unique(pad_idx, return_counts=True)
    print(f"   pad_pattern_idx counts: {dict(zip(uniq.tolist(), cnt.tolist()))}")
    peak = grp["peak_droop_bot"][:]
    static = grp["static_droop_bot"][:]
    worst = grp["worst_node_droop"][:]
    print(f"   peak_droop_bot   (mV): {_quantiles(peak * 1e3)}")
    print(f"   static_droop_bot (mV): {_quantiles(static * 1e3)}")
    print(f"   worst_node_droop (mV): {_quantiles(worst * 1e3)}")
    print(f"   skew(peak linear)={_skew(peak):.2f}   skew(log10 peak)={_skew(np.log10(np.maximum(peak, 1e-7))):.2f}")
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
        pn = json.loads(f.attrs["per_load_param_names"])
        print(f"   global params: {gn}")
        print(f"   per-load params: {pn}")
        print(f"   pad patterns (train): {json.loads(f.attrs['train_pad_patterns'])}")
        print(f"   pad patterns (ood):   {json.loads(f.attrs['ood_pad_patterns'])}")
        print(f"   max_n_loads: {int(f.attrs['max_n_loads'])}")
        print(f"   sim_config: {f.attrs['sim_config']}")

        for split in ("train", "val", "test"):
            if split in f["bulk"]:
                _describe(f["bulk"][split], f"bulk/{split}", gn, pn)
        if "ood" in f:
            for pp in f["ood"]:
                _describe(f["ood"][pp], f"ood/{pp}", gn, pn)
        if "analysis" in f and "sweeps" in f["analysis"]:
            sw = f["analysis"]["sweeps"]
            axes = list(sw.keys())
            patterns_in_first = list(sw[axes[0]].keys()) if axes else []
            counts = {ax: {pp: int(sw[ax][pp]["global_params"].shape[0]) for pp in sw[ax]} for ax in axes}
            print(f"\n-- analysis/sweeps: {len(axes)} axes × {len(patterns_in_first)} patterns")
            print(f"   axes: {axes}")
            print(f"   counts per (axis, pattern): {counts[axes[0]] if axes else {}}")


if __name__ == "__main__":
    main()
