"""Turn an IBM transient benchmark into a labeled per-node graph dataset.

Pipeline: parse netlist → solve full transient (fine dt for accuracy) → build the
merged electrical graph (shorts collapsed) with per-node features + a per-node
**droop** label, and save as ``datasets/ibmpg/graphs/<bench>.npz``.

Droop label (deviation from the ideal rail, like the synthetic dataset):
* VDD node (V_dc > 0.9): ``droop = 1.8 − min_t V(t)``  (sag)
* GND node:              ``droop = max_t V(t) − 0``     (bounce)

Only real ``n<layer>`` grid nodes are scored (``is_grid`` mask); the ``_X_/_Y_/_Z_``
package/decap aux nodes are kept in the graph (faithful topology) but masked out.

Usage:
    python3.12 scripts/build_ibmpg_dataset.py --bench ibmpg1t --dt-divide 2
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.spice_parser import GROUND, parse_netlist
from tools.spice_solver import solve_transient
from scripts.validate_ibmpg import _unzip

VDD_NOMINAL = 1.8
NET_THRESH = 0.9
_LAYER_RE = re.compile(r"n(\d)_")


def _is_grid(name: str) -> bool:
    return len(name) >= 2 and name[0] == "n" and name[1].isdigit()


def _layer(name: str) -> int:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def build(bench: str, dt_divide: int) -> dict:
    spice = _unzip(bench, "spice")
    print(f"parsing {spice.name} ...")
    circ = parse_netlist(spice)
    print("  ", circ.summary())

    dt = (circ.tran_dt or 1e-11) / dt_divide
    print(f"solving full transient at dt={dt:.1e} (×{dt_divide} finer) ...")
    t0 = time.time()
    res = solve_transient(circ, dt=dt, track_extrema=True)
    print(f"  done in {time.time() - t0:.1f}s")

    uf, full_idx, ground_rep, clamp_val, M = res["nodemap"]
    v_dc, vmin, vmax = res["v_dc"], res["vmin"], res["vmax"]

    def ridx(name: str) -> int:
        r = uf.find(name)
        return -1 if r == ground_rep else full_idx[r]

    # ----- per-node (per-rep) features -----
    has_grid = np.zeros(M, dtype=bool)
    layer = np.full(M, -1, dtype=np.int8)
    all_names = set()
    for lst in (circ.resistors, circ.capacitors, circ.inductors, circ.vsources):
        for a, b, *_ in lst:
            all_names.add(a); all_names.add(b)
    for a, b, *_ in circ.isources:
        all_names.add(a); all_names.add(b)
    for name in all_names:
        i = ridx(name)
        if i < 0:
            continue
        if _is_grid(name):
            has_grid[i] = True
            layer[i] = _layer(name)
        elif layer[i] < 0:
            layer[i] = _layer(name)   # aux node carries its location's layer

    is_clamp = ~np.isnan(clamp_val)
    net_vdd = v_dc > NET_THRESH

    # loads → node peak current; caps-to-ground → node cap
    load_I = np.zeros(M)
    for a, b, dc, pulse in circ.isources:
        ia, ib = ridx(a), ridx(b)
        peak = max(abs(pulse.i1), abs(pulse.i2)) if pulse is not None else abs(dc)
        if ia >= 0:
            load_I[ia] += peak
        if ib >= 0:
            load_I[ib] += peak
    cap_gnd = np.zeros(M)

    # ----- edges (collapse parallels) -----
    R_par: dict[tuple[int, int], float] = defaultdict(float)   # accumulate conductance
    C_par: dict[tuple[int, int], float] = defaultdict(float)
    L_edges: list[tuple[int, int, float]] = []
    for a, b, R in circ.resistors:
        ia, ib = ridx(a), ridx(b)
        if ia == ib or ia < 0 and ib < 0:
            continue
        if ia < 0 or ib < 0:
            continue                      # resistor to ground = node-to-ref, skip edge
        R_par[(min(ia, ib), max(ia, ib))] += 1.0 / R
    for a, b, C in circ.capacitors:
        ia, ib = ridx(a), ridx(b)
        if ia >= 0 and ib >= 0 and ia != ib:
            C_par[(min(ia, ib), max(ia, ib))] += C
        else:                              # cap to ground → node feature
            j = ia if ia >= 0 else ib
            if j >= 0:
                cap_gnd[j] += C
    for a, b, L in circ.inductors:
        ia, ib = ridx(a), ridx(b)
        if ia >= 0 and ib >= 0 and ia != ib:
            L_edges.append((ia, ib, L))

    R_ei = np.array(list(R_par.keys()), dtype=np.int64).T if R_par else np.zeros((2, 0), np.int64)
    R_val = np.array([1.0 / g for g in R_par.values()], dtype=np.float32)
    C_ei = np.array(list(C_par.keys()), dtype=np.int64).T if C_par else np.zeros((2, 0), np.int64)
    C_val = np.array(list(C_par.values()), dtype=np.float32)
    L_ei = np.array([(a, b) for a, b, _ in L_edges], dtype=np.int64).T if L_edges else np.zeros((2, 0), np.int64)
    L_val = np.array([v for _, _, v in L_edges], dtype=np.float32)

    # ----- droop label -----
    nominal = np.where(net_vdd, VDD_NOMINAL, 0.0)
    droop = np.where(net_vdd, nominal - vmin, vmax - nominal).astype(np.float32)

    return {
        "bench": bench,
        # node features
        "layer": layer, "is_grid": has_grid, "is_clamp": is_clamp,
        "net_vdd": net_vdd, "load_I": load_I.astype(np.float32),
        "cap_gnd": cap_gnd.astype(np.float32),
        "v_dc": v_dc.astype(np.float32), "vmin": vmin.astype(np.float32),
        "vmax": vmax.astype(np.float32), "droop": droop,
        # edges
        "R_edge_index": R_ei, "R_value": R_val,
        "C_edge_index": C_ei, "C_value": C_val,
        "L_edge_index": L_ei, "L_value": L_val,
        "n_nodes": np.int64(M), "dt": np.float32(dt),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ibmpg1t")
    ap.add_argument("--dt-divide", type=int, default=2,
                    help="solve at native_dt / this (finer = more accurate label)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = build(args.bench, args.dt_divide)
    out = args.out or Path(f"datasets/ibmpg/graphs/{args.bench}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)

    g = d["is_grid"]
    vdd = g & d["net_vdd"]
    gnd = g & ~d["net_vdd"]
    dr = d["droop"]
    print(f"\n=== {args.bench} graph dataset ===")
    print(f"nodes {int(d['n_nodes'])} ({g.sum()} grid / {(~g).sum()} aux) | "
          f"R-edges {d['R_value'].size} C-edges {d['C_value'].size} L-edges {d['L_value'].size}")
    print(f"loads on {(d['load_I']>0).sum()} nodes | clamps {d['is_clamp'].sum()}")
    for tag, m in (("VDD grid", vdd), ("GND grid", gnd)):
        if m.sum():
            v = dr[m] * 1e3
            print(f"  {tag}: n={m.sum():6d}  droop median {np.median(v):.3f} mV  "
                  f"p99 {np.percentile(v,99):.3f}  max {v.max():.3f} mV")
    print(f"saved → {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
