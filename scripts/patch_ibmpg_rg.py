"""Backfill node-to-ground resistors into the IBM per-node npz graphs.

``build_ibmpg_dataset.py`` skipped resistors with one terminal on the
merged ground cluster ("resistor to ground = node-to-ref, skip edge").
They are irrelevant to node-to-node message passing but essential for
any grounded-system computation (impedance sketches): without them the
whole GND net floats.

This re-parses the netlist, reproduces the solver's node map
(deterministic), extracts ``(node, conductance-to-ground)``, verifies
the node count matches the existing npz, and appends ``Rg_node`` /
``Rg_value`` arrays in place. No re-simulation.

Usage:
    python3.12 scripts/patch_ibmpg_rg.py            # all 6 benches
    python3.12 scripts/patch_ibmpg_rg.py ibmpg1t
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.spice_parser import parse_netlist
from tools.spice_solver import _build_node_map
from scripts.validate_ibmpg import _unzip

GRAPHS = Path("datasets/ibmpg/graphs")
ALL = ["ibmpg1t", "ibmpg2t", "ibmpg3t", "ibmpg4t", "ibmpg5t", "ibmpg6t"]


def patch(bench: str) -> None:
    out = GRAPHS / f"{bench}.npz"
    d = dict(np.load(out))
    if "Rg_node" in d and "Lg_node" in d:
        print(f"{bench}: already patched ({d['Rg_value'].size} R-gnd, "
              f"{d['Lg_value'].size} L-gnd)")
        return
    t0 = time.time()
    circ = parse_netlist(_unzip(bench, "spice"))
    uf, full_idx, M, clamp_val, ground_rep = _build_node_map(circ)
    assert M == int(d["n_nodes"]), f"node map mismatch: {M} != {int(d['n_nodes'])}"

    def ridx(name: str) -> int:
        r = uf.find(name)
        return -1 if r == ground_rep else full_idx[r]

    g_gnd: dict[int, float] = defaultdict(float)
    for a, b, R in circ.resistors:
        ia, ib = ridx(a), ridx(b)
        if (ia < 0) != (ib < 0):                # exactly one terminal on ground
            g_gnd[ia if ia >= 0 else ib] += 1.0 / R
    nodes = np.array(sorted(g_gnd), dtype=np.int64)
    vals = np.array([1.0 / g_gnd[i] for i in nodes], dtype=np.float32)  # back to ohms
    d["Rg_node"] = nodes
    d["Rg_value"] = vals

    # package inductors to ground: DC shorts to the reference — the GND
    # net's only tie without them.
    l_gnd: dict[int, float] = defaultdict(float)
    for a, b, L in circ.inductors:
        ia, ib = ridx(a), ridx(b)
        if (ia < 0) != (ib < 0):
            i = ia if ia >= 0 else ib
            l_gnd[i] = l_gnd[i] + L if i in l_gnd else L
    lnodes = np.array(sorted(l_gnd), dtype=np.int64)
    lvals = np.array([l_gnd[i] for i in lnodes], dtype=np.float32)
    d["Lg_node"] = lnodes
    d["Lg_value"] = lvals
    np.savez_compressed(out, **d)
    print(f"{bench}: +{nodes.size} R-gnd (median {np.median(vals):.3f} ohm), "
          f"+{lnodes.size} L-gnd in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    for b in (sys.argv[1:] or ALL):
        patch(b)
