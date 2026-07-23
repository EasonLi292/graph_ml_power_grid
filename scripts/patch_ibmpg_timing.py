"""Backfill load-timing data into the IBM npz graphs (stage 1, F5/F6).

The benchmark loads are ~10 ps PULSE sources whose delays span >1 ns with
mixed 2/3 ns periods; the npz previously stored only the per-node *peak*
current, so which loads fire together — the dominant driver of within-net
droop ranking — was never encoded. This adds, per bench:

- ``tqs_peak [n]``: exact quasi-static timing peak ``max_t |Z_dc I(t)|``
  over one 6 ns hyperperiod (240 solves off one LU factorization). Ranks
  held-out pg2t at 0.846/0.793 within-net vs the static baseline's
  0.437/0.397 — the new residual floor and reported baseline.
- ``wave_node [k]`` / ``wave_bins [k, 24]``: signed per-node injected
  current waveform, max-|.| pooled into 250 ps bins over the hyperperiod
  (sparse: only nodes with sources). Local timing features for the convs
  and time-structured attention values.
- ``wave_bin_dt``, ``tqs_dt``: bin width / solve step metadata.

Usage: python scripts/patch_ibmpg_timing.py [bench ...]   (default: all 6)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.spice_parser import parse_netlist
from tools.spice_solver import _build_node_map
from scripts.validate_ibmpg import _unzip

GRAPHS = Path("datasets/ibmpg/graphs")
ALL = [f"ibmpg{i}t" for i in (1, 2, 3, 4, 5, 6)]

L_SHORT_G = 1e6      # siemens; package inductors are DC shorts
DT = 25e-12          # quasi-static solve step
T_END = 6e-9         # LCM of the 2 ns / 3 ns pulse periods (all benches)
N_BINS = 24          # feature bins (250 ps each)


def _grounded_lu(d, idx, active):
    n_act = int(active.sum())
    R_ei, R_val = d["R_edge_index"], d["R_value"].astype(np.float64)
    L_ei = d["L_edge_index"]
    a = np.concatenate([R_ei[0], L_ei[0]])
    b = np.concatenate([R_ei[1], L_ei[1]])
    w = np.concatenate([1.0 / np.maximum(R_val, 1e-9),
                        np.full(L_ei.shape[1], L_SHORT_G)])
    ia, ib = idx[a], idx[b]
    rows, cols, vals = [], [], []
    both = (ia >= 0) & (ib >= 0)
    rows += [ia[both], ib[both], ia[both], ib[both]]
    cols += [ib[both], ia[both], ia[both], ib[both]]
    vals += [-w[both], -w[both], w[both], w[both]]
    for live, dead in ((ia, ib), (ib, ia)):
        m_ = (live >= 0) & (dead < 0)
        rows.append(live[m_]); cols.append(live[m_]); vals.append(w[m_])
    gn, gRv = d["Rg_node"], d["Rg_value"].astype(np.float64)
    ig = idx[gn]; mg = ig >= 0
    rows.append(ig[mg]); cols.append(ig[mg])
    vals.append((1.0 / np.maximum(gRv, 1e-9))[mg])
    ln = d["Lg_node"]; il = idx[ln]; ml = il >= 0
    rows.append(il[ml]); cols.append(il[ml])
    vals.append(np.full(int(ml.sum()), L_SHORT_G))
    Lg = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_act, n_act),
    ).tocsc()
    return splu(Lg)


def patch(bench: str, force: bool = False) -> None:
    out = GRAPHS / f"{bench}.npz"
    d = dict(np.load(out, allow_pickle=True))
    if "tqs_peak" in d and not force:
        print(f"{bench}: already patched")
        return
    t0 = time.time()
    n = int(d["n_nodes"])
    clamp = d["is_clamp"].astype(bool)
    active = ~clamp
    idx = np.full(n, -1, dtype=np.int64)
    idx[active] = np.arange(int(active.sum()))
    n_act = int(active.sum())

    circ = parse_netlist(_unzip(bench, "spice"))
    uf, full_idx, M, clamp_val, ground_rep = _build_node_map(circ)
    assert M == n, f"node map mismatch: {M} != {n}"

    def ridx(name: str) -> int:
        r = uf.find(name)
        return -1 if r == ground_rep else full_idx[r]

    # signed per-active-node injected current over the hyperperiod
    ts = np.arange(0.0, T_END, DT)
    T = ts.size
    I = np.zeros((n_act, T))
    for sa, sb, dc, pulse in circ.isources:
        wave = pulse.at(ts) if pulse is not None else np.full(T, dc)
        for nm, sgn in ((sa, -1.0), (sb, +1.0)):
            k = ridx(nm)
            if k >= 0 and idx[k] >= 0:
                I[idx[k]] += sgn * wave

    lu = _grounded_lu(d, idx, active)
    peak = np.zeros(n_act)
    for s in range(0, T, 60):
        V = lu.solve(I[:, s:s + 60])
        peak = np.maximum(peak, np.abs(V).max(axis=1))
    tqs = np.zeros(n, dtype=np.float32)
    tqs[active] = peak.astype(np.float32)

    # sparse binned waveform features: signed max-|.| per 250 ps bin
    nz = np.flatnonzero(np.abs(I).max(axis=1) > 0)
    per_bin = T // N_BINS
    seg = I[nz, : per_bin * N_BINS].reshape(nz.size, N_BINS, per_bin)
    am = np.abs(seg).argmax(axis=2)
    bins = np.take_along_axis(seg, am[:, :, None], axis=2)[:, :, 0]
    act_ids = np.flatnonzero(active)
    d["tqs_peak"] = tqs
    d["wave_node"] = act_ids[nz].astype(np.int64)
    d["wave_bins"] = bins.astype(np.float32)
    d["wave_bin_dt"] = np.float64(T_END / N_BINS)
    d["tqs_dt"] = np.float64(DT)
    np.savez_compressed(out, **d)
    print(f"{bench}: tqs_peak med {np.median(tqs[active])*1e3:.1f} mV, "
          f"{nz.size} wave nodes x {N_BINS} bins in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    force = "--force" in sys.argv
    benches = [a for a in sys.argv[1:] if not a.startswith("-")] or ALL
    for b in benches:
        patch(b, force=force)
