"""Quasi-static + timing probe: v_i(t) = Z_dc @ I(t), peak over t.

Same DC geometry the stage-0 sketch encodes, but with the true per-load
PULSE waveforms (delay/width/period from the netlist) instead of a
static peak. If this ranks droop much better than the static baseline,
the missing signal was load *timing*, not geometry.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.stats import spearmanr

from tools.spice_parser import parse_netlist
from tools.spice_solver import _build_node_map
from scripts.validate_ibmpg import _unzip

L_SHORT_G = 1e6
DT = 25e-12
T_END = 6e-9   # LCM of 2ns/3ns pulse periods

def run(bench):
    t0 = time.time()
    d = np.load(f'datasets/ibmpg/graphs/{bench}.npz', allow_pickle=True)
    n = int(d['n_nodes'])
    clamp = d['is_clamp'].astype(bool)
    grid = d['is_grid'].astype(bool) & ~clamp
    vdd = d['net_vdd'].astype(bool)
    droop = d['droop'].astype(np.float64)

    circ = parse_netlist(_unzip(bench, 'spice'))
    uf, full_idx, M, clamp_val, ground_rep = _build_node_map(circ)
    assert M == n

    def ridx(name):
        r = uf.find(name)
        return -1 if r == ground_rep else full_idx[r]

    active = ~clamp
    idx = np.full(n, -1, dtype=np.int64); idx[active] = np.arange(active.sum())
    n_act = int(active.sum())

    R_ei, R_val = d['R_edge_index'], d['R_value'].astype(np.float64)
    L_ei = d['L_edge_index']
    a = np.concatenate([R_ei[0], L_ei[0]])
    b = np.concatenate([R_ei[1], L_ei[1]])
    w = np.concatenate([1.0/np.maximum(R_val,1e-9), np.full(L_ei.shape[1], L_SHORT_G)])
    ia, ib = idx[a], idx[b]
    rows, cols, vals = [], [], []
    both = (ia >= 0) & (ib >= 0)
    rows += [ia[both], ib[both], ia[both], ib[both]]
    cols += [ib[both], ia[both], ia[both], ib[both]]
    vals += [-w[both], -w[both], w[both], w[both]]
    for live, dead in ((ia, ib), (ib, ia)):
        m_ = (live >= 0) & (dead < 0)
        rows.append(live[m_]); cols.append(live[m_]); vals.append(w[m_])
    gn, gRv = d['Rg_node'], d['Rg_value'].astype(np.float64)
    ig = idx[gn]; mg = ig >= 0
    rows.append(ig[mg]); cols.append(ig[mg]); vals.append((1.0/np.maximum(gRv,1e-9))[mg])
    ln = d['Lg_node']; il = idx[ln]; ml = il >= 0
    rows.append(il[ml]); cols.append(il[ml]); vals.append(np.full(int(ml.sum()), L_SHORT_G))
    Lg = sp.coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_act, n_act)).tocsc()
    lu = splu(Lg)
    print(f'{bench}: factorized n_act={n_act} ({time.time()-t0:.0f}s)')

    ts = np.arange(0.0, T_END, DT)
    T = ts.size
    I = np.zeros((n_act, T))
    n_src = 0
    for sa, sb, dc, pulse in circ.isources:
        wave = pulse.at(ts) if pulse is not None else np.full(T, dc)
        for nm, sgn in ((sa, -1.0), (sb, +1.0)):
            k = ridx(nm)
            if k >= 0 and idx[k] >= 0:
                I[idx[k]] += sgn * wave
        n_src += 1
    print(f'{bench}: {n_src} sources, {T} time steps, solving...')

    peak = np.zeros(n_act)
    B = 60
    for s in range(0, T, B):
        V = lu.solve(I[:, s:s+B])
        peak = np.maximum(peak, np.abs(V).max(axis=1))
    pred = np.zeros(n); pred[active] = peak

    v0 = d['v_dc'].astype(np.float64)
    vnom = v0[vdd & clamp].max() if (vdd & clamp).any() else v0[vdd].max()
    sd = np.where(vdd, vnom - v0, v0).clip(min=0)   # static baseline

    print(f'--- {bench} (quasi-static + timing, {time.time()-t0:.0f}s) ---')
    for name, m in [('all', grid), ('vdd', grid & vdd), ('gnd', grid & ~vdd)]:
        r_t = spearmanr(pred[m], droop[m]).statistic
        r_s = spearmanr(sd[m], droop[m]).statistic
        print(f'{name:4} | timing-QS vs droop {r_t:+.3f}  (static baseline {r_s:+.3f})')
    print(f'timing-QS peak mV: med {np.median(pred[grid])*1e3:.2f} '
          f'p99 {np.percentile(pred[grid],99)*1e3:.2f}  (droop med {np.median(droop[grid])*1e3:.1f})')

for bench in sys.argv[1:]:
    run(bench)
