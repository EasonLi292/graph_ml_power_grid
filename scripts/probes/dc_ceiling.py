"""Training-free ceiling check: exact DC superposition vs transient droop.

Solves the grounded DC system L_g v = I with I = per-node peak load
currents (the exact 'attention physics anchor' Z @ I, computed exactly
instead of via sketch+learning), and reports within-net Spearman against
the transient peak-droop target. Pure numpy/scipy on the raw npz.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.stats import spearmanr

L_SHORT_G = 1e6

def run(bench):
    d = np.load(f'datasets/ibmpg/graphs/{bench}.npz', allow_pickle=True)
    n = int(d['n_nodes'])
    clamp = d['is_clamp'].astype(bool)
    grid = d['is_grid'].astype(bool) & ~clamp
    vdd = d['net_vdd'].astype(bool)
    droop = d['droop'].astype(np.float64)

    R_ei, R_val = d['R_edge_index'], d['R_value'].astype(np.float64)
    L_ei = d['L_edge_index']
    a = np.concatenate([R_ei[0], L_ei[0]])
    b = np.concatenate([R_ei[1], L_ei[1]])
    w = np.concatenate([1.0 / np.maximum(R_val, 1e-9), np.full(L_ei.shape[1], L_SHORT_G)])
    gn = d['Rg_node']; gw = 1.0 / np.maximum(d['Rg_value'].astype(np.float64), 1e-9)
    ln = d['Lg_node']

    # active = non-clamp nodes connected to reference (assume patched grids are connected)
    active = ~clamp
    idx = np.full(n, -1, dtype=np.int64)
    idx[active] = np.arange(active.sum())
    n_act = int(active.sum())

    ia, ib = idx[a], idx[b]
    rows, cols, vals = [], [], []
    both = (ia >= 0) & (ib >= 0)
    rows += [ia[both], ib[both], ia[both], ib[both]]
    cols += [ib[both], ia[both], ia[both], ib[both]]
    vals += [-w[both], -w[both], w[both], w[both]]
    for live, dead in ((ia, ib), (ib, ia)):
        m_ = (live >= 0) & (dead < 0)
        rows.append(live[m_]); cols.append(live[m_]); vals.append(w[m_])
    ig = idx[gn]; mg = ig >= 0
    rows.append(ig[mg]); cols.append(ig[mg]); vals.append(gw[mg])
    il = idx[ln]; ml = il >= 0
    rows.append(il[ml]); cols.append(il[ml]); vals.append(np.full(ml.sum(), L_SHORT_G))
    Lg = sp.coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_act, n_act)).tocsc()

    I = np.zeros(n_act)
    li = idx[np.flatnonzero(d['load_I'] > 0)]
    I[li[li >= 0]] = d['load_I'][d['load_I'] > 0][li >= 0]

    lu = splu(Lg)
    v = np.zeros(n)
    v[active] = lu.solve(I)
    pred = np.abs(v)

    # t=0 static drop for reference (what the current baseline uses)
    v0 = d['v_dc'].astype(np.float64)
    vnom = v0[vdd & clamp].max() if (vdd & clamp).any() else v0[vdd].max()
    sd = np.where(vdd, vnom - v0, v0).clip(min=0)

    print(f'--- {bench}  (n={n}, grid={grid.sum()}) ---')
    for name, m in [('all', grid), ('vdd', grid & vdd), ('gnd', grid & ~vdd)]:
        r_dc  = spearmanr(pred[m], droop[m]).statistic
        r_sd  = spearmanr(sd[m],   droop[m]).statistic
        r_x   = spearmanr(pred[m], sd[m]).statistic
        print(f'{name:4} n={m.sum():7d} | exactDC-vs-droop {r_dc:+.3f} | '
              f't0static-vs-droop {r_sd:+.3f} | exactDC-vs-t0static {r_x:+.3f}')
    print(f'exactDC drop mV: med {np.median(pred[grid])*1e3:.2f} '
          f'p99 {np.percentile(pred[grid],99)*1e3:.2f} max {pred[grid].max()*1e3:.2f}')

for bench in sys.argv[1:]:
    run(bench)
