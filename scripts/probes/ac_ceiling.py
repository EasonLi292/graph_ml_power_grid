"""Training-free AC ceiling probe: |Z(w) I_peak| vs transient droop.

Complex nodal system: R edges 1/R, L edges 1/(jwL), C edges jwC,
cap_gnd jwC diag, Rg 1/R diag, Lg 1/(jwL) diag; clamps grounded.
Synchronous peak currents at load nodes (phase info not in npz).
Sweep w and report within-net Spearman vs droop.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.stats import spearmanr

def run(bench, omegas):
    d = np.load(f'datasets/ibmpg/graphs/{bench}.npz', allow_pickle=True)
    n = int(d['n_nodes'])
    clamp = d['is_clamp'].astype(bool)
    grid = d['is_grid'].astype(bool) & ~clamp
    vdd = d['net_vdd'].astype(bool)
    droop = d['droop'].astype(np.float64)

    active = ~clamp
    idx = np.full(n, -1, dtype=np.int64)
    idx[active] = np.arange(active.sum())
    n_act = int(active.sum())

    def add_branch(rows, cols, vals, ei, y):
        ia, ib = idx[ei[0]], idx[ei[1]]
        both = (ia >= 0) & (ib >= 0)
        rows += [ia[both], ib[both], ia[both], ib[both]]
        cols += [ib[both], ia[both], ia[both], ib[both]]
        vals += [-y[both], -y[both], y[both], y[both]]
        for live, dead in ((ia, ib), (ib, ia)):
            m_ = (live >= 0) & (dead < 0)
            rows.append(live[m_]); cols.append(live[m_]); vals.append(y[m_])

    R_ei, R_val = d['R_edge_index'], d['R_value'].astype(np.float64)
    L_ei, L_val = d['L_edge_index'], d['L_value'].astype(np.float64)
    C_ei, C_val = d['C_edge_index'], d['C_value'].astype(np.float64)
    cg = d['cap_gnd'].astype(np.float64)
    gn, gRv = d['Rg_node'], d['Rg_value'].astype(np.float64)
    ln, gLv = d['Lg_node'], d['Lg_value'].astype(np.float64)

    I = np.zeros(n_act)
    li_nodes = np.flatnonzero(d['load_I'] > 0)
    li = idx[li_nodes]
    I[li[li >= 0]] = d['load_I'][li_nodes][li >= 0]

    print(f'--- {bench} (n={n}) ---')
    for w_ in omegas:
        rows, cols, vals = [], [], []
        add_branch(rows, cols, vals, R_ei, (1.0/np.maximum(R_val,1e-9)).astype(complex))
        if w_ == 0.0:
            add_branch(rows, cols, vals, L_ei, np.full(L_ei.shape[1], 1e6, dtype=complex))
        else:
            add_branch(rows, cols, vals, L_ei, 1.0/(1j*w_*np.maximum(L_val,1e-18)))
            if C_ei.shape[1]:
                add_branch(rows, cols, vals, C_ei, (1j*w_*C_val).astype(complex))
        ig = idx[gn]; mg = ig >= 0
        rows.append(ig[mg]); cols.append(ig[mg])
        vals.append((1.0/np.maximum(gRv,1e-9))[mg].astype(complex))
        il = idx[ln]; ml = il >= 0
        rows.append(il[ml]); cols.append(il[ml])
        if w_ == 0.0:
            vals.append(np.full(int(ml.sum()), 1e6, dtype=complex))
        else:
            vals.append((1.0/(1j*w_*np.maximum(gLv,1e-18)))[ml])
        if w_ > 0:
            icg = idx[np.flatnonzero(cg > 0)]; mc = icg >= 0
            rows.append(icg[mc]); cols.append(icg[mc])
            vals.append((1j*w_*cg[cg > 0])[mc])
        Y = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate([r.astype(np.int64) for r in rows]),
                            np.concatenate([c.astype(np.int64) for c in cols]))),
                          shape=(n_act, n_act)).tocsc()
        v = np.zeros(n)
        v[active] = np.abs(splu(Y).solve(I.astype(complex)))
        out = []
        for name, m in [('all', grid), ('vdd', grid & vdd), ('gnd', grid & ~vdd)]:
            out.append(f'{name} {spearmanr(v[m], droop[m]).statistic:+.3f}')
        print(f'w={w_:9.2e} | ' + ' | '.join(out) +
              f' | medZ*I {np.median(v[grid])*1e3:8.2f} mV')

omegas = [0.0, 1e8, 1e9, 3e9, 1e10, 3e10, 1e11]
for bench in sys.argv[1:]:
    run(bench, omegas)
