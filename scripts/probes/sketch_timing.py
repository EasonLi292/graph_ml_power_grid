"""Sketched timing readout: pred_i = max_t |r_i . s(t)|, s(t) = sum_j r_j I_j(t).

Validates that the attention architecture's KV-cache path (JL sketch,
basis-invariant) preserves the timing-QS ranking, at m=32 and m=128.
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
T_END = 6e-9

def run(bench, ms):
    t0 = time.time()
    d = np.load(f'datasets/ibmpg/graphs/{bench}.npz', allow_pickle=True)
    n = int(d['n_nodes'])
    clamp = d['is_clamp'].astype(bool)
    grid = d['is_grid'].astype(bool) & ~clamp
    vdd = d['net_vdd'].astype(bool)
    droop = d['droop'].astype(np.float64)

    circ = parse_netlist(_unzip(bench, 'spice'))
    uf, full_idx, M, clamp_val, ground_rep = _build_node_map(circ)
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

    # per-load node waveforms, time-binned
    ts = np.arange(0.0, T_END, DT); T = ts.size
    I = np.zeros((n_act, T))
    for sa, sb, dc, pulse in circ.isources:
        wave = pulse.at(ts) if pulse is not None else np.full(T, dc)
        for nm, sgn in ((sa, -1.0), (sb, +1.0)):
            k = ridx(nm)
            if k >= 0 and idx[k] >= 0:
                I[idx[k]] += sgn * wave

    # JL sketch on this same numbering: r = Lg^-1 B^T W^1/2 q / sqrt(m)
    rng = np.random.default_rng(0)
    sw = np.sqrt(w); swg = np.sqrt(gRv*0 + (1.0/np.maximum(gRv,1e-9)))[mg]
    m_a = ia >= 0; m_b = ib >= 0
    mmax = max(ms)
    RHS = np.zeros((n_act, mmax))
    n_edges = w.size + int(mg.sum())
    for k in range(mmax):
        q = rng.choice([-1.0, 1.0], size=n_edges)
        contrib = sw * q[: w.size]
        rhs = np.zeros(n_act)
        np.add.at(rhs, ia[m_a], contrib[m_a])
        np.add.at(rhs, ib[m_b], -contrib[m_b])
        np.add.at(rhs, ig[mg], swg * q[w.size:])
        RHS[:, k] = rhs
    Xall = lu.solve(RHS)                      # [n_act, mmax]
    print(f'{bench}: sketch solves done ({time.time()-t0:.0f}s)')

    # exact reference
    peak_ex = np.zeros(n_act)
    for s in range(0, T, 60):
        V = lu.solve(I[:, s:s+60])
        peak_ex = np.maximum(peak_ex, np.abs(V).max(axis=1))

    v0 = d['v_dc'].astype(np.float64)
    for m_dim in ms:
        r = Xall[:, :m_dim] / np.sqrt(m_dim)  # [n_act, m]
        s_t = r.T @ I                          # [m, T] KV cache
        pred_sk = np.abs(r @ s_t).max(axis=1)  # [n_act]
        out = []
        for name, mm in [('all', grid), ('vdd', grid & vdd), ('gnd', grid & ~vdd)]:
            mmact = mm[active[np.arange(n)]] if False else None
            pa = np.zeros(n); pa[active] = pred_sk
            out.append(f'{name} {spearmanr(pa[mm], droop[mm]).statistic:+.3f}')
        pe = np.zeros(n); pe[active] = peak_ex
        ref = spearmanr(pe[grid], droop[grid]).statistic
        print(f'{bench} m={m_dim:4d} | sketched timing-QS: ' + ' | '.join(out)
              + f'   (exact all {ref:+.3f})')

for bench in sys.argv[1:]:
    run(bench, [32, 128])
