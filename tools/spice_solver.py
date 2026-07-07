"""General MNA transient solver for the parsed IBM PG netlists.

Scope is exactly what ``ibmpg*t.spice`` needs (validated against the shipped
``.output``):

* **shorts** — 0 V voltage sources are collapsed by union-find (the netlists
  use 0 V sources, not 0 Ω resistors, for shorts/measurement taps).
* **clamps** — nonzero voltage sources to ground are boundary conditions
  (the 1.8 V VDD pads); folded to the RHS like the synthetic solver's pads.
* **R / C / L** — resistors stamp conductance; capacitors and inductors use
  backward-Euler Norton companions (constant system matrix → factor once).
* **loads** — node-to-ground PULSE current sources, injected each step.
* **initial condition** — DC operating point (caps open, inductors as 0 V
  MNA branches so their steady currents come out directly), so t=0 matches.

Public entry: :func:`solve_transient(circuit) -> {t, V, probe_names}`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .spice_parser import GROUND, Circuit


class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _build_node_map(circ: Circuit):
    """Union-find over 0 V sources → representative node → dense index.

    Returns ``(rep_of, full_idx, M, clamp_val, ground_rep)`` where
    ``full_idx[rep]`` indexes every non-ground rep in ``[0, M)`` and
    ``clamp_val[idx]`` holds the fixed voltage of clamped reps (NaN if free).
    """
    uf = _UF()
    # register every node so isolated ones still resolve
    for lst in (circ.resistors, circ.capacitors, circ.inductors, circ.vsources):
        for a, b, *_ in lst:
            uf.find(a); uf.find(b)
    for a, b, *_ in circ.isources:
        uf.find(a); uf.find(b)
    uf.find(GROUND)
    # merge shorts (0 V sources)
    for a, b, val in circ.vsources:
        if val == 0.0:
            uf.union(a, b)

    ground_rep = uf.find(GROUND)
    reps = sorted({uf.find(n) for n in uf.p} - {ground_rep})
    full_idx = {r: i for i, r in enumerate(reps)}
    M = len(reps)

    clamp_val = np.full(M, np.nan)
    for a, b, val in circ.vsources:
        if val == 0.0:
            continue
        # nonzero source to ground: V_node = ±val
        ra, rb = uf.find(a), uf.find(b)
        if rb == ground_rep and ra != ground_rep:
            clamp_val[full_idx[ra]] = val
        elif ra == ground_rep and rb != ground_rep:
            clamp_val[full_idx[rb]] = -val
        else:
            raise NotImplementedError(
                "nonzero V source between two non-ground nodes not supported"
            )
    return uf, full_idx, M, clamp_val, ground_rep


def _idx_of(uf, full_idx, ground_rep, name: str) -> int:
    r = uf.find(name)
    return -1 if r == ground_rep else full_idx[r]


def _coo_conductance(pairs_g, M):
    """Assemble a symmetric conductance matrix from (p, q, g) with p/q in
    ``[-1, M)`` (-1 = ground reference, dropped)."""
    rows, cols, vals = [], [], []
    for p, q, g in pairs_g:
        if p == q:
            continue
        if p >= 0 and q >= 0:
            rows += [p, q, p, q]; cols += [p, q, q, p]; vals += [g, g, -g, -g]
        elif p >= 0:
            rows.append(p); cols.append(p); vals.append(g)
        elif q >= 0:
            rows.append(q); cols.append(q); vals.append(g)
    return sp.csr_matrix((vals, (rows, cols)), shape=(M, M))


def _dc_operating_point(circ, uf, full_idx, M, clamp_val, ground_rep,
                        r_idx, l_idx):
    """DC OP with caps open and inductors as 0 V MNA branches.

    Returns ``(V0_full[M], I_L0[n_ind])`` — node voltages and inductor
    steady currents (a→b), used as the transient initial condition.
    """
    clamp_mask = ~np.isnan(clamp_val)
    free = np.where(~clamp_mask)[0]
    F = free.size
    freepos = -np.ones(M, dtype=int)
    freepos[free] = np.arange(F)
    Vc = np.where(clamp_mask, np.nan_to_num(clamp_val), 0.0)

    # R-only conductance over all reps, then partition free/clamp.
    G = _coo_conductance([(p, q, 1.0 / R) for (p, q), R in zip(r_idx, circ_R(circ))], M)
    G_ff = sp.csc_matrix(G[free][:, free])
    G_fc = G[free][:, clamp_mask]
    rhs = -np.asarray(G_fc @ Vc[clamp_mask]).ravel()

    # DC current-source injections at t=0.
    J = np.zeros(M)
    for (a, b, dc, pulse), in_pair in zip(circ.isources, _isrc_idx(circ, uf, full_idx, ground_rep)):
        val = pulse.at(np.array([0.0]))[0] if pulse is not None else dc
        pa, pb = in_pair
        if pa >= 0:
            J[pa] -= val
        if pb >= 0:
            J[pb] += val
    rhs = rhs + J[free]

    # Inductor incidence over free nodes + constraint RHS.
    nL = len(circ.inductors)
    A = sp.lil_matrix((F, nL))
    c = np.zeros(nL)
    for j, (pa, pb) in enumerate(l_idx):
        ka = clamp_val[pa] if pa >= 0 and clamp_mask[pa] else (0.0 if pa < 0 else None)
        kb = clamp_val[pb] if pb >= 0 and clamp_mask[pb] else (0.0 if pb < 0 else None)
        if pa >= 0 and not clamp_mask[pa]:
            A[freepos[pa], j] += 1.0
        if pb >= 0 and not clamp_mask[pb]:
            A[freepos[pb], j] -= 1.0
        c[j] = (kb or 0.0) - (ka or 0.0)
    A = A.tocsr()

    # Augmented [[G_ff, A],[A^T, 0]] [Vf; IL] = [rhs; c]
    top = sp.hstack([G_ff, A])
    bot = sp.hstack([A.T, sp.csr_matrix((nL, nL))])
    K = sp.csc_matrix(sp.vstack([top, bot]))
    sol = spla.spsolve(K, np.concatenate([rhs, c]))
    Vf, IL = sol[:F], sol[F:]

    V0 = np.where(clamp_mask, Vc, 0.0)
    V0[free] = Vf
    return V0, IL


def circ_R(circ):
    return [R for _, _, R in circ.resistors]


def _isrc_idx(circ, uf, full_idx, ground_rep):
    return [
        (_idx_of(uf, full_idx, ground_rep, a), _idx_of(uf, full_idx, ground_rep, b))
        for a, b, *_ in circ.isources
    ]


def solve_transient(circ: Circuit, dt: float | None = None,
                    t_end: float | None = None, return_all: bool = False,
                    track_extrema: bool = False) -> dict:
    dt = dt or circ.tran_dt
    t_end = t_end or circ.tran_tend
    if dt is None or t_end is None:
        raise ValueError("dt / t_end required (not in netlist)")

    uf, full_idx, M, clamp_val, ground_rep = _build_node_map(circ)
    clamp_mask = ~np.isnan(clamp_val)
    free = np.where(~clamp_mask)[0]
    Vc_full = np.where(clamp_mask, np.nan_to_num(clamp_val), 0.0)

    # index arrays
    r_idx = [(_idx_of(uf, full_idx, ground_rep, a), _idx_of(uf, full_idx, ground_rep, b))
             for a, b, _ in circ.resistors]
    c_idx = [(_idx_of(uf, full_idx, ground_rep, a), _idx_of(uf, full_idx, ground_rep, b))
             for a, b, _ in circ.capacitors]
    l_idx = [(_idx_of(uf, full_idx, ground_rep, a), _idx_of(uf, full_idx, ground_rep, b))
             for a, b, _ in circ.inductors]
    i_idx = _isrc_idx(circ, uf, full_idx, ground_rep)

    # ----- DC initial condition -----
    V0, IL = _dc_operating_point(circ, uf, full_idx, M, clamp_val, ground_rep, r_idx, l_idx)

    # ----- constant transient conductance: R + C/dt + dt/L -----
    pairs = [(p, q, 1.0 / R) for (p, q), R in zip(r_idx, circ_R(circ))]
    gc = np.array([C / dt for _, _, C in circ.capacitors])
    gl = np.array([dt / L for _, _, L in circ.inductors])
    pairs += [(p, q, g) for (p, q), g in zip(c_idx, gc)]
    pairs += [(p, q, g) for (p, q), g in zip(l_idx, gl)]
    G = _coo_conductance(pairs, M)
    G_ff = sp.csc_matrix(G[free][:, free])
    G_fc = G[free][:, clamp_mask]
    rhs_const = np.asarray(G_fc @ Vc_full[clamp_mask]).ravel()
    solver = spla.splu(G_ff)

    n_steps = int(round(t_end / dt))
    t_arr = np.arange(n_steps + 1) * dt

    # vectorized scatter helpers
    def arr(idx_pairs):
        a = np.array([p for p, _ in idx_pairs], dtype=int)
        b = np.array([q for _, q in idx_pairs], dtype=int)
        return a, b
    ia, ib = arr(i_idx); ca, cb = arr(c_idx); la, lb = arr(l_idx)

    def vget(idx, V):
        out = np.zeros(idx.shape)
        m = idx >= 0
        out[m] = V[idx[m]]
        return out

    # Load waveforms evaluated per-step (no [n_i, T] matrix → scales to the
    # 0.7M-source benchmarks). Pack PULSE params as arrays; non-pulse sources
    # use i1=i2=dc so the same formula yields a constant.
    n_i = len(circ.isources)
    pi1 = np.array([(p.i1 if p else dc) for _, _, dc, p in circ.isources])
    pi2 = np.array([(p.i2 if p else dc) for _, _, dc, p in circ.isources])
    ptd = np.array([(p.td if p else 0.0) for _, _, _, p in circ.isources])
    ptr = np.array([(p.tr if p else 0.0) for _, _, _, p in circ.isources])
    ptf = np.array([(p.tf if p else 0.0) for _, _, _, p in circ.isources])
    ppw = np.array([(p.pw if p else 0.0) for _, _, _, p in circ.isources])
    pper = np.array([(p.per if p else 0.0) for _, _, _, p in circ.isources])

    def isrc_at(t: float) -> np.ndarray:
        out = pi1.copy()
        active = t >= ptd
        local = np.where(pper > 0, np.mod(t - ptd, pper), t - ptd)
        m = active & (ptr > 0) & (local < ptr)
        out[m] = pi1[m] + (pi2[m] - pi1[m]) * (local[m] / ptr[m])
        m = active & (local >= ptr) & (local < ptr + ppw)
        out[m] = pi2[m]
        m = active & (ptf > 0) & (local >= ptr + ppw) & (local < ptr + ppw + ptf)
        out[m] = pi2[m] + (pi1[m] - pi2[m]) * ((local[m] - ptr[m] - ppw[m]) / ptf[m])
        m = active & (local >= ptr + ppw + ptf)
        out[m] = pi1[m]
        return out

    store_full = return_all
    if store_full:
        Vfull = np.zeros((n_steps + 1, M))
        Vfull[0] = V0
        Vfull[:, clamp_mask] = Vc_full[clamp_mask]
    vmin = V0.copy()
    vmax = V0.copy()
    probe_idx = [_idx_of(uf, full_idx, ground_rep, n) for n in circ.probes]
    probe_V = np.zeros((n_steps + 1, len(probe_idx)))
    for j, pi in enumerate(probe_idx):
        probe_V[0, j] = V0[pi] if pi >= 0 else 0.0

    Vprev = V0.copy()
    ilc = IL.copy()
    mIa, mIb = ia >= 0, ib >= 0
    mca, mcb = ca >= 0, cb >= 0
    mla, mlb = la >= 0, lb >= 0

    for step in range(1, n_steps + 1):
        rhs = np.zeros(M)
        w = isrc_at(t_arr[step])
        np.add.at(rhs, ia[mIa], -w[mIa])           # load sources
        np.add.at(rhs, ib[mIb], +w[mIb])
        dVc = vget(ca, Vprev) - vget(cb, Vprev)    # cap BE history
        Ih = gc * dVc
        np.add.at(rhs, ca[mca], Ih[mca])
        np.add.at(rhs, cb[mcb], -Ih[mcb])
        np.add.at(rhs, la[mla], -ilc[mla])         # inductor BE history
        np.add.at(rhs, lb[mlb], +ilc[mlb])

        Vf = solver.solve(rhs[free] - rhs_const)
        Vnow = Vc_full.copy()
        Vnow[free] = Vf
        if store_full:
            Vfull[step] = Vnow
        np.minimum(vmin, Vnow, out=vmin)
        np.maximum(vmax, Vnow, out=vmax)
        for j, pi in enumerate(probe_idx):
            if pi >= 0:
                probe_V[step, j] = Vnow[pi]
        # inductor current update: I_n = I_{n-1} + gl (Va - Vb)
        ilc = ilc + gl * (vget(la, Vnow) - vget(lb, Vnow))
        Vprev = Vnow

    out = {
        "t": t_arr,
        "v_dc": V0,                 # DC operating point (per rep)
        "vmin": vmin,               # per-rep min over the window
        "vmax": vmax,               # per-rep max over the window
        "nodemap": (uf, full_idx, ground_rep, clamp_val, M),
    }
    if circ.probes:
        out["probe_names"] = list(circ.probes)
        out["probe_V"] = probe_V
    if store_full:
        out["V"] = Vfull
    return out
