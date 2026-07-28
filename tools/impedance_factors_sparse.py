"""Sparse, cached impedance factors for IBM-scale grids (forward only).

Same mechanism as :mod:`tools.impedance_factors` — randomized subspace
iteration giving ``p_i^T s_j ~= Z_ij(w)`` — but the admittance is assembled
sparsely and solved with a sparse LU, so it runs on the 25 k – 1.7 M node
IBM benchmarks where the dense prototype cannot.

    Y(w) = B^T diag(y(w)) B      y = 1/R | jwC | 1/(jwL)  (+ node-to-ground)
    X  = solve(Y, Omega);  repeat q:  X <- solve(Y, X)
    Qr = qr(X).Q
    T  = Qr^H Z Qr                       p_i = Qr_i,  s_j = (conj(Qr) T^T)_j

Conjugate transposes are required at w>0: a complex QR is unitary, not
complex-orthogonal (see tools/impedance_factors._subspace_factors).

**FORWARD ONLY — THIS IS A PROTOTYPE PATH.** The solve runs in SciPy, so
the returned factors are detached NumPy/torch tensors with no gradient to
R, C or L. That is sufficient for IBM *learning* experiments (where the
grids have no design knobs anyway) but it does **not** satisfy the repair
objective in docs/OBJECTIVE.md, which needs d(droop)/d(component). The
differentiable replacement is the implicit-adjoint sparse solver (the
adjoint of a linear solve is another solve with Y^T); until that lands, do
not present results computed from these factors as repair-capable.

Cache: ``datasets/ibmpg/graphs/_impfac/<bench>_m<m>_q<q>_f<hash>.npz``.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .ibmpg_patches import GRAPHS_DIR, IBMGraph

FACTOR_DIR_NAME = "_impfac"
L_SHORT_G = 1e6          # siemens; package inductors are DC shorts at w=0


def _branches(g: IBMGraph, omega: float):
    """(rows, cols, vals) triplets for every branch family at ``omega``.

    Node-node families: R, C, L. Node-ground families: g_gnd (backfilled
    node-to-ground resistors) and l_gnd (package inductors to the
    reference). Complex throughout so one code path serves DC and AC.
    """
    fams = []
    R_ei, R_val = g.edges["R"]
    fams.append((R_ei, (1.0 / np.maximum(R_val.astype(np.float64), 1e-12)).astype(np.complex128)))

    C_ei, C_val = g.edges["C"]
    if C_ei.size and omega != 0.0:
        fams.append((C_ei, (1j * omega * C_val.astype(np.float64))))

    L_ei, L_val = g.edges["L"]
    if L_ei.size:
        yl = (np.full(L_ei.shape[1], L_SHORT_G, dtype=np.complex128)
              if omega == 0.0
              else 1.0 / (1j * omega * np.maximum(L_val.astype(np.float64), 1e-18)))
        fams.append((L_ei, yl))
    return fams


def sparse_admittance(g: IBMGraph, omega: float, free_of: np.ndarray, n_free: int):
    """Complex sparse ``Y(omega)`` over free (non-clamp) nodes."""
    rows, cols, vals = [], [], []
    for ei, y in _branches(g, omega):
        ia, ib = free_of[ei[0]], free_of[ei[1]]
        both = (ia >= 0) & (ib >= 0)
        a, b, w = ia[both], ib[both], y[both]
        rows += [a, b, a, b]
        cols += [a, b, b, a]
        vals += [w, w, -w, -w]
        for live, dead in ((ia, ib), (ib, ia)):
            m = (live >= 0) & (dead < 0)
            if m.any():
                rows.append(live[m]); cols.append(live[m]); vals.append(y[m])

    gg = np.asarray(g.x_raw["g_gnd"], dtype=np.float64).astype(np.complex128)
    if omega == 0.0:
        # package inductors are exact shorts to the reference at DC
        gg = gg + np.asarray(g.x_raw["l_gnd"], dtype=bool) * L_SHORT_G
    else:
        # y = 1/(jwL), using the pooled reciprocal inductance (parallel
        # inductors add in 1/L). Never invent an L here: the DC-short
        # magnitude is meaningless at w>0.
        inv_l = np.asarray(g.x_raw.get("l_gnd_inv",
                                       np.zeros(g.n_nodes)), dtype=np.float64)
        gg = gg + inv_l / (1j * omega)
    nz = np.flatnonzero(gg != 0)
    ig = free_of[nz]
    m = ig >= 0
    rows.append(ig[m]); cols.append(ig[m]); vals.append(gg[nz][m])

    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_free, n_free)).tocsc()


def free_map(g: IBMGraph):
    """Non-clamp nodes get a dense index; clamps map to -1."""
    clamp = np.asarray(g.x_raw["is_clamp"], dtype=bool)
    free_of = np.full(g.n_nodes, -1, dtype=np.int64)
    free = np.flatnonzero(~clamp)
    free_of[free] = np.arange(free.size)
    return free_of, int(free.size)


def sparse_factors(g: IBMGraph, omegas, m: int = 16, n_power: int = 2,
                   seed: int = 0, verbose: bool = False):
    """Return ``(p, s)`` float32 ``[n_nodes, C_ch, m]`` (channels as in
    :func:`tools.impedance_factors.impedance_factors`) plus timing info."""
    free_of, n_free = free_map(g)
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((n_free, m))
    p_ch, s_ch, info = [], [], []

    for om in np.atleast_1d(np.asarray(omegas, dtype=np.float64)):
        t0 = time.time()
        Y = sparse_admittance(g, float(om), free_of, n_free)
        lu = splu(Y)
        t_fac = time.time() - t0
        X = lu.solve(probes.astype(Y.dtype))
        for _ in range(n_power):
            X = lu.solve(X)
        Qr, _ = np.linalg.qr(X)
        T = Qr.conj().T @ lu.solve(Qr)          # Qr^H Z Qr
        pf, sf = Qr, Qr.conj() @ T.T
        if float(om) == 0.0:
            p_ch += [pf.real]; s_ch += [sf.real]
        else:
            p_ch += [pf.real, pf.imag, pf.real, pf.imag]
            s_ch += [sf.real, sf.imag, sf.imag, sf.real]
        info.append({"omega": float(om), "lu_s": t_fac,
                     "total_s": time.time() - t0, "nnz": int(Y.nnz)})
        if verbose:
            print(f"    w={float(om):.3e}: LU {t_fac:.1f}s, "
                  f"total {time.time()-t0:.1f}s, nnz={Y.nnz:,}")

    n_ch = len(p_ch)
    p = np.zeros((g.n_nodes, n_ch, m), dtype=np.float32)
    s = np.zeros((g.n_nodes, n_ch, m), dtype=np.float32)
    live = free_of >= 0
    idx = free_of[live]
    for c in range(n_ch):
        p[live, c] = p_ch[c][idx]
        s[live, c] = s_ch[c][idx]
    return p, s, info


def load_or_compute(bench: str, g: IBMGraph, omegas, m: int = 16,
                    n_power: int = 2, seed: int = 0,
                    graphs_dir: Path = GRAPHS_DIR, verbose: bool = False):
    """Disk-cached wrapper. Cache key covers every shape-changing knob."""
    oms = np.atleast_1d(np.asarray(omegas, dtype=np.float64))
    h = hashlib.md5(oms.tobytes()).hexdigest()[:8]
    path = graphs_dir / FACTOR_DIR_NAME / f"{bench}_m{m}_q{n_power}_s{seed}_f{h}.npz"
    if path.exists():
        d = np.load(path)
        if int(d["n_nodes"]) == g.n_nodes:
            return d["p"], d["s"]
    p, s, _ = sparse_factors(g, oms, m=m, n_power=n_power, seed=seed,
                             verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p, s=s, n_nodes=np.int64(g.n_nodes),
                        omegas=oms)
    return p, s


def reconstruct(p: np.ndarray, s: np.ndarray, i, j, dc_only: bool = False):
    """``Z_ij`` from the real channels: DC = ch0; AC = (rr-ii) + 1j(ri+ir)."""
    pi, sj = p[i], s[j]
    z = np.einsum("...cm,...cm->...c", pi, sj) if pi.ndim > 1 else pi * sj
    dc = z[..., 0]
    if dc_only or z.shape[-1] == 1:
        return dc
    rr, ii, ri, ir = z[..., 1], z[..., 2], z[..., 3], z[..., 4]
    return dc, (rr - ii) + 1j * (ri + ir)
