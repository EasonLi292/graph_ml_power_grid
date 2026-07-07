"""DC impedance sketch for the IBM grids (stage-0 diagnostic geometry).

For the grounded resistive system ``L_g v = i`` (clamps = ground
reference, package inductors shorted — exact at DC), the transfer
impedance is ``Z = L_g^{-1}``. Because ``Z = (L_g^{-1} B^T W^{1/2})
(W^{1/2} B L_g^{-1})``, sketching with random ±1 edge vectors gives node
embeddings

    r_i = (1/sqrt(m)) * [x_1(i), ..., x_m(i)],
    x_k = L_g^{-1} B^T W^{1/2} q_k,   q_k ~ Rademacher over edges,

with ``r_i · r_j ≈ Z_ij`` (Johnson–Lindenstrauss on the factor rows).
So the DC superposition ``droop ≈ Σ_j Z_ij I_j`` is *exactly* linear
attention with the sketch as the feature map — the physics anchor of
the attention architecture.

Nodes with no resistive path to any clamp (nothing after L-shorting)
get zero embeddings + a floating flag.

Sketches are cached per (bench, m) under ``datasets/ibmpg/graphs/_sketch``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .ibmpg_patches import GRAPHS_DIR, IBMGraph

SKETCH_DIR_NAME = "_sketch"
L_SHORT_G = 1e6          # siemens; package inductors are DC shorts


def _grounded_system(g: IBMGraph):
    """Node-node edges (a, b, w) and node-ground edges (gn, gw).

    R-graph with package L as shorts (exact at DC); ground edges are the
    backfilled node-to-ground resistors (the GND net's tie to the
    reference — without them the whole GND net floats).
    """
    R_ei, R_val = g.edges["R"]
    L_ei, _ = g.edges["L"]
    a = np.concatenate([R_ei[0], L_ei[0]])
    b = np.concatenate([R_ei[1], L_ei[1]])
    w = np.concatenate([1.0 / R_val.astype(np.float64),
                        np.full(L_ei.shape[1], L_SHORT_G)])
    gg = np.asarray(g.x_raw["g_gnd"], dtype=np.float64).copy()
    # package inductors to ground = DC shorts to the reference
    gg[np.asarray(g.x_raw["l_gnd"], dtype=bool)] += L_SHORT_G
    gn = np.flatnonzero(gg > 0)
    return a, b, w, gn, gg[gn]


def compute_sketch(g: IBMGraph, m: int = 32, seed: int = 0):
    """Return (r [n, m] float32, floating [n] bool)."""
    n = g.n_nodes
    a, b, w, gn, gw = _grounded_system(g)
    clamp = np.asarray(g.x_raw["is_clamp"], dtype=bool)

    # connectivity to the reference (clamps or ground resistors) — via a
    # virtual ground node n; anything not in its component floats.
    va = np.concatenate([a, gn, np.flatnonzero(clamp)])
    vb = np.concatenate([b, np.full(gn.size, n), np.full(int(clamp.sum()), n)])
    adj = sp.coo_matrix((np.ones_like(va, dtype=np.float64), (va, vb)),
                        shape=(n + 1, n + 1))
    adj = adj + adj.T
    _, comp = sp.csgraph.connected_components(adj.tocsr(), directed=False)
    active = (comp[:n] == comp[n]) & ~clamp
    idx = np.full(n, -1, dtype=np.int64)
    idx[active] = np.arange(int(active.sum()))
    n_act = int(active.sum())

    # grounded Laplacian over active nodes (clamp/ground entries dropped)
    ia, ib = idx[a], idx[b]
    rows, cols, vals = [], [], []
    both = (ia >= 0) & (ib >= 0)
    rows += [ia[both], ib[both], ia[both], ib[both]]
    cols += [ib[both], ia[both], ia[both], ib[both]]
    vals += [-w[both], -w[both], w[both], w[both]]
    # edges to a clamp contribute only the diagonal of the live endpoint
    for live, dead in ((ia, ib), (ib, ia)):
        m_ = (live >= 0) & (dead < 0)
        rows.append(live[m_]); cols.append(live[m_]); vals.append(w[m_])
    # node-to-ground resistors: pure diagonal
    ig = idx[gn]
    mg = ig >= 0
    rows.append(ig[mg]); cols.append(ig[mg]); vals.append(gw[mg])
    Lg = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_act, n_act),
    ).tocsc()

    lu = splu(Lg)
    rng = np.random.default_rng(seed)
    r = np.zeros((n, m), dtype=np.float32)
    sw = np.sqrt(w)
    swg = np.sqrt(gw)
    n_edges = w.size + gw.size
    for k in range(m):
        q = rng.choice([-1.0, 1.0], size=n_edges)
        rhs = np.zeros(n_act)
        contrib = sw * q[: w.size]
        # b_k = B^T W^{1/2} q: +sqrt(w)q at a, -sqrt(w)q at b (live entries)
        m_a = ia >= 0
        m_b = ib >= 0
        np.add.at(rhs, ia[m_a], contrib[m_a])
        np.add.at(rhs, ib[m_b], -contrib[m_b])
        np.add.at(rhs, ig[mg], (swg * q[w.size:])[mg])   # ground edges: one live row
        x = lu.solve(rhs)
        r[active, k] = x / np.sqrt(m)
    return r, ~active & ~clamp   # clamps: zero embedding but not "floating"


def load_or_compute_sketch(
    bench: str, g: IBMGraph, m: int = 32, seed: int = 0,
    graphs_dir: Path = GRAPHS_DIR,
):
    cache = graphs_dir / SKETCH_DIR_NAME / f"{bench}_m{m}_s{seed}.npz"
    if cache.exists():
        d = np.load(cache)
        if int(d["n_nodes"]) == g.n_nodes:
            return d["r"], d["floating"]
    r, floating = compute_sketch(g, m=m, seed=seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, r=r, floating=floating,
                        n_nodes=np.int64(g.n_nodes))
    return r, floating
