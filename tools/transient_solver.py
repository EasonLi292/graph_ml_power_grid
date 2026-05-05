"""Backward-Euler MNA transient solver for ``PDNGraph`` instances.

The graph is reduced to a circuit of (n_top² + n_bot²) free voltage
unknowns. Vdd pads are clamped by elimination; gnd is the implicit zero
reference. Capacitors are stamped via the standard backward-Euler
companion (G_C = C/dt, history current g_c · V_prev). Load nodes are
projected out — each contributes a current injection at its M_bot
attachment point.

The system matrix is constant across timesteps, so we factor it once
with ``scipy.sparse.linalg.splu`` and reuse the factorization.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid_construction import PDNGraph


def square_wave(t, freq: float, duty: float, phase: float):
    """Square wave in [0, 1]. ``phase`` is in fractions of one period."""
    p = (np.asarray(t) * freq + phase) % 1.0
    return (p < duty).astype(float)


def simulate(g: PDNGraph, t_end: float = 5e-9, dt: float = 1e-11) -> dict:
    """Run a transient analysis and return per-node voltage trajectories.

    Initial condition: every mesh node at ``Vdd`` (decaps fully charged
    from t < 0, loads idle). The first cycle therefore captures the
    turn-on transient — discard it as warm-up if you only want the
    periodic steady state.
    """
    n_top_nodes = g.n_top_nodes
    n_bot_nodes = g.n_bot_nodes
    N = n_top_nodes + n_bot_nodes
    top0 = 0
    bot0 = n_top_nodes

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def stamp(a: int, b: int, gv: float) -> None:
        rows.extend([a, b, a, b])
        cols.extend([a, b, b, a])
        vals.extend([gv, gv, -gv, -gv])

    for u, v in g.top_edges:
        stamp(top0 + int(u), top0 + int(v), 1.0 / g.R_top)
    for u, v in g.bot_edges:
        stamp(bot0 + int(u), bot0 + int(v), 1.0 / g.R_bot)
    for ti, bi in g.via_pairs:
        stamp(top0 + int(ti), bot0 + int(bi), 1.0 / g.R_via)

    g_c = g.C_decap / dt
    decap_idx = bot0 + g.decap_attach_bot_idx.astype(int)
    for a in decap_idx:
        rows.append(int(a))
        cols.append(int(a))
        vals.append(g_c)

    G = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    pad_idx = top0 + g.vdd_pad_top_idx.astype(int)
    free_mask = np.ones(N, dtype=bool)
    free_mask[pad_idx] = False
    free = np.where(free_mask)[0]

    G_ff = sp.csc_matrix(G[free, :][:, free])
    G_fx = G[free, :][:, ~free_mask]
    Vx = np.full(pad_idx.size, g.Vdd)
    rhs_const = np.asarray(G_fx @ Vx).ravel()

    solver = spla.splu(G_ff)

    n_steps = int(np.round(t_end / dt))
    t_arr = np.arange(n_steps + 1) * dt

    V = np.full((n_steps + 1, N), g.Vdd, dtype=float)
    I_wave = g.I_peak * square_wave(t_arr, g.freq, g.duty, g.phase)
    load_idx = bot0 + g.load_attach_bot_idx.astype(int)

    I = np.zeros(N)
    for step in range(1, n_steps + 1):
        I.fill(0.0)
        I[load_idx] -= I_wave[step]
        I[decap_idx] += g_c * V[step - 1, decap_idx]
        rhs = I[free] - rhs_const
        V[step, free] = solver.solve(rhs)
        V[step, pad_idx] = g.Vdd

    return {
        "t": t_arr,
        "V_top": V[:, top0:bot0],
        "V_bot": V[:, bot0 : bot0 + n_bot_nodes],
        "I_load": I_wave,
    }
