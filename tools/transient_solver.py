"""Backward-Euler MNA transient solver + DC solver for ``PDNGraph``.

The graph is a two-rail (Vdd + Vss) circuit. Every intersection on
M_top and M_bot is one node; its net (Vdd or Vss) is read from
``g.top_is_vdd`` / ``g.bot_is_vdd``. Boundary conditions are per-node:

* Corner-pad nodes on M_top get clamped to ``Vdd`` (if Vdd) or ``0``
  (if Vss). The relevant index arrays are ``g.vdd_pad_top_idx`` and
  ``g.vss_pad_top_idx``.
* All other nodes are free.

Decap and load instances are encoded as ``(Vdd_bot_idx, Vss_bot_idx)``
pairs in ``g.decap_pairs`` and ``g.load_pairs``. Capacitors are stamped
via the standard backward-Euler companion across the two terminals; the
history-current source comes from the previous step's voltage delta.
Each load contributes a current waveform that flows *out* of its
Vdd-bot terminal and *into* its Vss-bot terminal — the way a real
switching gate pulls supply current.

The transient system matrix is constant across timesteps, so we factor
it once with ``scipy.sparse.linalg.splu`` and reuse the factorization.

``solve_static_dc`` shares the same conductance-stamping but skips the
capacitor companion and the time loop: it solves the DC operating point
under the time-averaged load current (``I_peak × duty`` per load). This
is a clean topology-only signal independent of decap dynamics.

The returned ``V_bot`` / ``V_top`` arrays are *raw* per-node voltages;
the dataset runner subtracts the Vss-side from the Vdd-side at each
load location to get the local supply-rail voltage.
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


def _stamp_resistors(g: PDNGraph, top0: int, bot0: int):
    """Conductance stamps for every resistor in the two-rail mesh.

    Three families: top straps, bot straps, vias. All edges always
    connect same-net nodes (by construction in the builder), so no
    cross-net resistor stamps are emitted here.
    """
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
    return rows, cols, vals


def _stamp_decap_companion(g: PDNGraph, bot0: int, dt: float):
    """Backward-Euler companion conductance for each cross-net decap.

    A capacitor between bot[u] (Vdd) and bot[v] (Vss) stamps a 2×2
    G-block with g_c on the diagonals and −g_c off-diagonals.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    g_c = g.C_decap / dt
    for u, v in g.decap_pairs:
        a, b = bot0 + int(u), bot0 + int(v)
        rows.extend([a, b, a, b])
        cols.extend([a, b, b, a])
        vals.extend([g_c, g_c, -g_c, -g_c])
    return rows, cols, vals


def _clamp_indices(g: PDNGraph, top0: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and clamp voltages for the corner-bump boundary condition."""
    vdd_idx = top0 + g.vdd_pad_top_idx.astype(int)
    vss_idx = top0 + g.vss_pad_top_idx.astype(int)
    pad_idx = np.concatenate([vdd_idx, vss_idx])
    pad_v = np.concatenate([
        np.full(vdd_idx.size, g.Vdd),
        np.zeros(vss_idx.size),
    ])
    return pad_idx, pad_v


def _initial_voltages(g: PDNGraph, top0: int, bot0: int, N: int) -> np.ndarray:
    """Initial condition: Vdd nodes at Vdd, Vss nodes at 0.

    This puts the decaps at full charge and the mesh at rest before the
    loads start switching at t=0.
    """
    V = np.zeros(N, dtype=float)
    V[top0:top0 + g.n_top_nodes] = np.where(g.top_is_vdd == 1, g.Vdd, 0.0)
    V[bot0:bot0 + g.n_bot_nodes] = np.where(g.bot_is_vdd == 1, g.Vdd, 0.0)
    return V


def simulate(g: PDNGraph, t_end: float = 5e-9, dt: float = 1e-11) -> dict:
    """Run a transient analysis and return per-node voltage trajectories.

    Returns a dict with::

        t       : [T+1]                       sample times
        V_top   : [T+1, n_top_nodes]          raw voltages on every M_top node
        V_bot   : [T+1, n_bot_nodes]          raw voltages on every M_bot node
        I_loads : [n_loads, T+1]              per-load instantaneous current
    """
    top0 = 0
    bot0 = g.n_top_nodes
    N = g.n_top_nodes + g.n_bot_nodes

    rows, cols, vals = _stamp_resistors(g, top0, bot0)
    drows, dcols, dvals = _stamp_decap_companion(g, bot0, dt)
    rows.extend(drows); cols.extend(dcols); vals.extend(dvals)
    G = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    pad_idx, pad_v = _clamp_indices(g, top0)
    free_mask = np.ones(N, dtype=bool)
    free_mask[pad_idx] = False
    free = np.where(free_mask)[0]

    G_ff = sp.csc_matrix(G[free, :][:, free])
    # Use ``pad_idx`` directly (not ``~free_mask``) so the column order of
    # G_fx matches the order of ``pad_v``. Boolean masking re-sorts by
    # natural index, which silently misaligns ``pad_v`` when Vdd and Vss
    # clamps differ.
    G_fx = G[free, :][:, pad_idx]
    rhs_const = np.asarray(G_fx @ pad_v).ravel()

    solver = spla.splu(G_ff)

    n_steps = int(np.round(t_end / dt))
    t_arr = np.arange(n_steps + 1) * dt

    V = np.zeros((n_steps + 1, N), dtype=float)
    V[0] = _initial_voltages(g, top0, bot0, N)
    V[:, pad_idx] = pad_v  # pads stay clamped throughout

    # Per-load current waveforms.
    if g.n_loads > 0:
        load_vdd_idx = bot0 + g.load_pairs[:, 0].astype(int)
        load_vss_idx = bot0 + g.load_pairs[:, 1].astype(int)
        I_waves = np.stack(
            [
                g.loads[k, 0] * square_wave(t_arr, g.loads[k, 1], g.loads[k, 2], g.loads[k, 3])
                for k in range(g.n_loads)
            ]
        )
    else:
        load_vdd_idx = np.empty(0, dtype=int)
        load_vss_idx = np.empty(0, dtype=int)
        I_waves = np.empty((0, n_steps + 1))

    # Decap history terminals (paired Vdd / Vss bot indices).
    if g.n_decaps > 0:
        dec_vdd_idx = bot0 + g.decap_pairs[:, 0].astype(int)
        dec_vss_idx = bot0 + g.decap_pairs[:, 1].astype(int)
    else:
        dec_vdd_idx = np.empty(0, dtype=int)
        dec_vss_idx = np.empty(0, dtype=int)
    g_c = g.C_decap / dt

    I = np.zeros(N)
    for step in range(1, n_steps + 1):
        I.fill(0.0)
        # Load sinks current at Vdd-bot, returns at Vss-bot.
        if g.n_loads > 0:
            np.subtract.at(I, load_vdd_idx, I_waves[:, step])
            np.add.at(I, load_vss_idx, I_waves[:, step])
        # Backward-Euler decap history: g_c × (V_vdd_prev − V_vss_prev),
        # injected positively into the Vdd-bot terminal.
        if g.n_decaps > 0:
            dV_prev = V[step - 1, dec_vdd_idx] - V[step - 1, dec_vss_idx]
            I[dec_vdd_idx] += g_c * dV_prev
            I[dec_vss_idx] -= g_c * dV_prev

        rhs = I[free] - rhs_const
        V[step, free] = solver.solve(rhs)

    V_top = V[:, top0:top0 + g.n_top_nodes]
    V_bot = V[:, bot0:bot0 + g.n_bot_nodes]
    return {"t": t_arr, "V_top": V_top, "V_bot": V_bot, "I_loads": I_waves}


def solve_static_dc(g: PDNGraph) -> dict:
    """DC operating point under the time-averaged load current.

    Average current per load is ``I_peak × duty``. Solves the same linear
    system as the transient case but without decap stamping or history
    sources. Returns raw per-node DC voltages on both meshes.
    """
    top0 = 0
    bot0 = g.n_top_nodes
    N = g.n_top_nodes + g.n_bot_nodes

    rows, cols, vals = _stamp_resistors(g, top0, bot0)
    G = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    pad_idx, pad_v = _clamp_indices(g, top0)
    free_mask = np.ones(N, dtype=bool)
    free_mask[pad_idx] = False
    free = np.where(free_mask)[0]

    G_ff = sp.csc_matrix(G[free, :][:, free])
    # Use ``pad_idx`` directly (not ``~free_mask``) so the column order of
    # G_fx matches the order of ``pad_v``. Boolean masking re-sorts by
    # natural index, which silently misaligns ``pad_v`` when Vdd and Vss
    # clamps differ.
    G_fx = G[free, :][:, pad_idx]
    rhs_const = np.asarray(G_fx @ pad_v).ravel()

    I = np.zeros(N)
    if g.n_loads > 0:
        load_vdd_idx = bot0 + g.load_pairs[:, 0].astype(int)
        load_vss_idx = bot0 + g.load_pairs[:, 1].astype(int)
        I_avg = g.loads[:, 0] * g.loads[:, 2]  # I_peak × duty per load
        np.subtract.at(I, load_vdd_idx, I_avg)
        np.add.at(I, load_vss_idx, I_avg)

    V = np.zeros(N, dtype=float)
    V[pad_idx] = pad_v
    V[free] = spla.spsolve(G_ff, I[free] - rhs_const)

    return {
        "V_top": V[top0:top0 + g.n_top_nodes].copy(),
        "V_bot": V[bot0:bot0 + g.n_bot_nodes].copy(),
    }
