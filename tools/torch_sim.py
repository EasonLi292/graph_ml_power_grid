"""Differentiable backward-Euler transient sim (torch twin of
``transient_solver.simulate`` + ``dataset_runner.run_one`` labels).

Purpose: exact Jacobians of peak droop w.r.t. every design knob —
``∂(droop)/∂(ww_edge)`` for all strap edges and ``∂/∂(C_decap_site)``
per decap site — via autograd through the time loop (the "adjoint
solve"). One backward pass returns the full Jacobian row; this is the
label generator for Sobolev (gradient-supervised) training and the
gradient oracle for physics-search baselines in the repair harness.

Faithfulness: same MNA stamping, same backward-Euler companion, same
initial condition, warmup rule, and measurement window as the numpy
pipeline, in float64. The one *extension* is per-site decap
(``C_sites``): the numpy solver only supports a scalar ``C_decap``;
with a uniform vector the two agree (validated by
``scripts/gen_jacobian_labels.py --validate``).

Grids are small (≤ ~350 nodes), so the free-node system is dense; we
LU-factor once (``torch.linalg.lu_factor``) and reuse across steps —
autograd flows through the factorization.
"""
from __future__ import annotations

import numpy as np
import torch

from .dataset_runner import SimConfig, _warmup_periods
from .grid_construction import PDNGraph


def _square_wave(t: torch.Tensor, freq: float, duty: float, phase: float):
    p = (t * freq + phase) % 1.0
    return (p < duty).to(t.dtype)


def peak_droop_torch(
    g: PDNGraph,
    ww_top: torch.Tensor,          # [n_top_edges] widths (differentiable)
    ww_bot: torch.Tensor,          # [n_bot_edges]
    C_sites: torch.Tensor,         # [n_decaps] per-site decap F (differentiable)
    Rsheet_top: float,
    Rsheet_bot: float,
    cfg: SimConfig | None = None,
    freq: float | None = None,
) -> torch.Tensor:
    """Per-load peak droop [n_loads], differentiable w.r.t. inputs."""
    cfg = cfg or SimConfig()
    dtype = torch.float64
    ww_top = ww_top.to(dtype); ww_bot = ww_bot.to(dtype)
    C_sites = C_sites.to(dtype)

    if freq is None:
        freq = float(g.loads[:, 1].max())   # global clock (broadcast per load)
    period = 1.0 / freq
    dt = period / cfg.steps_per_period
    warmup_periods = _warmup_periods(g.R_bot, float(C_sites.detach().max()),
                                     period, cfg)
    n_steps = int(np.round((warmup_periods + cfg.measure_periods)
                           * period / dt))

    top0, bot0 = 0, g.n_top_nodes
    N = g.n_top_nodes + g.n_bot_nodes

    # ---- conductance matrix (dense, differentiable) ----
    R_top = Rsheet_top * g.pitch_top / ww_top          # [Te]
    R_bot = Rsheet_bot * g.pitch_bot / ww_bot          # [Be]
    gv_via = torch.full((g.via_pairs.shape[0],), 1.0 / g.R_via, dtype=dtype)
    g_c = C_sites / dt                                  # [D]

    src = np.concatenate([g.top_edges[:, 0] + top0, g.bot_edges[:, 0] + bot0,
                          g.via_pairs[:, 0] + top0, g.decap_pairs[:, 0] + bot0])
    dst = np.concatenate([g.top_edges[:, 1] + top0, g.bot_edges[:, 1] + bot0,
                          g.via_pairs[:, 1] + bot0, g.decap_pairs[:, 1] + bot0])
    gv = torch.cat([1.0 / R_top, 1.0 / R_bot, gv_via, g_c])

    G = torch.zeros((N, N), dtype=dtype)
    si = torch.from_numpy(src.astype(np.int64))
    di = torch.from_numpy(dst.astype(np.int64))
    G.index_put_((si, si), gv, accumulate=True)
    G.index_put_((di, di), gv, accumulate=True)
    G.index_put_((si, di), -gv, accumulate=True)
    G.index_put_((di, si), -gv, accumulate=True)

    # ---- boundary partition ----
    pad_idx = np.concatenate([top0 + g.vdd_pad_top_idx.astype(int),
                              top0 + g.vss_pad_top_idx.astype(int)])
    pad_v_np = np.concatenate([np.full(g.vdd_pad_top_idx.size, g.Vdd),
                               np.zeros(g.vss_pad_top_idx.size)])
    free_mask = np.ones(N, dtype=bool); free_mask[pad_idx] = False
    free = np.where(free_mask)[0]
    free_t = torch.from_numpy(free)
    pad_t = torch.from_numpy(pad_idx.astype(np.int64))
    pad_v = torch.from_numpy(pad_v_np).to(dtype)

    G_ff = G[free_t][:, free_t]
    rhs_const = G[free_t][:, pad_t] @ pad_v
    LU, piv = torch.linalg.lu_factor(G_ff)

    # ---- waveforms and terminal indices ----
    t_arr = torch.arange(n_steps + 1, dtype=dtype) * dt
    lv = torch.from_numpy(bot0 + g.load_pairs[:, 0].astype(np.int64))
    ls = torch.from_numpy(bot0 + g.load_pairs[:, 1].astype(np.int64))
    I_waves = torch.stack([
        float(g.loads[k, 0]) * _square_wave(t_arr, float(g.loads[k, 1]),
                                            float(g.loads[k, 2]),
                                            float(g.loads[k, 3]))
        for k in range(g.n_loads)
    ])                                                       # [L, T+1]
    dv = torch.from_numpy(bot0 + g.decap_pairs[:, 0].astype(np.int64))
    ds = torch.from_numpy(bot0 + g.decap_pairs[:, 1].astype(np.int64))

    # ---- initial condition ----
    V = torch.zeros(N, dtype=dtype)
    V[top0:top0 + g.n_top_nodes] = torch.from_numpy(
        np.where(g.top_is_vdd == 1, g.Vdd, 0.0))
    V[bot0:bot0 + g.n_bot_nodes] = torch.from_numpy(
        np.where(g.bot_is_vdd == 1, g.Vdd, 0.0))
    V[pad_t] = pad_v

    warmup_steps = warmup_periods * cfg.steps_per_period
    lvi = torch.from_numpy(g.load_pairs[:, 0].astype(np.int64))  # bot-local
    lsi = torch.from_numpy(g.load_pairs[:, 1].astype(np.int64))
    min_dv = None
    for step in range(1, n_steps + 1):
        I = torch.zeros(N, dtype=dtype)
        I.index_put_((lv,), -I_waves[:, step], accumulate=True)
        I.index_put_((ls,), I_waves[:, step], accumulate=True)
        dV_prev = V[dv] - V[ds]
        I.index_put_((dv,), g_c * dV_prev, accumulate=True)
        I.index_put_((ds,), -g_c * dV_prev, accumulate=True)
        x = torch.linalg.lu_solve(LU, piv, (I[free_t] - rhs_const).unsqueeze(-1)).squeeze(-1)
        V = torch.zeros(N, dtype=dtype).index_put_((pad_t,), pad_v)
        V = V.index_put_((free_t,), x)
        if step > warmup_steps:
            dV_loads = V[bot0 + lvi] - V[bot0 + lsi]           # [L]
            min_dv = dV_loads if min_dv is None else torch.minimum(min_dv, dV_loads)

    return cfg.Vdd - min_dv                                    # [L] peak droop


def worst_droop_jacobian(
    g: PDNGraph,
    ww_top: np.ndarray,
    ww_bot: np.ndarray,
    C_sites: np.ndarray,
    Rsheet_top: float,
    Rsheet_bot: float,
    cfg: SimConfig | None = None,
    freq: float | None = None,
):
    """Exact ∂(worst-load peak droop)/∂(ln ww_e, ln C_site) via one backward.

    Log-parameter Jacobians are scale-free (volts per relative change),
    which is the natural target for Sobolev supervision.
    Returns (droop_loads [L], jac_lnww_top [Te], jac_lnww_bot [Be],
    jac_lnC [D], worst_idx).
    """
    wt = torch.tensor(ww_top, dtype=torch.float64, requires_grad=True)
    wb = torch.tensor(ww_bot, dtype=torch.float64, requires_grad=True)
    cs = torch.tensor(C_sites, dtype=torch.float64, requires_grad=True)
    droop = peak_droop_torch(g, wt, wb, cs, Rsheet_top, Rsheet_bot, cfg, freq)
    worst = droop.max()
    worst.backward()
    return (
        droop.detach().numpy(),
        (wt.grad * wt.detach()).numpy(),   # ∂/∂ln ww = ∂/∂ww · ww
        (wb.grad * wb.detach()).numpy(),
        (cs.grad * cs.detach()).numpy(),
        int(droop.argmax()),
    )
