"""Drive transient + DC sims for an LHS batch and collect labels.

Per-sample warmup length is set by the grid's slowest time constant
(``R_bot × C_decap``) rather than by a fixed number of periods. After
the warmup window the load runs for ``measure_periods`` periods at
100 steps/period; peak droop is taken over that window.

Each sample yields two label sets:

* ``peak_droop_*`` — dynamic peak droop from the transient solve
  (``Vdd − V.min`` over the measurement window).
* ``static_droop_*`` — DC IR drop under the time-averaged load current
  (``I_peak × duty`` per load). Independent of decap dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable

import numpy as np

from .grid_construction import build_regular_pdn
from .transient_solver import simulate, solve_static_dc


STEPS_PER_PERIOD = 100
MEASURE_PERIODS = 8
MIN_WARMUP_PERIODS = 2
SETTLING_TAU_FACTOR = 5.0  # 5τ ⇒ ≲1% residual on the initial-condition transient


@dataclass
class SimConfig:
    steps_per_period: int = STEPS_PER_PERIOD
    measure_periods: int = MEASURE_PERIODS
    min_warmup_periods: int = MIN_WARMUP_PERIODS
    settling_tau_factor: float = SETTLING_TAU_FACTOR
    Vdd: float = 1.0


def _warmup_periods(R_bot: float, C_decap: float, period: float, cfg: SimConfig) -> int:
    """Warmup periods s.t. warmup ≥ settling_tau_factor × τ_grid (R_bot × C_decap)."""
    tau = R_bot * C_decap
    n_from_tau = int(np.ceil(cfg.settling_tau_factor * tau / period))
    return max(cfg.min_warmup_periods, n_from_tau)


def run_one(p: dict, keep_full_traj: bool = False, cfg: SimConfig | None = None) -> dict:
    """Build a regular PDN with the given params, simulate, and extract targets.

    Expected keys in ``p``:

    * Global continuous: ``Rsheet_top``, ``Rsheet_bot``, ``wire_width``,
      ``R_via``, ``C_decap``, ``freq``.
    * Topology: ``pad_pattern``.
    * Per-load: ``loads`` (an ``[n_loads, 4]`` array of ``(I_peak, freq,
      duty, phase)``; ``freq`` is the global value broadcast to every
      load). If absent, ``build_regular_pdn`` broadcasts scalar defaults.
    """
    cfg = cfg or SimConfig()
    period = 1.0 / p["freq"]
    dt = period / cfg.steps_per_period

    g = build_regular_pdn(
        n_top=int(p.get("n_top", 4)),
        n_bot=int(p.get("n_bot", 7)),
        Vdd=cfg.Vdd,
        Rsheet_top=p["Rsheet_top"],
        Rsheet_bot=p["Rsheet_bot"],
        wire_width=p["wire_width"],
        R_via=p["R_via"],
        C_decap=p["C_decap"],
        freq=p["freq"],
        pad_pattern=p.get("pad_pattern", "corner"),
        loads=p.get("loads"),
    )

    warmup_periods = _warmup_periods(g.R_bot, p["C_decap"], period, cfg)
    n_periods = warmup_periods + cfg.measure_periods
    t_end = period * n_periods

    res = simulate(g, t_end=t_end, dt=dt)
    dc = solve_static_dc(g)

    warmup = warmup_periods * cfg.steps_per_period
    V_bot_ss = res["V_bot"][warmup:]

    peak_droop_bot = (cfg.Vdd - V_bot_ss.min(axis=0)).astype(np.float32)
    static_droop_bot = (cfg.Vdd - dc["V_bot"]).astype(np.float32)

    out: dict = {
        "peak_droop_bot":   peak_droop_bot,
        "static_droop_bot": static_droop_bot,
        "worst_node_idx":   int(np.argmax(peak_droop_bot)),
        "worst_node_droop": float(peak_droop_bot.max()),
    }
    if keep_full_traj:
        out["V_bot_full"] = V_bot_ss.astype(np.float32)
        out["t_full"] = res["t"][warmup:].astype(np.float32) - res["t"][warmup]
    return out


def _worker(args):
    p, keep, cfg = args
    return run_one(p, keep_full_traj=keep, cfg=cfg)


def run_many(
    samples: Iterable[dict],
    keep_traj_idx: set[int] | None = None,
    cfg: SimConfig | None = None,
    n_workers: int | None = None,
    chunksize: int = 16,
) -> list[dict]:
    """Run a list of parameter dicts in parallel; results returned in input order."""
    cfg = cfg or SimConfig()
    keep_traj_idx = keep_traj_idx or set()
    samples = list(samples)
    args = [(p, i in keep_traj_idx, cfg) for i, p in enumerate(samples)]

    if n_workers == 1:
        return [_worker(a) for a in args]

    with Pool(n_workers) as pool:
        return list(pool.imap(_worker, args, chunksize=chunksize))
