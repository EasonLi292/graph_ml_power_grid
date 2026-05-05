"""Drive transient sims for an LHS batch and write per-split HDF5 datasets.

Each sample's simulation step count is fixed at 10 periods × 100 steps/period;
the first 2 periods are dropped as warm-up before computing peak droop.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable

import numpy as np

from .grid_construction import build_regular_pdn
from .transient_solver import simulate


N_PERIODS = 10
STEPS_PER_PERIOD = 100
WARMUP_PERIODS = 2


@dataclass
class SimConfig:
    n_periods: int = N_PERIODS
    steps_per_period: int = STEPS_PER_PERIOD
    warmup_periods: int = WARMUP_PERIODS
    Vdd: float = 1.0
    phase: float = 0.0


def run_one(p: dict[str, float], keep_full_traj: bool = False, cfg: SimConfig | None = None) -> dict:
    """Build a regular PDN with the given params, simulate, and extract targets."""
    cfg = cfg or SimConfig()
    period = 1.0 / p["freq"]
    dt = period / cfg.steps_per_period
    t_end = period * cfg.n_periods

    g = build_regular_pdn(
        Vdd=cfg.Vdd,
        R_top=p["R_top"],
        R_bot=p["R_bot"],
        R_via=p["R_via"],
        C_decap=p["C_decap"],
        I_peak=p["I_peak"],
        freq=p["freq"],
        duty=p["duty"],
        phase=cfg.phase,
    )
    res = simulate(g, t_end=t_end, dt=dt)

    n_steps = res["t"].size
    warmup = int(round(n_steps * (cfg.warmup_periods / cfg.n_periods)))
    V_bot_ss = res["V_bot"][warmup:]
    V_top_ss = res["V_top"][warmup:]

    peak_droop_bot = (cfg.Vdd - V_bot_ss.min(axis=0)).astype(np.float32)
    peak_droop_top = (cfg.Vdd - V_top_ss.min(axis=0)).astype(np.float32)

    out: dict = {
        "peak_droop_bot": peak_droop_bot,
        "peak_droop_top": peak_droop_top,
        "worst_node_idx": int(np.argmax(peak_droop_bot)),
        "worst_node_droop": float(peak_droop_bot.max()),
    }
    if keep_full_traj:
        out["V_bot_full"] = V_bot_ss.astype(np.float32)
        out["V_top_full"] = V_top_ss.astype(np.float32)
        out["t_full"] = res["t"][warmup:].astype(np.float32) - res["t"][warmup]
    return out


def _worker(args):
    p, keep, cfg = args
    return run_one(p, keep_full_traj=keep, cfg=cfg)


def run_many(
    samples: Iterable[dict[str, float]],
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
