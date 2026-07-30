"""Where does the incremental update actually pay, and where does it stop?

Three questions, all answered by measurement rather than by flop counts:

1. **End to end, per proposal** — proposal in, model prediction out. Against
   a full factor rebuild, and against the transient simulator. Total time,
   not the isolated Woodbury core: the core is microseconds and quoting it
   alone would be dishonest, since the subspace iteration and the model
   forward dominate.

2. **Rank** — the update replaces one O(N^3) factorization with r
   back-substitutions plus an r x r solve, so there is a rank at which it
   stops winning. That rank is measured here and written to
   ``docs/analysis/woodbury_crossover.json``, which
   ``FallbackPolicy.calibrated()`` reads. No hard-coded theoretical claim.

3. **Frequency count** — the base is factorized once per frequency, so the
   saving should scale with the number of frequencies. Verified, not assumed.

Three baselines are timed, because two of them are easy to confuse:

    refactor(default)  the shipped path: torch.linalg.solve per solve, i.e.
                       a fresh factorization for EVERY back-substitution
    refactor(LU)       build Y, one LU per frequency, reuse it within the
                       factor build -- the fair baseline the update must beat
    update             Woodbury against an immutable, already-factorized base

Quoting only the first would inflate the result. The headline number is
against ``refactor(LU)``.

    python scripts/probes/woodbury_crossover.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eason.impedance_attention_model import (ImpAttnConfig,
                                             ImpedanceAttentionRegressor)
from tools.dataset_runner import SimConfig, run_one
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (admittance, branch_system,
                                     dc_symmetric_factor, impedance_factors,
                                     local_rc_features, node_features)
from tools.incremental_impedance import DenseLUSolve, IncrementalImpedance
from tools.pdn_actions import ActionSpace
from tools.sampler import (FIXED_CONSTANTS, FIXED_DUTY, FIXED_FREQ,
                           FIXED_I_PEAK, FIXED_PHASE, FIXED_R_VIA,
                           FIXED_RSHEET_BOT, FIXED_RSHEET_TOP)

DT = torch.float64
BASE_W = 2 * np.pi * FIXED_FREQ
FREQ_GRID = [0.0, BASE_W, 5 * BASE_W, 25 * BASE_W, 125 * BASE_W]
M_FACTOR, N_POWER = 16, 2
ANCHORS = ((3, 7), (7, 13), (13, 13), (19, 19), (25, 25))


def timeit(fn, n=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


def setup(nt, nb, seed=0):
    rng = np.random.default_rng(seed)
    g0 = build_regular_pdn(n_top=nt, n_bot=nb)
    loads = np.tile([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]],
                    (g0.n_loads, 1))
    wt = rng.uniform(0.3, 0.8, g0.top_edges.shape[0])
    wb = rng.uniform(0.3, 0.8, g0.bot_edges.shape[0])
    cs = np.full(g0.n_decaps, 2e-10)
    g = build_regular_pdn(n_top=nt, n_bot=nb, Rsheet_top=FIXED_RSHEET_TOP,
                          Rsheet_bot=FIXED_RSHEET_BOT, wire_width=0.5,
                          R_via=FIXED_R_VIA, C_decap=2e-10, freq=FIXED_FREQ,
                          loads=loads, ww_top_edges=wt, ww_bot_edges=wb,
                          C_decap_sites=cs)
    sys_ = branch_system(g)
    acts = ActionSpace(g, sys_, torch.tensor(wt, dtype=DT),
                       torch.tensor(wb, dtype=DT), torch.tensor(cs, dtype=DT),
                       torch.tensor(loads, dtype=DT), FIXED_RSHEET_TOP,
                       FIXED_RSHEET_BOT, FIXED_R_VIA)
    return g, sys_, acts, loads


def make_model(n_freq):
    torch.manual_seed(0)
    cfg = ImpAttnConfig(hidden_dim=64, heads=4, n_freq=n_freq, m_factor=M_FACTOR,
                        score="dynamic_kernel", local_rc=True, invariant=True,
                        freq_norm=True)
    mdl = ImpedanceAttentionRegressor(cfg, init_bias=-3.6).to(DT).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return mdl, cfg


def score_paths(cand, sys_, om, mdl, cfg, inc):
    """The three per-proposal pipelines, each ending at a model prediction."""
    x = torch.cat([node_features(cand.sys, cand.loads),
                   local_rc_features(cand.sys, cand.R, cand.C)], -1)

    def finish(ops):
        p, s = impedance_factors(cand.sys, cand.R, cand.C, om, m=M_FACTOR,
                                 n_power=N_POWER, solvers=ops)
        fdc = dc_symmetric_factor(cand.sys, cand.R, cand.C, m=M_FACTOR,
                                  n_power=N_POWER, solver=ops[0])
        with torch.no_grad():
            return mdl(x, p, s, cand.sys.n_elec, fdc=fdc)

    def refactor_default():
        p, s = impedance_factors(cand.sys, cand.R, cand.C, om, m=M_FACTOR,
                                 n_power=N_POWER)
        fdc = dc_symmetric_factor(cand.sys, cand.R, cand.C, m=M_FACTOR,
                                  n_power=N_POWER)
        with torch.no_grad():
            return mdl(x, p, s, cand.sys.n_elec, fdc=fdc)

    def refactor_lu():
        return finish([DenseLUSolve(admittance(cand.sys, cand.R, cand.C, w))
                       for w in om])

    def update():
        ops, _ = inc.solvers_for(cand.proposal, R_new=cand.R, C_new=cand.C,
                                 sys_new=cand.sys)
        return finish(ops)

    return refactor_default, refactor_lu, update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("docs/analysis/woodbury_crossover.json"))
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    out = {"anchors": [], "rank_sweep": [], "freq_sweep": []}

    # ---- 1. end to end, per proposal, by topology -----------------------
    om3 = torch.tensor(FREQ_GRID[:3], dtype=DT)
    mdl3, cfg3 = make_model(3)
    print("1. ONE proposal, end to end: proposal -> model prediction  (ms)")
    print(f"{'anchor':>9} {'n_free':>7} | {'refac(dflt)':>12} {'refac(LU)':>10} "
          f"{'update':>8} | {'vs refac(LU)':>13} | {'simulator':>10} {'vs sim':>8}")
    for nt, nb in ANCHORS:
        g, sys_, acts, loads = setup(nt, nb)
        inc = IncrementalImpedance(sys_, acts.R, acts.C, om3)
        cand = acts.wire_width(1, [3], 1.4)              # rank 1
        f_def, f_lu, f_up = score_paths(cand, sys_, om3, mdl3, cfg3, inc)
        t_def, t_lu, t_up = (timeit(f, args.reps) for f in (f_def, f_lu, f_up))
        p = dict(FIXED_CONSTANTS)
        p.update(n_top=nt, n_bot=nb, wire_width=0.5, C_decap=2e-10, loads=loads,
                 ww_top_edges=acts.wt.numpy(), ww_bot_edges=acts.wb.numpy())
        t_sim = timeit(lambda: run_one(p, cfg=SimConfig()), max(1, args.reps - 1))
        print(f"{f'({nt},{nb})':>9} {sys_.n_free:>7} | {t_def:>12.1f} "
              f"{t_lu:>10.1f} {t_up:>8.1f} | {t_lu / t_up:>12.2f}x | "
              f"{t_sim:>10.1f} {t_sim / t_up:>7.2f}x")
        out["anchors"].append(dict(anchor=[nt, nb], n_free=sys_.n_free,
                                   refactor_default_ms=t_def, refactor_lu_ms=t_lu,
                                   update_ms=t_up, simulator_ms=t_sim))
    print("  'vs sim' > 1 means the surrogate is FASTER than simulating.")

    # ---- 2. rank sweep -> the calibrated threshold ----------------------
    print("\n2. rank sweep at (13,13) and (25,25): where does the update stop paying?")
    print(f"{'anchor':>9} {'rank':>6} {'refac(LU)':>10} {'update':>8} "
          f"{'speedup':>8}  {'verdict':>9}")
    crossover = {}
    for nt, nb in ((13, 13), (25, 25)):
        g, sys_, acts, loads = setup(nt, nb)
        inc = IncrementalImpedance(sys_, acts.R, acts.C, om3)
        inc.policy.max_rank = 10 ** 9        # measure past the guard
        inc.policy.max_branch_frac = 1.0
        n_bot_e = acts.n_bot_e
        prev_ok, cross = None, None
        for r in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            if r > n_bot_e:
                break
            cand = acts.wire_width(1, np.arange(r), 1.3)
            f_def, f_lu, f_up = score_paths(cand, sys_, om3, mdl3, cfg3, inc)
            t_lu, t_up = timeit(f_lu, args.reps), timeit(f_up, args.reps)
            sp = t_lu / t_up
            verdict = "wins" if sp > 1.0 else "LOSES"
            if sp <= 1.0 and cross is None:
                cross = r
            print(f"{f'({nt},{nb})':>9} {r:>6} {t_lu:>10.1f} {t_up:>8.1f} "
                  f"{sp:>7.2f}x  {verdict:>9}")
            out["rank_sweep"].append(dict(anchor=[nt, nb], rank=r,
                                          refactor_lu_ms=t_lu, update_ms=t_up,
                                          speedup=sp))
        crossover[f"{nt},{nb}"] = cross
        print(f"          -> crossover at rank {cross if cross else '> tested range'}")

    # ---- 3. frequency count --------------------------------------------
    print("\n3. frequency count at (13,13), rank 1  (the base is factorized once per w)")
    print(f"{'n_freq':>7} {'refac(LU)':>10} {'update':>8} {'speedup':>8}")
    g, sys_, acts, loads = setup(13, 13)
    for nf in (1, 2, 3, 5):
        om = torch.tensor(FREQ_GRID[:nf], dtype=DT)
        mdl, cfg = make_model(nf)
        inc = IncrementalImpedance(sys_, acts.R, acts.C, om)
        cand = acts.wire_width(1, [3], 1.4)
        _, f_lu, f_up = score_paths(cand, sys_, om, mdl, cfg, inc)
        t_lu, t_up = timeit(f_lu, args.reps), timeit(f_up, args.reps)
        print(f"{nf:>7} {t_lu:>10.1f} {t_up:>8.1f} {t_lu / t_up:>7.2f}x")
        out["freq_sweep"].append(dict(n_freq=nf, refactor_lu_ms=t_lu,
                                      update_ms=t_up, speedup=t_lu / t_up))

    # ---- 4. the case this was actually built for ------------------------
    # The paired dataset is ONE base circuit plus K perturbations of it, so a
    # factor-cache build is exactly the shared-base workload Woodbury wants.
    # Model forward is excluded here: cache construction stops at the factors.
    print("\n4. factor cache for one paired GROUP: 1 base + 8 perturbations")
    print(f"{'anchor':>9} {'n_free':>7} | {'refactor':>9} {'update':>8} "
          f"{'speedup':>8} | {'per group, s':>13}")
    for nt, nb in ANCHORS[1:]:
        g, sys_, acts, loads = setup(nt, nb)
        K = 8

        def facs(ops, cand):
            impedance_factors(cand.sys, cand.R, cand.C, om3, m=M_FACTOR,
                              n_power=N_POWER, solvers=ops)
            dc_symmetric_factor(cand.sys, cand.R, cand.C, m=M_FACTOR,
                                n_power=N_POWER, solver=ops[0])

        cands = [acts.wire_width(1, np.arange(i % 4 + 1), 1.0 + 0.1 * i)
                 for i in range(K)]

        def group_refactor():
            for c in [acts.wire_width(1, [0], 1.0)] + cands:
                facs([DenseLUSolve(admittance(c.sys, c.R, c.C, w))
                      for w in om3], c)

        def group_update():
            inc = IncrementalImpedance(sys_, acts.R, acts.C, om3)
            facs(inc.solvers, acts.wire_width(1, [0], 1.0))     # the base
            for c in cands:
                ops, _ = inc.solvers_for(c.proposal, R_new=c.R, C_new=c.C,
                                         sys_new=c.sys)
                facs(ops, c)

        t_r, t_u = timeit(group_refactor, 2), timeit(group_update, 2)
        print(f"{f'({nt},{nb})':>9} {sys_.n_free:>7} | {t_r:>9.0f} {t_u:>8.0f} "
              f"{t_r / t_u:>7.2f}x | {t_r / 1e3:>6.2f} -> {t_u / 1e3:<5.2f}")
        out.setdefault("group_cache", []).append(
            dict(anchor=[nt, nb], n_free=sys_.n_free, refactor_ms=t_r,
                 update_ms=t_u, speedup=t_r / t_u, k_perturbations=K))

    # ---- recommendation -------------------------------------------------
    seen = [v for v in crossover.values() if v]
    rec = (min(seen) // 2) if seen else 128
    rec = max(8, int(rec))
    out["crossover_by_anchor"] = crossover
    out["recommended_max_rank"] = rec
    out["note"] = ("recommended_max_rank is half the smallest MEASURED "
                   "crossover rank, so the guard trips before the update "
                   "becomes the slower option rather than at the exact "
                   "break-even point")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nmeasured crossover by anchor: {crossover}")
    print(f"recommended FallbackPolicy.max_rank = {rec}  -> {args.out}")


if __name__ == "__main__":
    main()
