"""What does one forward query actually cost, end to end?

The plan "a proposer suggests a change, the forward model scores it
immediately" only pays off if scoring a change is cheaper than simulating
it. The model's inputs are impedance factors, and computing those requires
solving the circuit — so the honest unit of comparison is

    factors + model forward     vs     the exact simulator

measured on the same grids. Also reports the exact DC solve, which is the
cheap physics floor that already beats every model on forward accuracy.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/Users/eason/Desktop/graph_ml_power_grid")

from eason.impedance_attention_model import (ImpAttnConfig,
                                             ImpedanceAttentionRegressor)
from tools.dataset_runner import SimConfig, run_one
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (admittance, branch_system,
                                     dc_symmetric_factor, impedance_factors,
                                     knob_tensors, local_rc_features,
                                     node_features)
from tools.sampler import (FIXED_CONSTANTS, FIXED_DUTY, FIXED_FREQ,
                           FIXED_I_PEAK, FIXED_PHASE, FIXED_R_VIA,
                           FIXED_RSHEET_BOT, FIXED_RSHEET_TOP)

DT = torch.float64
B = 2 * np.pi * FIXED_FREQ
OM = torch.tensor([0.0, B, 5 * B], dtype=DT)
M = 16
ANCHORS = ((3, 7), (7, 13), (13, 13), (13, 25), (19, 19), (25, 25))


def timeit(fn, n=5):
    fn()
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1e3


print(f"{'anchor':>9} {'n_free':>7} {'transient':>10} {'DC solve':>9} "
      f"{'factors':>9} {'fwd dyn':>9} {'fwd simp':>9} | {'surrogate':>10} {'vs sim':>7}")
for nt, nb in ANCHORS:
    proto = build_regular_pdn(n_top=nt, n_bot=nb)
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (proto.n_loads, 1))
    g = build_regular_pdn(n_top=nt, n_bot=nb, Rsheet_top=FIXED_RSHEET_TOP,
                          Rsheet_bot=FIXED_RSHEET_BOT, wire_width=0.5,
                          R_via=FIXED_R_VIA, C_decap=2e-10, freq=FIXED_FREQ,
                          loads=loads)
    s_ = branch_system(g)
    wt = torch.full((g.top_edges.shape[0],), 0.5, dtype=DT)
    wb = torch.full((g.bot_edges.shape[0],), 0.5, dtype=DT)
    cd = torch.tensor(2e-10, dtype=DT)
    R, C = knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                        FIXED_R_VIA)

    p = dict(FIXED_CONSTANTS)
    p.update(n_top=nt, n_bot=nb, wire_width=0.5, C_decap=2e-10, loads=loads)
    t_sim = timeit(lambda: run_one(p, cfg=SimConfig()), n=3)

    def dc():
        Y = admittance(s_, R, C, torch.zeros((), dtype=DT))
        return torch.linalg.solve(Y, torch.ones(s_.n_free, dtype=Y.dtype))
    t_dc = timeit(dc)

    def fac():
        pp, ss = impedance_factors(s_, R, C, OM, m=M, n_power=2)
        f = dc_symmetric_factor(s_, R, C, m=M, n_power=2)
        return pp, ss, f
    t_fac = timeit(fac)

    pp, ss, fdc = fac()
    x = torch.cat([node_features(s_, torch.tensor(g.loads, dtype=DT)),
                   local_rc_features(s_, R, C)], -1)
    outs = {}
    for sc in ("dynamic_kernel", "simple"):
        torch.manual_seed(0)
        cfg = ImpAttnConfig(hidden_dim=64, heads=4, n_freq=3, m_factor=M,
                            score=sc, local_rc=True)
        mdl = ImpedanceAttentionRegressor(cfg, init_bias=-3.6).to(DT).eval()
        with torch.no_grad():
            outs[sc] = timeit(lambda: mdl(x, pp, ss, s_.n_elec, fdc=fdc))
    surro = t_fac + outs["dynamic_kernel"]
    print(f"{f'({nt},{nb})':>9} {s_.n_free:>7} {t_sim:>10.1f} {t_dc:>9.2f} "
          f"{t_fac:>9.1f} {outs['dynamic_kernel']:>9.1f} {outs['simple']:>9.1f} | "
          f"{surro:>10.1f} {surro/t_sim:>7.2f}x")

print("\nall times in ms, single query, CPU. 'surrogate' = factors + dynamic")
print("forward, i.e. what scoring ONE proposed change costs from scratch.")
print("'vs sim' > 1 means the surrogate is SLOWER than just simulating.")
print("\nNote the factor path builds a DENSE Y and does ~9 dense solves")
print("(3 frequencies x (1 + n_power) plus the DC symmetric factor), so it")
print("is O(N^3) while the transient solver is sparse.")
