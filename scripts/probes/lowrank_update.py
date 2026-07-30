"""Does a modification let us avoid re-solving? Rank of the update decides.

Every entry of Z changes when you touch one wire — but the CHANGE is
low-rank. Conductance enters Y as g_e * a_e a_e^T with a_e = e_i - e_j, so

    one edge      ->  rank-1 update to Y
    k-edge strap  ->  rank-k
    global decap  ->  rank = number of decap sites  (NOT low rank)

For a rank-r update, Woodbury gives Z' exactly from Z with r solves:

    Z' = Z - Z A (S^-1 + A^T Z A)^-1 A^T Z

so the cost is r back-substitutions against an EXISTING factorization,
instead of a fresh factorization.

Measured here: exactness, cost vs refactorization, and the solve COUNT
against the transient simulator, which needs one solve per timestep.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/Users/eason/Desktop/graph_ml_power_grid")

from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import admittance, branch_system, knob_tensors
from tools.sampler import (FIXED_DUTY, FIXED_FREQ, FIXED_I_PEAK, FIXED_PHASE,
                           FIXED_R_VIA, FIXED_RSHEET_BOT, FIXED_RSHEET_TOP)

DT = torch.float64
SIM_STEPS = 8 * 100          # measure_periods * steps_per_period


def setup(nt, nb, ww=0.5, cd=2e-10):
    proto = build_regular_pdn(n_top=nt, n_bot=nb)
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (proto.n_loads, 1))
    g = build_regular_pdn(n_top=nt, n_bot=nb, Rsheet_top=FIXED_RSHEET_TOP,
                          Rsheet_bot=FIXED_RSHEET_BOT, wire_width=ww,
                          R_via=FIXED_R_VIA, C_decap=cd, freq=FIXED_FREQ,
                          loads=loads)
    s_ = branch_system(g)
    wt = torch.full((g.top_edges.shape[0],), ww, dtype=DT)
    wb = torch.full((g.bot_edges.shape[0],), ww, dtype=DT)
    return g, s_, wt, wb, torch.tensor(cd, dtype=DT)


def incidence(s_, edge_row):
    """a_e over free nodes for one resistive branch."""
    a = torch.zeros(s_.n_free, dtype=DT)
    ia = int(s_.free_of[edge_row[0]]); ib = int(s_.free_of[edge_row[1]])
    if ia >= 0:
        a[ia] += 1.0
    if ib >= 0:
        a[ib] -= 1.0
    return a


print(f"{'anchor':>9} {'n_free':>7} | {'rank-1 exact':>13} | "
      f"{'refactor ms':>12} {'rank1 upd ms':>13} {'speedup':>8} | {'decap rank':>11}")
for nt, nb in ((7, 13), (13, 13), (19, 19), (25, 25)):
    g, s_, wt, wb, cd = setup(nt, nb)
    R, C = knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                        FIXED_R_VIA)
    Y = admittance(s_, R, C, torch.zeros((), dtype=DT)).real.contiguous()

    # pick one BOT strap edge; its resistive-branch row index in r_edges
    te = g.top_edges.shape[0]
    j_edge = te + 5
    a = incidence(s_, s_.r_edges[j_edge])

    # modified: widen that edge by 25 % -> conductance rises
    wb2 = wb.clone(); wb2[5] *= 1.25
    R2, C2 = knob_tensors(g, wt, wb2, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                          FIXED_R_VIA)
    Y2 = admittance(s_, R2, C2, torch.zeros((), dtype=DT)).real.contiguous()
    dg = float(1.0 / R2[j_edge] - 1.0 / R[j_edge])

    # is the Y update really rank-1 with that incidence vector?
    resid = float((Y2 - Y - dg * torch.outer(a, a)).abs().max() / Y.abs().max())

    lu = torch.linalg.lu_factor(Y)
    Zex = torch.linalg.inv(Y2)

    def refactor():
        return torch.linalg.inv(Y2)

    def rank1():
        # u = Z a  via one back-substitution against the EXISTING factorization
        u = torch.linalg.lu_solve(*lu, a.unsqueeze(1))
        denom = 1.0 + dg * float(a @ u.squeeze(1))
        return u, dg / denom

    u, c = rank1()
    Z = torch.linalg.lu_solve(*lu, torch.eye(s_.n_free, dtype=DT))
    Zupd = Z - c * (u @ u.T)
    err = float((Zupd - Zex).abs().max() / Zex.abs().max())

    for f in (refactor, rank1):
        f()
    t0 = time.time()
    for _ in range(3):
        refactor()
    t_re = (time.time() - t0) / 3 * 1e3
    t0 = time.time()
    for _ in range(20):
        rank1()
    t_r1 = (time.time() - t0) / 20 * 1e3

    print(f"{f'({nt},{nb})':>9} {s_.n_free:>7} | {err:>13.2e} | "
          f"{t_re:>12.1f} {t_r1:>13.3f} {t_re/t_r1:>7.0f}x | "
          f"{s_.c_edges.shape[0]:>11}")
    if resid > 1e-12:
        print(f"          ! Y update is not rank-1 as modelled: resid {resid:.2e}")

print(f"\nrank-1 exact = ||Z_woodbury - Z_exact|| / ||Z_exact||")
print(f"decap rank = number of decap sites, i.e. the rank of a GLOBAL decap")
print(f"change -- not low rank, so Woodbury does not help there.")
print(f"\nsolve COUNT per proposal, which is what dominates at large N:")
print(f"  transient simulator : ~{SIM_STEPS} (one per timestep)")
print(f"  surrogate w/ rank-1 : 1 per modified edge")
