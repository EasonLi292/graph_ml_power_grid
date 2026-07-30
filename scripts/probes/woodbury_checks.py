"""Is the incremental impedance update exact, and where does it stop being safe?

Ten checks, each comparing the UPDATE path against a complete
refactorization of the same modified circuit. The two paths are handed the
same modification by construction (``tools.pdn_actions.Candidate`` carries
both representations), so a disagreement is a real defect, not two
descriptions drifting apart.

Two error scales are kept apart on purpose:

* **update vs refactor** — should be floating point (~1e-15). Woodbury is
  an identity, not an approximation; anything above ~1e-12 is a bug.
* **factors vs the true Z** — the rank-m sketch error, which the base path
  already pays and the update neither adds to nor removes. Reported
  separately in check 1 so the first number is never quietly credited with
  the second.

    python scripts/probes/woodbury_checks.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eason.impedance_attention_model import (ImpAttnConfig,
                                             ImpedanceAttentionRegressor)
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (admittance, branch_system,
                                     dc_symmetric_factor, impedance_factors,
                                     invariant_channels, local_rc_features,
                                     node_features)
from tools.incremental_impedance import (DenseLUSolve, FallbackPolicy,
                                         IncrementalImpedance, WoodburySolve,
                                         gather_rows, incidence_matrix)
from tools.pdn_actions import ActionSpace, column_groups, wire_section
from tools.sampler import (FIXED_DUTY, FIXED_FREQ, FIXED_I_PEAK, FIXED_PHASE,
                           FIXED_R_VIA, FIXED_RSHEET_BOT, FIXED_RSHEET_TOP)

DT = torch.float64
CDT = torch.complex128
BASE_W = 2 * np.pi * FIXED_FREQ
OMEGAS = torch.tensor([0.0, BASE_W, 5 * BASE_W], dtype=DT)
M_FACTOR, N_POWER = 16, 2
EXACT_TOL = 1e-11          # update vs refactor; identity, so this is tight
# Anything downstream of the QR is held to a looser bar, and the reason is
# not the update: n_power subspace iteration deliberately collapses X toward
# the dominant subspace, so X's columns are near-dependent and qr() rotates
# Qr by much more than the 1e-15 perturbation that produced it. The model
# then puts that through LayerNorm and log10. Measured amplification is
# ~1e-15 -> ~1e-12 at the channels and ~1e-11 at the output; the bar below
# leaves margin without pretending the whole chain is bitwise reproducible.
FEATURE_TOL = 1e-9

_results = []


def report(name, err, tol=EXACT_TOL, note=""):
    ok = bool(np.isfinite(err)) and err <= tol
    _results.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] {name:<46} {err:>10.3e}"
          f"{'  ' + note if note else ''}")
    return ok


# ---------------------------------------------------------------- fixtures

def make_grid(nt, nb, seed=0, decap_fill=1.0):
    """Base circuit with heterogeneous widths and a partly-populated decap map.

    ``decap_fill < 1`` leaves some candidate slots EMPTY, which is what makes
    "add a decap here" a distinct action from "resize that one".
    """
    rng = np.random.default_rng(seed)
    g0 = build_regular_pdn(n_top=nt, n_bot=nb)
    loads = np.tile([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]],
                    (g0.n_loads, 1))
    wt = rng.uniform(0.3, 0.8, g0.top_edges.shape[0])
    wb = rng.uniform(0.3, 0.8, g0.bot_edges.shape[0])
    cs = np.full(g0.n_decaps, 2e-10)
    if decap_fill < 1.0:
        empty = rng.choice(g0.n_decaps,
                           size=int(g0.n_decaps * (1 - decap_fill)),
                           replace=False)
        cs[empty] = 0.0
    g = build_regular_pdn(
        n_top=nt, n_bot=nb, Rsheet_top=FIXED_RSHEET_TOP,
        Rsheet_bot=FIXED_RSHEET_BOT, wire_width=0.5, R_via=FIXED_R_VIA,
        C_decap=2e-10, freq=FIXED_FREQ, loads=loads,
        ww_top_edges=wt, ww_bot_edges=wb, C_decap_sites=cs)
    sys_ = branch_system(g)
    acts = ActionSpace(g, sys_, torch.tensor(wt, dtype=DT),
                       torch.tensor(wb, dtype=DT), torch.tensor(cs, dtype=DT),
                       torch.tensor(loads, dtype=DT), FIXED_RSHEET_TOP,
                       FIXED_RSHEET_BOT, FIXED_R_VIA)
    return g, sys_, acts, rng


def make_model(seed=0, score="dynamic_kernel"):
    torch.manual_seed(seed)
    cfg = ImpAttnConfig(hidden_dim=64, heads=4, n_freq=len(OMEGAS),
                        m_factor=M_FACTOR, score=score, local_rc=True,
                        invariant=True, freq_norm=True)
    mdl = ImpedanceAttentionRegressor(cfg, init_bias=-3.6).to(DT).eval()
    # The regressor ships with a ZERO final decoder weight so training starts
    # at the mean-predictor baseline. Left that way it is a constant function:
    # every output check would compare two constants and pass without ever
    # touching the factors, and every gradient would be exactly zero. Give it
    # a real head so the checks below can actually fail.
    torch.nn.init.normal_(mdl.decoder[-1].weight, std=0.5)
    for p in mdl.parameters():
        p.requires_grad_(False)
    return mdl, cfg


def factors_via(cand, inc, use_update: bool):
    """(p, s, fdc) for a candidate, by update or by refactorization."""
    if use_update:
        ops, info = inc.solvers_for(cand.proposal, R_new=cand.R, C_new=cand.C,
                                  sys_new=cand.sys)
    else:
        ops = [DenseLUSolve(admittance(cand.sys, cand.R, cand.C, w))
               for w in OMEGAS]
        info = {"mode": "refactor(reference)"}
    p, s = impedance_factors(cand.sys, cand.R, cand.C, OMEGAS, m=M_FACTOR,
                             n_power=N_POWER, solvers=ops)
    fdc = dc_symmetric_factor(cand.sys, cand.R, cand.C, m=M_FACTOR,
                              n_power=N_POWER, solver=ops[0])
    return p, s, fdc, info


def model_out(mdl, cfg, cand, p, s, fdc):
    x = node_features(cand.sys, cand.loads)
    if cfg.local_rc:
        x = torch.cat([x, local_rc_features(cand.sys, cand.R, cand.C)], -1)
    return mdl(x, p, s, cand.sys.n_elec, fdc=fdc)


def relerr(a, b):
    d = (a - b).abs().max()
    scale = b.abs().max().clamp_min(1e-300)
    return float(d / scale)


def candidate_set(acts, g, nb, rng):
    """One candidate per supported action family."""
    gb = column_groups(g.bot_edges, nb)
    gt = column_groups(g.top_edges, g.n_top)
    empty = np.nonzero(acts.C_sites.numpy() == 0.0)[0]
    full = np.nonzero(acts.C_sites.numpy() > 0.0)[0]
    out = [
        ("one wire width", acts.wire_width(1, [3], 1.4)),
        ("several wire changes", acts.wire_width(1, [1, 5, 9], 0.8)),
        ("local wire section", acts.wire_width(1, wire_section(gb[0], rng), 1.5)),
        ("full bot strap", acts.wire_width(1, gb[1], 0.6)),
        ("full top strap (rail)", acts.wire_width(0, gt[0], 1.7)),
        ("via resistance", acts.via_resistance([0, 1], 2.0)),
        ("remove an edge", acts.remove_resistive_edge(acts.n_top_e + 4)),
        ("add an edge", acts.add_resistive_edge(
            int(acts.sys.r_edges[0, 0]), int(acts.sys.r_edges[7, 1]),
            torch.tensor(1.0, dtype=DT))),
        ("resize one decap", acts.decap_resize(int(full[0]), 3.0)),
        ("remove one decap", acts.decap_remove(int(full[1]))),
        ("add one decap", acts.decap_add(int(empty[0]),
                                         torch.tensor(2e-10, dtype=DT))),
        ("move one decap", acts.decap_move(int(full[2]), int(empty[1]))),
        ("several local decaps", acts.decap_multi(full[:3], 1.5)),
        ("redistribute decap", acts.decap_redistribute(full[:2], full[-2:], 0.5)),
        ("global decap scaling", acts.decap_global(2.0)),
        ("load current (rank 0)", acts.load_scale([0], 2.0)),
        ("load frequency (rank 0)", acts.load_freq(1.5)),
    ]
    return out


# ------------------------------------------------------------------ checks

def check_1_inverse_action(acts, sys_, inc):
    print("\n1. updated inverse action  Z' B")
    g_, nb = acts.g, acts.g.n_bot
    B = torch.randn(sys_.n_free, 8, dtype=DT).to(CDT)
    worst = 0.0
    for name, cand in candidate_set(acts, g_, nb, np.random.default_rng(1)):
        ops, info = inc.solvers_for(cand.proposal, R_new=cand.R, C_new=cand.C,
                                  sys_new=cand.sys)
        e = 0.0
        for f, w in enumerate(OMEGAS):
            ex = torch.linalg.solve(admittance(cand.sys, cand.R, cand.C, w), B)
            e = max(e, relerr(ops[f].solve(B), ex))
        report(f"{name} (rank {cand.rank}, {info['mode']})", e)
        worst = max(worst, e)

    # the OTHER error scale, so the number above is never credited with it
    Y = admittance(sys_, acts.R, acts.C, OMEGAS[0])
    Z = torch.linalg.inv(Y)
    p, s = impedance_factors(sys_, acts.R, acts.C, OMEGAS, m=M_FACTOR,
                             n_power=N_POWER)
    n_f = sys_.n_free
    live = sys_.free_of >= 0
    idx = sys_.free_of[live]
    Zh = (p[:sys_.n_elec][live][:, 0] @ s[:sys_.n_elec][live][:, 0].T)
    Zt = Z.real[idx][:, idx]
    print(f"   -- separately, rank-{M_FACTOR} SKETCH error of the factors "
          f"vs true Z: {relerr(Zh, Zt):.3e}")
    print( "      (the base path already pays this; the update neither adds "
           "to nor removes it)")


def check_2_invariant_channels(acts, inc):
    print("\n2. updated invariant channels  Re Z, Im Z")
    for name, cand in candidate_set(acts, acts.g, acts.g.n_bot,
                                    np.random.default_rng(2))[:8]:
        pu, su, _, _ = factors_via(cand, inc, True)
        pr, sr, _, _ = factors_via(cand, inc, False)
        cu = invariant_channels(pu, su, OMEGAS)
        cr = invariant_channels(pr, sr, OMEGAS)
        e = 0.0
        for (Au, Bu, nm), (Ar, Br, _) in zip(cu, cr):
            e = max(e, relerr(Au @ Bu.T, Ar @ Br.T))   # [N,N]: TEST PATH ONLY
        report(f"{name}: Re/Im Z channels", e, tol=FEATURE_TOL)


def check_3_normalized_features(acts, inc):
    print("\n3. updated normalized factor features")
    mdl, cfg = make_model()
    for name, cand in candidate_set(acts, acts.g, acts.g.n_bot,
                                    np.random.default_rng(3))[:8]:
        pu, su, _, _ = factors_via(cand, inc, True)
        pr, sr, _, _ = factors_via(cand, inc, False)
        x = node_features(cand.sys, cand.loads)
        x = torch.cat([x, local_rc_features(cand.sys, cand.R, cand.C)], -1)
        hu, chu = mdl.embed_invariant(x, pu, su, cand.sys.n_elec)
        hr, chr_ = mdl.embed_invariant(x, pr, sr, cand.sys.n_elec)
        e = max(relerr(hu, hr),
                max(max(relerr(a, b), relerr(c, d))
                    for (a, c, _), (b, d, _) in zip(chu, chr_)))
        report(f"{name}: frob-normalized channels + hstate", e,
               tol=FEATURE_TOL)


def check_4_5_model_output(acts, inc):
    print("\n4/5. model output and per-load CHANGE, identical weights")
    mdl, cfg = make_model()
    pb, sb, fb, _ = factors_via(_identity(acts), inc, False)
    y0 = model_out(mdl, cfg, _identity(acts), pb, sb, fb)
    for name, cand in candidate_set(acts, acts.g, acts.g.n_bot,
                                    np.random.default_rng(4)):
        pu, su, fu, info = factors_via(cand, inc, True)
        pr, sr, fr, _ = factors_via(cand, inc, False)
        yu = model_out(mdl, cfg, cand, pu, su, fu)
        yr = model_out(mdl, cfg, cand, pr, sr, fr)
        e_out = relerr(yu, yr)
        e_d = relerr(yu - y0, yr - y0)
        report(f"{name}: output / per-load change", max(e_out, e_d),
               tol=FEATURE_TOL, note=f"d-err {e_d:.1e}")


def _identity(acts):
    """The unmodified base as a Candidate, for change comparisons."""
    return acts.wire_width(1, [0], 1.0)


def check_6_gradients(acts, inc):
    print("\n6. gradients w.r.t. wire width and capacitance")
    mdl, cfg = make_model()
    g_, sys_ = acts.g, acts.sys
    full = np.nonzero(acts.C_sites.numpy() > 0.0)[0]

    def out_for(t, kind, at_update):
        wb2 = acts.wb.clone()
        cs2 = acts.C_sites.clone()
        if kind == "width":
            wb2 = wb2.clone(); wb2[3] = wb2[3] * t
        else:
            cs2 = cs2.clone(); cs2[int(full[0])] = cs2[int(full[0])] * t
        a2 = ActionSpace(g_, sys_, acts.wt, wb2, cs2, acts.loads,
                         FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
        cand = (acts.wire_width(1, [3], t) if kind == "width"
                else acts.decap_resize(int(full[0]), t))
        p, s, fdc, _ = factors_via(cand, inc, at_update)
        return model_out(mdl, cfg, cand, p, s, fdc).sum()

    for kind in ("width", "cap"):
        for t0, tag in ((1.3, "away from zero"), (1.0, "AT zero change")):
            t = torch.tensor(t0, dtype=DT, requires_grad=True)
            gu = torch.autograd.grad(out_for(t, kind, True), t)[0]
            h = 1e-6
            fd = (float(out_for(torch.tensor(t0 + h, dtype=DT), kind, False))
                  - float(out_for(torch.tensor(t0 - h, dtype=DT), kind, False))) / (2 * h)
            denom = max(abs(fd), 1e-12)
            e = abs(float(gu) - fd) / denom
            tol = 2e-4 if kind == "width" else 5e-3      # central-difference floor
            report(f"d(out)/d({kind}) {tag}", e, tol=tol,
                   note=f"autograd {float(gu):+.4e} vs FD {fd:+.4e}")


def check_7_random_circuits(inc_unused):
    print("\n7. several random circuits x modification magnitudes")
    B_cols = 6
    worst = 0.0
    for seed, (nt, nb) in enumerate(((3, 7), (5, 13), (7, 13))):
        g_, sys_, acts, rng = make_grid(nt, nb, seed=seed + 10, decap_fill=0.7)
        inc = IncrementalImpedance(sys_, acts.R, acts.C, OMEGAS)
        B = torch.randn(sys_.n_free, B_cols, dtype=DT).to(CDT)
        gb = column_groups(g_.bot_edges, nb)
        e_a = 0.0
        for mag in (0.25, 0.5, 0.9, 1.0, 1.1, 2.0, 8.0):
            for cand in (acts.wire_width(1, [2], mag),
                         acts.wire_width(1, gb[0], mag),
                         acts.decap_resize(
                             int(np.nonzero(acts.C_sites.numpy() > 0)[0][0]), mag)):
                ops, _ = inc.solvers_for(cand.proposal, R_new=cand.R, C_new=cand.C,
                                  sys_new=cand.sys)
                for f, w in enumerate(OMEGAS):
                    ex = torch.linalg.solve(
                        admittance(cand.sys, cand.R, cand.C, w), B)
                    e_a = max(e_a, relerr(ops[f].solve(B), ex))
        report(f"({nt},{nb}) n_free={sys_.n_free}, 7 magnitudes x 3 families", e_a)
        worst = max(worst, e_a)


def check_8_multi_edge(acts, sys_, inc):
    print("\n8. multi-edge updates, including INTERACTING adjacent edges")
    B = torch.randn(sys_.n_free, 5, dtype=DT).to(CDT)
    # adjacency in the update sense = sharing a node, so the rank-r core is
    # NOT diagonal and the changes genuinely interact
    r_e = sys_.r_edges.numpy()
    shared, j0 = None, None
    for j in range(acts.n_top_e, acts.n_top_e + acts.n_bot_e):
        nb_ = [k for k in range(acts.n_top_e, acts.n_top_e + acts.n_bot_e)
               if k != j and len(set(r_e[j]) & set(r_e[k]))]
        if len(nb_) >= 2:
            j0, shared = j, nb_[:2]
            break
    n_b = acts.n_bot_e
    groups = [("2 adjacent (share a node)", [j0, shared[0]]),
              ("3 adjacent chain", [j0] + shared),
              ("8 scattered", list(range(acts.n_top_e,
                                         acts.n_top_e + n_b, max(1, n_b // 8)))[:8])]
    for name, rows in groups:
        idx = np.array(rows) - acts.n_top_e
        cand = acts.wire_width(1, idx, 1.6)
        ops, info = inc.solvers_for(cand.proposal, R_new=cand.R, C_new=cand.C,
                                  sys_new=cand.sys)
        e = 0.0
        for f, w in enumerate(OMEGAS):
            ex = torch.linalg.solve(admittance(cand.sys, cand.R, cand.C, w), B)
            e = max(e, relerr(ops[f].solve(B), ex))
        off = _core_offdiag(inc, cand)
        report(f"{name} (rank {cand.rank})", e,
               note=f"core off-diag/diag {off:.2f}")


def _core_offdiag(inc, cand):
    """How far the r x r core is from diagonal -- i.e. do the edges interact."""
    ia = torch.tensor([inc.sys.free_of[c.u] for c in cand.proposal.changes])
    ib = torch.tensor([inc.sys.free_of[c.v] for c in cand.proposal.changes])
    U = inc.solvers[0].solve(incidence_matrix(ia, ib, inc.solvers[0].n, CDT))
    K = gather_rows(U, ia, ib).abs()
    d = torch.diagonal(K)
    return float((K - torch.diag(d)).abs().max() / d.abs().max())


def check_9_add_remove_move(acts, sys_, inc):
    print("\n9. add / remove / move")
    B = torch.randn(sys_.n_free, 5, dtype=DT).to(CDT)
    full = np.nonzero(acts.C_sites.numpy() > 0.0)[0]
    empty = np.nonzero(acts.C_sites.numpy() == 0.0)[0]
    cases = [
        ("remove resistive edge", acts.remove_resistive_edge(acts.n_top_e + 6)),
        ("add resistive edge", acts.add_resistive_edge(
            int(sys_.r_edges[2, 0]), int(sys_.r_edges[9, 1]),
            torch.tensor(0.7, dtype=DT))),
        ("remove decap", acts.decap_remove(int(full[0]))),
        ("add decap", acts.decap_add(int(empty[0]), torch.tensor(3e-10, dtype=DT))),
        ("move decap", acts.decap_move(int(full[1]), int(empty[1]))),
    ]
    for name, cand in cases:
        ops, info = inc.solvers_for(cand.proposal, R_new=cand.R, C_new=cand.C,
                                  sys_new=cand.sys)
        e = 0.0
        for f, w in enumerate(OMEGAS):
            ex = torch.linalg.solve(admittance(cand.sys, cand.R, cand.C, w), B)
            e = max(e, relerr(ops[f].solve(B), ex))
        report(f"{name} (rank {cand.rank})", e)

    # a decap action must leave the DC operator IDENTICAL, not merely close
    cand = acts.decap_resize(int(full[0]), 5.0)
    ops, _ = inc.solvers_for(cand.proposal, R_new=cand.R, C_new=cand.C,
                                  sys_new=cand.sys)
    same = ops[0] is inc.solvers[0]
    _results.append(same)
    print(f"  [{'ok' if same else 'FAIL'}] decap change does no DC update at all"
          f"{'':<9} {'(operator reused)' if same else '(rebuilt!)'}")


def check_10_fallback_boundary(acts, sys_, inc):
    print("\n10. behaviour at the fallback boundary")
    # (a) sweep a conductance delta toward the value that makes the core
    #     singular: K = 1 + d a^T Z a = 0  =>  d* = -1 / R_eff(edge)
    j = acts.n_top_e + 5
    u, v = int(sys_.r_edges[j, 0]), int(sys_.r_edges[j, 1])
    ia = torch.tensor([sys_.free_of[u]]); ib = torch.tensor([sys_.free_of[v]])
    U = inc.solvers[0].solve(incidence_matrix(ia, ib, sys_.n_free, CDT))
    reff = float(gather_rows(U, ia, ib).real[0, 0])
    d_star = -1.0 / reff
    print(f"   edge {j}: R_eff={reff:.4f}, core is singular at d*={d_star:.4f}")
    print(f"   this is where the rcond THRESHOLD comes from: solve error "
          f"tracks eps/rcond,")
    print(f"   so the default rcond_min=1e-8 buys ~1e-8 relative accuracy.")
    print(f"   {'d / d*':>10} {'core rcond':>12} {'mode':>10} {'rel err':>11} "
          f"{'err*rcond':>11}")
    B = torch.randn(sys_.n_free, 4, dtype=DT).to(CDT)
    from tools.incremental_impedance import BranchChange, KIND_G, Proposal
    g_old = 1.0 / acts.R[j]
    monotone = []
    for frac in (0.5, 0.9, 0.99, 0.9999, 1.0 - 1e-8, 1.0):
        d = d_star * frac
        prop = Proposal(changes=[BranchChange(u, v, KIND_G, old=g_old,
                                              new=g_old + d)])
        R2 = acts.R.clone(); R2[j] = 1.0 / (g_old + d)
        ops, info = inc.solvers_for(prop, R_new=R2, C_new=acts.C)
        rc = info["rcond"][0] if info["rcond"] else float("nan")
        ex = torch.linalg.solve(admittance(sys_, R2, acts.C, OMEGAS[0]), B)
        e = relerr(ops[0].solve(B), ex)
        print(f"   {frac:>10.8f} {rc:>12.3e} {info['mode']:>10} {e:>11.3e} "
              f"{e * rc:>11.3e}")
        monotone.append((rc, e, info["mode"]))
    # the guard must engage before accuracy is lost, and only then
    ok = (all(m == "update" for _, _, m in monotone[:-1])
          and monotone[-1][2] == "refactor"
          and all(e < 1e-7 for _, e, m in monotone if m == "update"))
    _results.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] every accepted update stayed under "
          f"1e-7; the singular one fell back")

    # (b) a modification that genuinely disconnects: strip every branch
    #     incident to one interior node
    deg = np.zeros(sys_.n_elec, dtype=int)
    for a, b in sys_.r_edges.numpy():
        deg[a] += 1; deg[b] += 1
    cand_nodes = [n for n in range(sys_.n_elec)
                  if sys_.free_of[n] >= 0 and 0 < deg[n] <= 4]
    node = cand_nodes[len(cand_nodes) // 2]
    rows = [j for j, (a, b) in enumerate(sys_.r_edges.numpy())
            if a == node or b == node]
    ch = [BranchChange(int(sys_.r_edges[j, 0]), int(sys_.r_edges[j, 1]),
                       KIND_G, old=1.0 / acts.R[j],
                       new=torch.zeros((), dtype=DT)) for j in rows]
    keep = torch.ones(acts.R.shape[0], dtype=torch.bool)
    keep[rows] = False
    from dataclasses import replace as _replace
    sys2 = _replace(sys_, r_edges=sys_.r_edges[keep])
    from tools.incremental_impedance import SingularCircuit
    diagnosed = False
    try:
        inc.solvers_for(Proposal(changes=ch, label=f"isolate node {node}"),
                        R_new=acts.R[keep], C_new=acts.C, sys_new=sys2)
        msg = "returned a solver for a disconnected circuit"
    except SingularCircuit as exc:
        diagnosed, msg = True, str(exc)
    _results.append(diagnosed)
    print(f"\n  [{'ok' if diagnosed else 'FAIL'}] isolating node {node} "
          f"(degree {len(rows)}) is DIAGNOSED, not silently wrong")
    print(f"        {msg}")
    print(f"        (refactorization cannot rescue this one either — the "
          f"modified circuit genuinely has no solution)")

    # (c) the rank / branch-fraction guards
    pol = FallbackPolicy(max_rank=8, max_branch_frac=0.25)
    inc2 = IncrementalImpedance(sys_, acts.R, acts.C, OMEGAS, policy=pol)
    big = acts.decap_global(2.0)
    _, info2 = inc2.solvers_for(big.proposal, R_new=big.R, C_new=big.C)
    ok = info2["mode"] == "refactor"
    _results.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] global decap (rank {big.rank}) trips "
          f"max_rank=8 -> {info2['mode']}")
    print(f"        reason: {info2['reason']}")

    # (d) a proposal that would need new nodes is refused outright
    p_new = Proposal(changes=list(big.proposal.changes), new_nodes=1)
    _, info3 = inc2.solvers_for(p_new, R_new=big.R, C_new=big.C)
    ok = info3["reason"] == "new nodes"
    _results.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] proposal introducing new nodes -> "
          f"{info3['mode']} ({info3['reason']})")


def check_11_reuse_and_drift(acts, sys_, inc):
    """Part 2: base immutability, the ZA cache, and chained-update drift."""
    print("\n11. reuse: immutable base, ZA cache, periodic refactorization")
    B = torch.randn(sys_.n_free, 4, dtype=DT).to(CDT)
    # a COLD instance, so the count below measures the cache and not whatever
    # the earlier checks happened to leave warm
    incc = IncrementalImpedance(sys_, acts.R, acts.C, OMEGAS)
    ref = incc.solvers[0].solve(B).clone()
    for mag in (0.5, 0.75, 0.9, 1.1, 1.5, 2.0):       # SAME strap, 6 magnitudes
        cand = acts.wire_width(1, [4, 5], mag)
        incc.solvers_for(cand.proposal, R_new=cand.R, C_new=cand.C,
                         sys_new=cand.sys)
    hits, miss = incc.stats["u_hits"], incc.stats["u_misses"]
    want = 2 * len(OMEGAS)          # 2 branches x 3 frequencies, solved once
    ok = miss == want and hits == 6 * want - want
    _results.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] ZA cache: {miss} solves (want {want}) "
          f"for 6 re-scorings of the same 2 branches, {hits} hits")

    after = incc.solvers[0].solve(B)
    report("base unchanged after 6 candidate evaluations", relerr(after, ref))

    # chained accepts: drift, then a forced refactorization
    pol = FallbackPolicy(max_depth=4)
    inc3 = IncrementalImpedance(sys_, acts.R, acts.C, OMEGAS, policy=pol)
    wb = acts.wb.clone()
    depth_seen = []
    for step in range(5):
        wb2 = wb.clone(); wb2[step] = wb2[step] * 1.3
        a2 = ActionSpace(acts.g, sys_, acts.wt, wb2, acts.C_sites, acts.loads,
                         FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
        ch = [type(acts.wire_width(1, [step], 1.3).proposal.changes[0])(
            int(sys_.r_edges[acts.n_top_e + step, 0]),
            int(sys_.r_edges[acts.n_top_e + step, 1]), "G",
            1.0 / _R_of(acts, wb, step), 1.0 / _R_of(acts, wb2, step))]
        from tools.incremental_impedance import Proposal as _P
        info = inc3.accept(_P(changes=ch), R_new=a2.R, C_new=a2.C)
        depth_seen.append(max(info.get("depth", [0]) or [0]))
        wb = wb2
    a_fin = ActionSpace(acts.g, sys_, acts.wt, wb, acts.C_sites, acts.loads,
                        FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
    ex = torch.linalg.solve(admittance(sys_, a_fin.R, a_fin.C, OMEGAS[0]), B)
    report("5 chained accepted updates still exact", relerr(inc3.solvers[0].solve(B), ex))
    print(f"        chain depth per accept: {depth_seen}; "
          f"refactorizations: {inc3.stats['refactored']}")


def _R_of(acts, wb, k):
    return acts.knobs(acts.wt, wb, acts.C_sites)[0][acts.n_top_e + k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="5,13")
    args = ap.parse_args()
    nt, nb = (int(v) for v in args.anchor.split(","))

    g, sys_, acts, rng = make_grid(nt, nb, seed=0, decap_fill=0.6)
    inc = IncrementalImpedance(sys_, acts.R, acts.C, OMEGAS)
    print(f"anchor ({nt},{nb})  n_free={sys_.n_free}  "
          f"r_edges={sys_.r_edges.shape[0]}  decap sites={sys_.c_edges.shape[0]}"
          f"  frequencies={len(OMEGAS)}")
    print(f"exactness tolerance for update-vs-refactor: {EXACT_TOL:.0e}")

    check_1_inverse_action(acts, sys_, inc)
    check_2_invariant_channels(acts, inc)
    check_3_normalized_features(acts, inc)
    check_4_5_model_output(acts, inc)
    check_6_gradients(acts, inc)
    check_7_random_circuits(inc)
    check_8_multi_edge(acts, sys_, inc)
    check_9_add_remove_move(acts, sys_, inc)
    check_10_fallback_boundary(acts, sys_, inc)
    check_11_reuse_and_drift(acts, sys_, inc)

    n_ok = sum(_results)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
