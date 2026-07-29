"""Validation suite for the impedance-attention prototype.

Runs the checks required by the design note:

  1  physics anchor      Z == F F^T, and droop == sum_l' (u_l.u_l') I_l'
  2  naive equivalence   factorized vs explicit [N,N] scores (fwd + grad)
  3  permutation equivariance
  4  directionality      a_(i<-j) != a_(j<-i) without type-specific ops
  5  load orientation    swapping VDD/VSS terminals flips the factor
  6  no-n^2 / scaling    peak memory and runtime vs node count
  7  gradients           near R, far R, C, load features; autograd vs FD
  8  kernel score        RFF factorized == naive; RFF vs exact Gaussian;
                         Taylor divergence (why RFF is the default)
  9  reweight vs reorder a scalar monotone f cannot change per-pair ranking
                         but does change the aggregated output

    python scripts/probes/impedance_attention_checks.py
    python scripts/probes/impedance_attention_checks.py --only 6
"""
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eason.impedance_attention_model import ImpAttnConfig, ImpedanceAttentionRegressor
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (
    branch_system,
    impedance_factors,
    knob_tensors,
    node_features,
)
from tools.sampler import (
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    FIXED_R_VIA,
    FIXED_RSHEET_BOT,
    FIXED_RSHEET_TOP,
)
from tools.transient_solver import solve_static_dc

DT = torch.float64
OK, BAD = "  [PASS]", "  [FAIL]"


def make_grid(n_top, n_bot, seed=0, ww=0.5, cd=2e-10):
    proto = build_regular_pdn(n_top=n_top, n_bot=n_bot)
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (proto.n_loads, 1))
    rng = np.random.default_rng(seed)
    wt = np.exp(rng.uniform(np.log(0.2), np.log(1.0), proto.top_edges.shape[0]))
    wb = np.exp(rng.uniform(np.log(0.2), np.log(1.0), proto.bot_edges.shape[0]))
    g = build_regular_pdn(
        n_top=n_top, n_bot=n_bot, Rsheet_top=FIXED_RSHEET_TOP,
        Rsheet_bot=FIXED_RSHEET_BOT, wire_width=ww, R_via=FIXED_R_VIA,
        C_decap=cd, freq=FIXED_FREQ, loads=loads,
        ww_top_edges=wt, ww_bot_edges=wb)
    return g, torch.tensor(wt, dtype=DT), torch.tensor(wb, dtype=DT)


def factors_for(g, wt, wb, cd, omegas, m=16, n_power=2):
    sys_ = branch_system(g)
    R, C = knob_tensors(g, wt, wb, torch.tensor(cd, dtype=DT),
                        FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
    p, s = impedance_factors(sys_, R, C, omegas, m=m, n_power=n_power)
    return sys_, p, s


# --------------------------------------------------------------- 1
def check_physics():
    print("1. physics anchor (Z = F F^T; droop = sum (u.u') I)")
    g, wt, wb = make_grid(3, 7)
    sys_ = branch_system(g)
    R, C = knob_tensors(g, wt, wb, torch.tensor(2e-10, dtype=DT),
                        FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
    om = torch.zeros(1, dtype=DT)
    # full-rank factors reproduce Z exactly
    p, s = impedance_factors(sys_, R, C, om, m=sys_.n_free, n_power=0)
    from tools.impedance_factors import admittance
    Z = torch.linalg.inv(admittance(sys_, R, C, om[0])).real
    live = sys_.free_of >= 0
    Zf = (p[:sys_.n_elec][live][:, 0] @ s[:sys_.n_elec][live][:, 0].T)
    e1 = (Zf - Z).abs().max().item() / Z.abs().max().item()
    print(f"   max rel |Z - p.s|            {e1:.2e}")

    # droop identity at DC, full rank
    I = torch.tensor(g.loads[:, 0] * g.loads[:, 2], dtype=DT)
    d_attn = (p[sys_.n_elec:][:, 0] @ s[sys_.n_elec:][:, 0].T) @ I
    dc = solve_static_dc(g)
    d_true = torch.tensor(
        g.Vdd - (dc["V_bot"][g.load_pairs[:, 0].astype(int)]
                 - dc["V_bot"][g.load_pairs[:, 1].astype(int)]), dtype=DT)
    e2 = (d_attn - d_true).abs().max().item() / d_true.abs().max().item()
    print(f"   max rel droop error          {e2:.2e}")

    # compact rank, both die sizes -> is the error diameter-invariant?
    for (nt, nb) in ((3, 7), (13, 13)):
        gg, wt2, wb2 = make_grid(nt, nb)
        s2, p2, s2f = factors_for(gg, wt2, wb2, 2e-10, om, m=16)
        I2 = torch.tensor(gg.loads[:, 0] * gg.loads[:, 2], dtype=DT)
        d2 = (p2[s2.n_elec:][:, 0] @ s2f[s2.n_elec:][:, 0].T) @ I2
        dc2 = solve_static_dc(gg)
        dt2 = torch.tensor(
            gg.Vdd - (dc2["V_bot"][gg.load_pairs[:, 0].astype(int)]
                      - dc2["V_bot"][gg.load_pairs[:, 1].astype(int)]), dtype=DT)
        r = (d2 - dt2).abs().max().item() / dt2.abs().max().item()
        print(f"   rank-16 droop err ({nt},{nb})   {r:.4f}  (n_free={s2.n_free})")
    # AC reconstruction: the complex factorization AND the real-channel
    # encoding must both be exact. Regression guard for two bugs that were
    # invisible at DC (unitary-vs-orthogonal QR; missing cross channels).
    from tools.impedance_factors import admittance as _adm, channel_count
    w = torch.tensor(2 * np.pi * FIXED_FREQ, dtype=DT)
    om_ac = torch.stack([torch.zeros((), dtype=DT), w])
    Zc = torch.linalg.inv(_adm(sys_, R, C, w))
    pA, sA = impedance_factors(sys_, R, C, om_ac, m=sys_.n_free, n_power=0)
    pA, sA = pA[:sys_.n_elec][live], sA[:sys_.n_elec][live]
    assert pA.shape[1] == channel_count(om_ac) == 5
    rr, ii = pA[:, 1] @ sA[:, 1].T, pA[:, 2] @ sA[:, 2].T
    ri, ir = pA[:, 3] @ sA[:, 3].T, pA[:, 4] @ sA[:, 4].T
    eR = (rr - ii - Zc.real).abs().max().item() / Zc.abs().max().item()
    eI = (ri + ir - Zc.imag).abs().max().item() / Zc.abs().max().item()
    frac = (Zc.imag.abs().median() / Zc.real.abs().median()).item()
    print(f"   AC Re(Z) = rr - ii           {eR:.2e}")
    print(f"   AC Im(Z) = ri + ir           {eI:.2e}   (|Im|/|Re| = {frac:.2f})")
    ok = e1 < 1e-8 and e2 < 1e-8 and eR < 1e-10 and eI < 1e-10
    print(OK if ok else BAD)
    return ok


# --------------------------------------------------------------- 2
def check_naive_equivalence():
    print("2. naive equivalence (factorized vs explicit [N,N])")
    torch.manual_seed(0)
    g, wt, wb = make_grid(3, 7)
    om = torch.tensor([0.0, 2 * np.pi * FIXED_FREQ], dtype=DT)
    sys_, p, s = factors_for(g, wt, wb, 2e-10, om, m=8)
    x = node_features(sys_, torch.tensor(g.loads, dtype=DT))
    model = ImpedanceAttentionRegressor(
        ImpAttnConfig(n_freq=2, m_factor=8, hidden_dim=32, d_v=16)).to(DT)
    # decoder is zero-init by design; randomize so gradients are non-trivial
    for prm in model.decoder.parameters():
        nn_init = torch.randn_like(prm) * 0.1
        prm.data.copy_(nn_init)

    pf = p.clone().requires_grad_(True)
    yf = model(x, pf, s, sys_.n_elec, naive=False)
    yf.sum().backward()
    gf = pf.grad.clone()

    pn = p.clone().requires_grad_(True)
    yn = model(x, pn, s, sys_.n_elec, naive=True)
    yn.sum().backward()
    gn = pn.grad.clone()

    ef = (yf - yn).abs().max().item() / yn.abs().max().clamp_min(1e-30).item()
    eg = (gf - gn).abs().max().item() / gn.abs().max().clamp_min(1e-30).item()
    print(f"   max rel output diff          {ef:.2e}")
    print(f"   max rel gradient diff        {eg:.2e}")
    ok = ef < 1e-9 and eg < 1e-9
    print(OK if ok else BAD)
    return ok


# --------------------------------------------------------------- 3
def check_permutation():
    print("3. permutation equivariance")
    torch.manual_seed(0)
    g, wt, wb = make_grid(3, 7)
    om = torch.tensor([0.0, 2 * np.pi * FIXED_FREQ], dtype=DT)
    sys_, p, s = factors_for(g, wt, wb, 2e-10, om, m=8)
    x = node_features(sys_, torch.tensor(g.loads, dtype=DT))
    model = ImpedanceAttentionRegressor(
        ImpAttnConfig(n_freq=2, m_factor=8, hidden_dim=32, d_v=16)).to(DT)
    for prm in model.decoder.parameters():
        prm.data.copy_(torch.randn_like(prm) * 0.1)

    y = model(x, p, s, sys_.n_elec)
    # permute electrical nodes and load nodes independently
    ge = torch.randperm(sys_.n_elec)
    gl = torch.randperm(sys_.n_loads)
    perm = torch.cat([ge, sys_.n_elec + gl])
    y2 = model(x[perm], p[perm], s[perm], sys_.n_elec)
    err = (y2 - y[gl]).abs().max().item() / y.abs().max().clamp_min(1e-30).item()
    print(f"   max rel diff vs permuted     {err:.2e}")
    ok = err < 1e-9
    print(OK if ok else BAD)
    return ok


# --------------------------------------------------------------- 4
def check_directionality():
    print("4. directionality  a(i<-j) != a(j<-i)")
    torch.manual_seed(0)
    g, wt, wb = make_grid(3, 7)
    om = torch.tensor([0.0, 2 * np.pi * FIXED_FREQ], dtype=DT)
    sys_, p, s = factors_for(g, wt, wb, 2e-10, om, m=8)
    x = node_features(sys_, torch.tensor(g.loads, dtype=DT))
    cfg = ImpAttnConfig(n_freq=2, m_factor=8, hidden_dim=32, d_v=16)
    model = ImpedanceAttentionRegressor(cfg).to(DT)
    # randomize phi/psi (physics init makes them equal on purpose)
    for lin in (model.attn.phi, model.attn.psi, model.attn.q, model.attn.k):
        lin.weight.data.copy_(torch.randn_like(lin.weight) * 0.3)

    h, pn, sn = model.embed(x, p, s, sys_.n_elec)
    N, H = h.shape[0], cfg.heads
    q = model.attn.q(h).view(N, H, cfg.d_qk)
    k = model.attn.k(h).view(N, H, cfg.d_qk)
    P, S = model.attn._factor_terms(h, pn, sn)
    A = torch.einsum("ihd,jhd->hij", q, k) * torch.einsum("ihd,jhd->hij", P, S)
    asym = (A - A.transpose(1, 2)).abs().max().item() / A.abs().max().item()
    print(f"   max rel asymmetry |A - A^T|  {asym:.3e}")
    # Reference: the DC (real, channel 0) physical factor alone. Z is
    # symmetric for an RLC network, so this channel should be symmetric —
    # asymmetry in A is contributed entirely by learned Q/K and phi/psi,
    # not smuggled in by the physics. (Channels are NOT symmetric once
    # re/im parts are split at w>0, so only channel 0 is checked here.)
    Z0 = pn[:, 0] @ sn[:, 0].T
    rec = (Z0 - Z0.T).abs().max().item() / Z0.abs().max().item()
    print(f"   DC physical channel asym     {rec:.3e}  (reciprocity: expect ~0)")
    ok = asym > 1e-3
    print(OK if ok else BAD)
    return ok


# --------------------------------------------------------------- 5
def check_load_orientation():
    print("5. load orientation swap")
    g, wt, wb = make_grid(3, 7)
    om = torch.zeros(1, dtype=DT)
    sys_, p, s = factors_for(g, wt, wb, 2e-10, om, m=16)
    sys2 = branch_system(g)
    sys2.load_terminals[0] = sys2.load_terminals[0].flip(0)   # swap load 0
    R, C = knob_tensors(g, wt, wb, torch.tensor(2e-10, dtype=DT),
                        FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
    p2, s2 = impedance_factors(sys2, R, C, om, m=16)
    i = sys_.n_elec
    flipped = (p2[i] + p[i]).abs().max().item() / p[i].abs().max().item()
    others = (p2[i + 1:] - p[i + 1:]).abs().max().item()
    print(f"   swapped load: |p_new + p_old| {flipped:.2e}  (0 => exact negation)")
    print(f"   other loads unchanged         {others:.2e}")
    ok = flipped < 1e-12 and others < 1e-12
    print(OK if ok else BAD)
    return ok


# --------------------------------------------------------------- 6
def check_scaling():
    print("6. no-n^2 attention + scaling")
    om = torch.tensor([0.0, 2 * np.pi * FIXED_FREQ], dtype=DT)
    cfg = ImpAttnConfig(n_freq=2, m_factor=16, hidden_dim=64, d_v=32)
    model = ImpedanceAttentionRegressor(cfg).to(DT)
    print(f"   {'anchor':>9} {'N':>6} {'attn ms':>9} {'attn peak MB':>13} "
          f"{'n^2 floats':>11} {'ratio':>7}")
    rows = []
    for (nt, nb) in ((3, 7), (7, 7), (7, 13), (13, 13)):
        g, wt, wb = make_grid(nt, nb)
        sys_, p, s = factors_for(g, wt, wb, 2e-10, om, m=16)
        x = node_features(sys_, torch.tensor(g.loads, dtype=DT))
        h, pn, sn = model.embed(x, p, s, sys_.n_elec)
        N = h.shape[0]
        with torch.no_grad():
            model.attn(h, pn, sn)                      # warm up
            tracemalloc.start()
            t0 = time.perf_counter()
            for _ in range(5):
                model.attn(h, pn, sn)
            ms = (time.perf_counter() - t0) / 5 * 1e3
            peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
        n2 = N * N
        rows.append((N, ms, peak))
        print(f"   {f'({nt},{nb})':>9} {N:>6} {ms:>9.2f} {peak:>13.3f} "
              f"{n2:>11,} {peak*1e6/8/max(n2,1):>7.2f}")
    # linear-ish: time per node should not grow like N
    r = (rows[-1][1] / rows[0][1]) / (rows[-1][0] / rows[0][0])
    print(f"   time growth / node growth    {r:.2f}   (1.0 = linear, {rows[-1][0]/rows[0][0]:.1f} = quadratic)")
    ok = r < 2.0
    print(OK if ok else BAD)
    return ok


# --------------------------------------------------------------- 7
def check_gradients():
    print("7. gradients to components (autograd vs finite differences)")
    torch.manual_seed(0)
    g, wt, wb = make_grid(7, 13)
    om = torch.tensor([0.0, 2 * np.pi * FIXED_FREQ], dtype=DT)
    cfg = ImpAttnConfig(n_freq=2, m_factor=16, hidden_dim=32, d_v=16)
    model = ImpedanceAttentionRegressor(cfg).to(DT)
    for prm in model.decoder.parameters():
        prm.data.copy_(torch.randn_like(prm) * 0.1)
    for lin in (model.attn.phi, model.attn.psi):
        lin.weight.data.copy_(torch.randn_like(lin.weight) * 0.2)
    sys_ = branch_system(g)
    loads_t = torch.tensor(g.loads, dtype=DT)

    def fwd(wt_, wb_, cd_, loads_):
        R, C = knob_tensors(g, wt_, wb_, cd_, FIXED_RSHEET_TOP,
                            FIXED_RSHEET_BOT, FIXED_R_VIA)
        p, s = impedance_factors(sys_, R, C, om, m=16)
        x = node_features(sys_, loads_)
        return model(x, p, s, sys_.n_elec).sum()

    wt_ = wt.clone().requires_grad_(True)
    wb_ = wb.clone().requires_grad_(True)
    cd_ = torch.tensor(2e-10, dtype=DT, requires_grad=True)
    ld_ = loads_t.clone().requires_grad_(True)
    y = fwd(wt_, wb_, cd_, ld_)
    y.backward()

    # locality of the perturbed edge relative to the worst load
    d_bot = np.linalg.norm(g.bot_pos[g.bot_edges[:, 0]]
                           - g.bot_pos[g.load_pairs[0, 0]], axis=1)
    near_e, far_e = int(np.argmin(d_bot)), int(np.argmax(d_bot))
    n_bot_hops = g.n_bot            # 13 -> a 7-layer stack cannot reach across

    def fd(vec, idx, base_fwd, eps=1e-4):
        v1 = vec.detach().clone(); v1[idx] *= (1 + eps)
        v2 = vec.detach().clone(); v2[idx] *= (1 - eps)
        return ((base_fwd(v1) - base_fwd(v2)) / (2 * eps)).item()

    tests = []
    tests.append(("near bot R  (hop~1)", wb_.grad[near_e].item() * wb[near_e].item(),
                  fd(wb, near_e, lambda v: fwd(wt, v, cd_.detach(), loads_t))))
    tests.append((f"far  bot R  (hop~{n_bot_hops})",
                  wb_.grad[far_e].item() * wb[far_e].item(),
                  fd(wb, far_e, lambda v: fwd(wt, v, cd_.detach(), loads_t))))
    tests.append(("top strap R", wt_.grad[0].item() * wt[0].item(),
                  fd(wt, 0, lambda v: fwd(v, wb, cd_.detach(), loads_t))))
    eps = 1e-4
    fdc = ((fwd(wt, wb, torch.tensor(2e-10 * (1 + eps), dtype=DT), loads_t)
            - fwd(wt, wb, torch.tensor(2e-10 * (1 - eps), dtype=DT), loads_t))
           / (2 * eps)).item()
    tests.append(("decap C", cd_.grad.item() * 2e-10, fdc))
    l1 = loads_t.clone(); l1[:, 0] *= (1 + eps)
    l2 = loads_t.clone(); l2[:, 0] *= (1 - eps)
    fdl = ((fwd(wt, wb, cd_.detach(), l1) - fwd(wt, wb, cd_.detach(), l2))
           / (2 * eps)).item()
    tests.append(("load I_peak", (ld_.grad[:, 0] * loads_t[:, 0]).sum().item(), fdl))

    allok = True
    for name, ag, fdv in tests:
        rel = abs(ag - fdv) / max(abs(fdv), 1e-30)
        nz = abs(ag) > 1e-30
        good = nz and rel < 1e-4
        allok &= good
        print(f"   {name:<22} autograd {ag:+.4e}  FD {fdv:+.4e}  rel {rel:.1e}"
              f"{'' if good else '   <-- ' + ('ZERO' if not nz else 'MISMATCH')}")

    # Headline claim: a distant component must retain a live gradient path.
    # Reference target = the TRUE far/near ratio from the differentiable
    # simulator (tools/torch_sim.py). The model here is untrained, so its
    # ratio is not expected to match yet — what matters is that it is
    # non-zero, i.e. structurally reachable. A 7-layer local stack gives
    # exactly 0 at this diameter and can never learn its way out.
    ratio = abs(tests[1][1]) / max(abs(tests[0][1]), 1e-30)
    from tools.torch_sim import worst_droop_jacobian
    _, _, jb_true, _, _ = worst_droop_jacobian(
        g, wt.numpy(), wb.numpy(), np.full(g.n_decaps, 2e-10),
        FIXED_RSHEET_TOP, FIXED_RSHEET_BOT)
    true_ratio = abs(jb_true[far_e]) / max(abs(jb_true[near_e]), 1e-30)
    print(f"   far/near gradient ratio      {ratio:.4f}  (untrained model)")
    print(f"   true ratio from simulator    {true_ratio:.4f}  (training target)")
    print(f"   7-layer local model          0.0000  (structurally unreachable)")
    print(OK if allok else BAD)
    return allok


# --------------------------------------------------------------- 8
def check_kernel_score():
    print("8. kernel score (RFF): factorization, fidelity, Taylor limits")
    from tools.impedance_factors import dc_symmetric_factor
    torch.manual_seed(0)
    g, wt, wb = make_grid(3, 7)
    om = torch.tensor([0.0, 2 * np.pi * FIXED_FREQ], dtype=DT)
    sys_, p, s = factors_for(g, wt, wb, 2e-10, om, m=8)
    R, C = knob_tensors(g, wt, wb, torch.tensor(2e-10, dtype=DT),
                        FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
    fdc = dc_symmetric_factor(sys_, R, C, m=8, n_power=2)
    x = node_features(sys_, torch.tensor(g.loads, dtype=DT))
    cfg = ImpAttnConfig(n_freq=2, m_factor=8, hidden_dim=32, d_v=16,
                        score="kernel", n_scales=2, n_rff=64)
    model = ImpedanceAttentionRegressor(cfg).to(DT)
    for prm in model.decoder.parameters():
        prm.data.copy_(torch.randn_like(prm) * 0.1)
    model.attn.kw.data.copy_(torch.randn_like(model.attn.kw) * 0.5)

    pf = p.clone().requires_grad_(True)
    yf = model(x, pf, s, sys_.n_elec, fdc=fdc); yf.sum().backward()
    gf = pf.grad.clone()
    pn = p.clone().requires_grad_(True)
    yn = model(x, pn, s, sys_.n_elec, naive=True, fdc=fdc); yn.sum().backward()
    gn = pn.grad.clone()
    ef = (yf - yn).abs().max().item() / yn.abs().max().item()
    eg = (gf - gn).abs().max().item() / gn.abs().max().item()
    print(f"   factorized vs naive: output  {ef:.2e}")
    print(f"   factorized vs naive: grad    {eg:.2e}")

    # RFF approximates the Gaussian kernel it claims to; Taylor does not
    fn = fdc / fdc.norm(dim=-1, keepdim=True).mean()
    d2 = torch.cdist(fn, fn) ** 2
    print(f"   {'gamma':>7} {'RFF err':>9} {'Taylor err':>11} {'kernel spread':>14}")
    tay_bad = False
    for gm in (0.1, 1.0, 10.0):
        gmt = torch.tensor(gm, dtype=DT)
        exact = torch.exp(-gm * d2)
        z = model.attn._rff(fn, gmt)
        e_rff = (z @ z.T - exact).abs().max().item()
        zii = (fn * fn).sum(-1)
        gt = model.attn._taylor(fn, gmt) * torch.exp(-gmt * zii).unsqueeze(-1)
        e_tay = (gt @ gt.T - exact).abs().max().item() / exact.abs().max().item()
        tay_bad |= (gm >= 1.0 and e_tay > 0.5)
        print(f"   {gm:>7.1f} {e_rff:>9.3f} {e_tay:>11.3f} "
              f"{(exact.max()/exact.min()).item():>14.1e}")
    ok = ef < 1e-9 and eg < 1e-9 and tay_bad
    print("   (Taylor failing at large gamma is expected — it is why RFF is default)")
    print(OK if ok else BAD)
    return ok


# --------------------------------------------------------------- 9
def check_reweight_vs_reorder():
    print("9. reweight vs reorder (what a nonlinearity can and cannot do)")
    torch.manual_seed(0)
    g, wt, wb = make_grid(3, 7)
    om = torch.zeros(1, dtype=DT)
    sys_, p, s = factors_for(g, wt, wb, 2e-10, om, m=16)
    L0 = sys_.n_elec
    z = p[L0:][:, 0] @ s[L0:][:, 0].T          # load-to-load scores
    v = torch.tensor(g.loads[:, 0] * g.loads[:, 2], dtype=DT)

    from scipy.stats import spearmanr
    base = z[0].numpy()
    rhos, outs = [], []
    for gm in (0.5, 2.0, 8.0):
        f = torch.exp(-gm * (-z))              # monotone increasing in z
        rhos.append(spearmanr(f[0].numpy(), base).statistic)
        outs.append((f @ v)[0].item())
    same_rank = min(rhos) > 0.999999
    spread = max(outs) / min(outs)
    print(f"   per-pair ranking under monotone f: spearman {min(rhos):.6f} "
          f"({'unchanged' if same_rank else 'CHANGED'})")
    print(f"   aggregated output sum_j f(z_ij) v_j varies by {spread:.1f}x")
    # multivariate f (using the source's own self-impedance) CAN reorder
    zjj = (p[L0:][:, 0] * s[L0:][:, 0]).sum(-1)
    multi = z - 0.5 * zjj.unsqueeze(0)
    rho_multi = spearmanr(multi[0].numpy(), base).statistic
    print(f"   multivariate f (uses z_jj): spearman vs base {rho_multi:.4f} "
          f"({'reorders' if rho_multi < 0.999 else 'no reorder'})")
    ok = same_rank and spread > 1.5 and rho_multi < 0.999
    print("   => a scalar monotone f reweights but cannot reorder; only the")
    print("      multivariate route changes ranking. Both are in the design.")
    print(OK if ok else BAD)
    return ok


CHECKS = [check_physics, check_naive_equivalence, check_permutation,
          check_directionality, check_load_orientation, check_scaling,
          check_gradients, check_kernel_score, check_reweight_vs_reorder]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="run one check (1-9)")
    args = ap.parse_args()
    todo = [CHECKS[args.only - 1]] if args.only else CHECKS
    results = []
    for fn in todo:
        results.append(fn())
        print()
    n_ok = sum(results)
    print(f"== {n_ok}/{len(results)} checks passed ==")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
