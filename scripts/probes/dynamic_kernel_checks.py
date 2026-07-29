"""Correctness checks for the unified dynamic-impedance kernel.

Run before any training. Every check is a measurement, not an assertion of
intent.

  1  factorized == explicit [N,N] reference, forward
  2  factorized == explicit gradients: parameters, wire R, decap C
  3  capacitance gradient is finite and NON-ZERO after real init
     (the DC-only kernel it replaces has exactly zero here)
  4  permutation equivariance
  5  basis invariance under a different factor basis
  6  directionality: ordered scores differ while the impedance stays reciprocal
  7  no [N,N] allocation on the production path
  8  scaling vs node count and feature count
  9  learned per-head frequency mixture is observable (no specialization required)

    python scripts/probes/dynamic_kernel_checks.py
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eason.impedance_attention_model import (ImpAttnConfig,
                                             ImpedanceAttentionRegressor)
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (branch_system, impedance_factors,
                                     invariant_channels, knob_tensors,
                                     node_features)
from tools.sampler import (FIXED_DUTY, FIXED_FREQ, FIXED_I_PEAK, FIXED_PHASE,
                           FIXED_R_VIA, FIXED_RSHEET_BOT, FIXED_RSHEET_TOP)

DT = torch.float64
OK, BAD = "  [PASS]", "  [FAIL]"
N_FREQ, M = 3, 8


def omegas(n=N_FREQ):
    b = 2 * np.pi * FIXED_FREQ
    return torch.tensor([0.0, b, 5 * b, 25 * b][:n], dtype=DT)


def make(nt=3, nb=7, ww=0.5, cd=2e-10):
    proto = build_regular_pdn(n_top=nt, n_bot=nb)
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (proto.n_loads, 1))
    g = build_regular_pdn(n_top=nt, n_bot=nb, Rsheet_top=FIXED_RSHEET_TOP,
                          Rsheet_bot=FIXED_RSHEET_BOT, wire_width=ww,
                          R_via=FIXED_R_VIA, C_decap=cd, freq=FIXED_FREQ,
                          loads=loads)
    return g, branch_system(g)


def factors(g, s_, wt, wb, cd, m=M, om=None):
    R, C = knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                        FIXED_R_VIA)
    return impedance_factors(s_, R, C, om if om is not None else omegas(),
                             m=m, n_power=2)


def model(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = ImpAttnConfig(hidden_dim=32, heads=3, d_v=16, n_freq=N_FREQ,
                        m_factor=M, score="dynamic_kernel", **kw)
    mdl = ImpedanceAttentionRegressor(cfg, init_bias=-3.6).to(DT)
    # non-degenerate init: a zero-init decoder makes every gradient zero and
    # every check pass vacuously
    with torch.no_grad():
        for prm in mdl.decoder.parameters():
            prm.copy_(torch.randn_like(prm) * 0.2)
        mdl.attn.phi.weight.copy_(torch.randn_like(mdl.attn.phi.weight) * 0.1)
        mdl.attn.psi.weight.copy_(torch.randn_like(mdl.attn.psi.weight) * 0.1)
    return mdl


def _setup(nt=3, nb=7, seed=0):
    g, s_ = make(nt, nb)
    wt = torch.full((g.top_edges.shape[0],), 0.5, dtype=DT)
    wb = torch.full((g.bot_edges.shape[0],), 0.5, dtype=DT)
    cd = torch.tensor(2e-10, dtype=DT)
    p, s = factors(g, s_, wt, wb, cd)
    x = node_features(s_, torch.tensor(g.loads, dtype=DT))
    return g, s_, wt, wb, cd, p, s, x, model(seed)


# ------------------------------------------------------------------ 1
def check_factorized_vs_naive():
    print("1. factorized vs explicit [N,N] reference (forward)")
    _, s_, _, _, _, p, s, x, mdl = _setup()
    yf = mdl(x, p, s, s_.n_elec, naive=False)
    yn = mdl(x, p, s, s_.n_elec, naive=True)
    e = (yf - yn).abs().max().item() / yn.abs().max().item()
    print(f"   rel diff {e:.3e}   (N={p.shape[0]})")
    ok = e < 1e-12
    print(OK if ok else BAD)
    return ok


# ------------------------------------------------------------------ 2
def check_factorized_vs_naive_grads():
    print("2. factorized vs explicit gradients (params, wire R, decap C)")
    g, s_, wt, wb, cd, _, _, x, mdl = _setup()
    outs = {}
    for naive in (False, True):
        w_t = wt.clone().requires_grad_(True)
        b_t = wb.clone().requires_grad_(True)
        c_t = cd.clone().requires_grad_(True)
        p, s = factors(g, s_, w_t, b_t, c_t)
        y = mdl(x, p, s, s_.n_elec, naive=naive).sum()
        gw, gb, gc = torch.autograd.grad(y, [w_t, b_t, c_t], retain_graph=True)
        gp = torch.autograd.grad(y, list(mdl.parameters()), allow_unused=True)
        outs[naive] = (gw, gb, gc, gp)
    gw0, gb0, gc0, gp0 = outs[False]
    gw1, gb1, gc1, gp1 = outs[True]
    rel = lambda a, b: ((a - b).abs().max() / a.abs().max().clamp_min(1e-300)).item()
    e_w, e_b = rel(gw0, gw1), rel(gb0, gb1)
    e_c = abs(float(gc0 - gc1)) / max(abs(float(gc0)), 1e-300)
    e_p = max(rel(a, b) for a, b in zip(gp0, gp1) if a is not None)
    print(f"   d/d ww_top {e_w:.2e}   d/d ww_bot {e_b:.2e}   "
          f"d/d C {e_c:.2e}   params {e_p:.2e}")
    ok = max(e_w, e_b, e_c, e_p) < 1e-9
    print(OK if ok else BAD)
    return ok


# ------------------------------------------------------------------ 3
def check_capacitance_gradient():
    print("3. capacitance gradient finite and NON-ZERO (vs FD)")
    g, s_, wt, wb, cd, _, _, x, mdl = _setup()
    c_t = cd.clone().requires_grad_(True)
    p, s = factors(g, s_, wt, wb, c_t)
    y = mdl(x, p, s, s_.n_elec).sum()
    (gc,) = torch.autograd.grad(y, [c_t])
    h = float(cd) * 1e-6
    with torch.no_grad():
        yp = mdl(x, *factors(g, s_, wt, wb, cd + h), s_.n_elec).sum()
        ym = mdl(x, *factors(g, s_, wt, wb, cd - h), s_.n_elec).sum()
    fd = float((yp - ym) / (2 * h))
    rel = abs(float(gc) - fd) / max(abs(fd), 1e-300)
    print(f"   autograd {float(gc):+.6e}   FD {fd:+.6e}   rel {rel:.2e}")
    # the DC-only kernel this replaces returns exactly 0 here
    nonzero = abs(float(gc)) > 1e-12 * max(abs(float(y)), 1.0)
    print(f"   non-zero: {nonzero}   (the DC-only Gaussian kernel gives exactly 0)")
    ok = nonzero and rel < 1e-4
    print(OK if ok else BAD)
    return ok


# ------------------------------------------------------------------ 4
def check_permutation():
    print("4. permutation equivariance")
    _, s_, _, _, _, p, s, x, mdl = _setup()
    y0 = mdl(x, p, s, s_.n_elec)
    ne = s_.n_elec
    g_ = torch.Generator().manual_seed(3)
    perm = torch.randperm(ne, generator=g_)
    full = torch.cat([perm, torch.arange(ne, p.shape[0])])
    y1 = mdl(x[full], p[full], s[full], ne)
    e = (y0 - y1).abs().max().item() / y0.abs().max().item()
    print(f"   permuting the {ne} electrical nodes: rel diff {e:.3e}")
    ok = e < 1e-12
    print(OK if ok else BAD)
    return ok


# ------------------------------------------------------------------ 5
def check_basis_invariance():
    print("5. basis invariance (independent factor basis, exact rank)")
    g, s_ = make(3, 7)
    wt = torch.full((g.top_edges.shape[0],), 0.5, dtype=DT)
    wb = torch.full((g.bot_edges.shape[0],), 0.5, dtype=DT)
    cd = torch.tensor(2e-10, dtype=DT)
    R, C = knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                        FIXED_R_VIA)
    x = node_features(s_, torch.tensor(g.loads, dtype=DT))
    mfull = s_.n_free                      # exact -> only basis differs
    torch.manual_seed(0)
    cfg = ImpAttnConfig(hidden_dim=32, heads=3, d_v=16, n_freq=N_FREQ,
                        m_factor=mfull, score="dynamic_kernel")
    mdl = ImpedanceAttentionRegressor(cfg, init_bias=-3.6).to(DT)
    with torch.no_grad():
        for prm in mdl.decoder.parameters():
            prm.copy_(torch.randn_like(prm) * 0.2)
    ys = []
    for seed in (0, 4321):
        p, s = impedance_factors(s_, R, C, omegas(), m=mfull, n_power=2, seed=seed)
        ys.append(mdl(x, p, s, s_.n_elec))
    e = (ys[0] - ys[1]).abs().max().item() / ys[0].abs().max().item()
    print(f"   two probe seeds, m=n_free={mfull}: rel diff {e:.3e}")
    # contrast: the raw AC channels the old score consumes are NOT invariant
    p0, s0 = impedance_factors(s_, R, C, omegas(), m=mfull, n_power=2, seed=0)
    p1, s1 = impedance_factors(s_, R, C, omegas(), m=mfull, n_power=2, seed=4321)
    zr0 = torch.einsum("icm,jcm->cij", p0, s0)
    zr1 = torch.einsum("icm,jcm->cij", p1, s1)
    raw = ((zr0 - zr1).abs().max(-1).values.max(-1).values
           / zr0.abs().amax((-2, -1)).clamp_min(1e-300))
    print(f"   for contrast, RAW per-channel z: DC {raw[0]:.1e}, "
          f"AC worst {raw[1:].max():.1e}  <- why regrouping is required")
    ok = e < 1e-9
    print(OK if ok else BAD)
    return ok


# ------------------------------------------------------------------ 6
def check_directionality():
    print("6. directionality of the score, reciprocity of the physics")
    g, s_ = make(3, 7)
    wt = torch.full((g.top_edges.shape[0],), 0.5, dtype=DT)
    wb = torch.full((g.bot_edges.shape[0],), 0.5, dtype=DT)
    cd = torch.tensor(2e-10, dtype=DT)
    x = node_features(s_, torch.tensor(g.loads, dtype=DT))
    # Reciprocity is a property of the PHYSICS, so assert it where the
    # factorization is exact. At reduced rank the AC sketch is NOT
    # reciprocal (reported below) — a limitation of the factorization that
    # predates this score and applies to every arm.
    mfull = s_.n_free
    torch.manual_seed(0)
    cfg = ImpAttnConfig(hidden_dim=32, heads=3, d_v=16, n_freq=N_FREQ,
                        m_factor=mfull, score="dynamic_kernel")
    mdl = ImpedanceAttentionRegressor(cfg, init_bias=-3.6).to(DT)
    with torch.no_grad():
        mdl.attn.phi.weight.copy_(torch.randn_like(mdl.attn.phi.weight) * 0.3)
        mdl.attn.psi.weight.copy_(torch.randn_like(mdl.attn.psi.weight) * 0.3)
    p, s = factors(g, s_, wt, wb, cd, m=mfull)
    hstate, chans = mdl.embed_invariant(x, p, s, s_.n_elec)
    Phi, Psi = mdl.attn._maps(hstate, chans)
    score = torch.einsum("ihd,jhd->hij", Phi, Psi)
    rel_asym = ((score - score.transpose(-1, -2)).abs().max()
                / score.abs().max()).item()
    recip = max(((z - z.T).abs().max() / z.abs().max()).item()
                for z in (torch.einsum("id,jd->ij", A, B) for A, B, _ in chans))
    print(f"   |A(i<-j) - A(j<-i)| rel {rel_asym:.3e}   <- must be > 0")
    print(f"   impedance channel asymmetry (exact rank) {recip:.3e}  <- ~0")
    # how badly the reduced-rank sketch violates reciprocity, for the record
    print(f"   {'rank':>6}  AC reciprocity error (max over channels)")
    for m in (8, 16, 32, mfull):
        pm, sm = factors(g, s_, wt, wb, cd, m=m)
        ch = invariant_channels(pm, sm, omegas())
        e = max(((z - z.T).abs().max() / z.abs().max()).item()
                for z in (torch.einsum("id,jd->ij", A, B) for A, B, _ in ch[1:]))
        print(f"   {m:>6}  {e:.2e}" + ("   <- exact" if m == mfull else ""))
    ok = rel_asym > 1e-3 and recip < 1e-9
    print(OK if ok else BAD)
    return ok


# ------------------------------------------------------------------ 7
def check_no_n2():
    print("7. no [N,N] allocation on the production path")
    print(f"   {'anchor':>9} {'N':>5} {'peak MB':>9} {'N^2 floats MB':>14} {'ratio':>7}")
    ok = True
    for nt, nb in ((3, 7), (7, 7), (7, 13), (13, 13)):
        g, s_ = make(nt, nb)
        wt = torch.full((g.top_edges.shape[0],), 0.5, dtype=DT)
        wb = torch.full((g.bot_edges.shape[0],), 0.5, dtype=DT)
        p, s = factors(g, s_, wt, wb, torch.tensor(2e-10, dtype=DT))
        x = node_features(s_, torch.tensor(g.loads, dtype=DT))
        mdl = model()
        N = p.shape[0]
        with torch.no_grad():
            tracemalloc.start()
            mdl(x, p, s, s_.n_elec)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        n2 = N * N * 8 / 1e6
        ratio = peak / 1e6 / n2
        print(f"   {f'({nt},{nb})':>9} {N:>5} {peak/1e6:>9.3f} {n2:>14.3f} {ratio:>7.2f}")
        ok &= ratio < 1.0
    print(OK if ok else BAD)
    return ok


# ------------------------------------------------------------------ 8
def check_scaling():
    print("8. scaling vs node count and feature count")
    print(f"   {'anchor':>9} {'N':>5} {'F':>7} {'fwd+bwd ms':>11}")
    rows = []
    for nt, nb in ((3, 7), (7, 7), (7, 13), (13, 13)):
        g, s_ = make(nt, nb)
        wt = torch.full((g.top_edges.shape[0],), 0.5, dtype=DT)
        wb = torch.full((g.bot_edges.shape[0],), 0.5, dtype=DT)
        p, s = factors(g, s_, wt, wb, torch.tensor(2e-10, dtype=DT))
        x = node_features(s_, torch.tensor(g.loads, dtype=DT))
        mdl = model()
        for _ in range(2):
            mdl(x, p, s, s_.n_elec).sum().backward()
        t0 = time.time()
        for _ in range(5):
            mdl.zero_grad(); mdl(x, p, s, s_.n_elec).sum().backward()
        dt = (time.time() - t0) / 5 * 1e3
        rows.append((p.shape[0], dt))
        print(f"   {f'({nt},{nb})':>9} {p.shape[0]:>5} {mdl.attn.F:>7} {dt:>11.2f}")
    gN = rows[-1][0] / rows[0][0]
    gT = rows[-1][1] / rows[0][1]
    print(f"   time growth / node growth = {gT/gN:.2f}  (1.0 linear, {gN:.1f} quadratic)")
    print(f"   feature widths per (channel,degree): {mdl.attn.widths}")
    ok = gT / gN < 2.0
    print(OK if ok else BAD)
    return ok


# ------------------------------------------------------------------ 9
def check_head_mixture():
    print("9. learned per-head mixture is observable (no specialization required)")
    _, s_, _, _, _, p, s, x, mdl = _setup()
    a = mdl.attn.alpha.detach()
    names = []
    for c in range(mdl.n_inv):
        lbl = "dc" if c == 0 else f"{'re' if c % 2 else 'im'}_w{(c+1)//2}"
        for k in range(mdl.cfg.max_degree + 1):
            names.append(f"{lbl}^{k}")
    print(f"   alpha [heads x (channel,degree)] = {tuple(a.shape)}")
    for hh in range(a.shape[0]):
        top = torch.argsort(a[hh].abs(), descending=True)[:3]
        print(f"   head {hh}: " + ", ".join(
            f"{names[int(t)]}={a[hh, t]:+.3f}" for t in top))
    print("   (at init these reflect the symmetry-breaking scales only)")
    ok = a.shape == (mdl.cfg.heads, mdl.attn.n_blocks)
    print(OK if ok else BAD)
    return ok


def main() -> int:
    checks = [check_factorized_vs_naive, check_factorized_vs_naive_grads,
              check_capacitance_gradient, check_permutation,
              check_basis_invariance, check_directionality, check_no_n2,
              check_scaling, check_head_mixture]
    res = []
    for fn in checks:
        res.append(fn()); print()
    n = sum(bool(r) for r in res)
    print(f"== {n}/{len(res)} checks passed ==")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
