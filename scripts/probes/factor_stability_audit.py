"""Factor-stability and reciprocity audit.

Does the reduced-rank impedance factorization's reciprocity defect actually
matter? Answer it before changing the factorization.

The physical transfer impedance of a passive R/C network is reciprocal
(Z_ij = Z_ji). The learned attention score is NOT required to be, and this
audit never touches it — any correction here applies to the physical
representation only.

Variants compared:
  A  hermitian     Z ~= Qr (Qr^H Z Qr) Qr^H     -- current, width m
  B  symmetrized   (Z_hat + Z_hat^T)/2 in factored form, width 2m
  C  complex_sym   Z ~= Qr (Qr^T Z Qr) Qr^T     -- symmetric, width m,
                   but not a Galerkin projection
  D  exact         hermitian at m = n_free, the reference

Measured, per retained frequency and per rank:
  1 reciprocity        ||Z_hat - Z_hat^T|| / ||Z_hat||
  2 reconstruction     ||Z_hat - Z_exact|| / ||Z_exact||   (small circuits)
  3 channel stability  invariant channels across probe seeds
  4 prediction         model output across probe seeds
  5 sensitivity        d(worst)/d(ww) and d(worst)/d(C) across probe seeds
  6 attention          learned score matrix across probe seeds
plus factor width, runtime and memory.

    python scripts/probes/factor_stability_audit.py
    python scripts/probes/factor_stability_audit.py --out docs/analysis/factor_audit.json
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
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (admittance, branch_system,
                                     impedance_factors, invariant_channels,
                                     knob_tensors, node_features)
from tools.sampler import (FIXED_DUTY, FIXED_FREQ, FIXED_I_PEAK, FIXED_PHASE,
                           FIXED_R_VIA, FIXED_RSHEET_BOT, FIXED_RSHEET_TOP)

DT = torch.float64
VARIANTS = {"A_hermitian": "hermitian", "B_symmetrized": "symmetrized",
            "C_complex_sym": "complex_sym"}
SEEDS = (0, 101, 202, 303)


def R_C(g, wt, wb, cd):
    return knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                        FIXED_R_VIA)


def omegas(n=3):
    b = 2 * np.pi * FIXED_FREQ
    return torch.tensor([0.0, b, 5 * b, 25 * b][:n], dtype=DT)


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


def exact_Z(g, s_, wt, wb, cd, om):
    """Dense per-frequency Z over free nodes."""
    R, C = R_C(g, wt, wb, cd)
    out = []
    for f in range(om.shape[0]):
        Y = admittance(s_, R, C, om[f])
        out.append(torch.linalg.inv(Y))
    return out


def per_freq_Zhat(s_, chans, om):
    """Invariant channels -> per-frequency complex Z over ELECTRICAL nodes."""
    live = s_.free_of >= 0
    idx = s_.free_of[live]
    order = torch.argsort(idx)
    sel = torch.nonzero(live).squeeze(-1)[order]
    zs, k = [], 0
    for f in range(om.shape[0]):
        if float(om[f]) == 0.0:
            A, B, _ = chans[k]; k += 1
            z = torch.einsum("id,jd->ij", A[sel], B[sel]).to(torch.complex128)
        else:
            Ar, Br, _ = chans[k]; Ai, Bi, _ = chans[k + 1]; k += 2
            z = (torch.einsum("id,jd->ij", Ar[sel], Br[sel]).to(torch.complex128)
                 + 1j * torch.einsum("id,jd->ij", Ai[sel], Bi[sel]).to(torch.complex128))
        zs.append(z)
    return zs


def relnorm(a, b=None):
    if b is None:
        return float(a.abs().pow(2).sum().sqrt())
    return float((a - b).abs().pow(2).sum().sqrt()
                 / b.abs().pow(2).sum().sqrt().clamp_min(1e-300))


def make_model(m_dc, m_ac, n_freq, score, seed=0):
    torch.manual_seed(seed)
    cfg = ImpAttnConfig(hidden_dim=32, heads=3, d_v=16, n_freq=n_freq,
                        m_factor=m_dc, score=score)
    mdl = ImpedanceAttentionRegressor(cfg, init_bias=-3.6).to(DT)
    if score == "dynamic_kernel":
        # rebuild attention with the real per-channel widths
        from eason.impedance_attention_model import DynamicKernelAttention
        dims = [m_dc] + [2 * m_ac] * (mdl.n_inv - 1)
        torch.manual_seed(seed)
        mdl.attn = DynamicKernelAttention(cfg, dims).to(DT)
    with torch.no_grad():
        for prm in mdl.decoder.parameters():
            prm.copy_(torch.randn_like(prm) * 0.2)
    return mdl


def audit_circuit(nt, nb, ranks, om, want_model=True):
    g, s_, wt, wb, cd = setup(nt, nb)
    x = node_features(s_, torch.tensor(g.loads, dtype=DT))
    Zex = exact_Z(g, s_, wt, wb, cd, om)
    rows = []
    for vname, proj in VARIANTS.items():
        for m in ranks:
            per_seed = {"pred": [], "gw": [], "gc": [], "attn": [], "chan": []}
            recip, recon, width, t_fac = None, None, None, None
            for si, seed in enumerate(SEEDS):
                t0 = time.time()
                R0, C0 = R_C(g, wt, wb, cd)
                p, s = impedance_factors(s_, R0, C0, om, m=m,
                                         n_power=2, seed=seed, proj=proj)
                t_fac = (time.time() - t0) if si == 0 else t_fac
                chans = invariant_channels(p, s, om)
                zh = per_freq_Zhat(s_, chans, om)
                if si == 0:
                    recip = [float((z - z.T).abs().pow(2).sum().sqrt()
                                   / z.abs().pow(2).sum().sqrt().clamp_min(1e-300))
                             for z in zh]
                    recon = [relnorm(z, ze) for z, ze in zip(zh, Zex)]
                    width = int(p.shape[-1])
                per_seed["chan"].append(torch.cat(
                    [torch.einsum("id,jd->ij", A, B).flatten() for A, B, _ in chans]))
                if want_model:
                    wid = p.shape[-1]
                    for sc_name in ("bilinear", "dynamic_kernel"):
                        mdl = make_model(wid, wid, om.shape[0], sc_name)
                        wt_g = wt.clone().requires_grad_(True)
                        cd_g = cd.clone().requires_grad_(True)
                        Rg, Cg = knob_tensors(g, wt_g, wb, cd_g,
                                             FIXED_RSHEET_TOP,
                                             FIXED_RSHEET_BOT, FIXED_R_VIA)
                        pg, sg = impedance_factors(s_, Rg, Cg, om, m=m,
                                                   n_power=2, seed=seed,
                                                   proj=proj)
                        y = (10.0 ** mdl(x, pg, sg, s_.n_elec)).max()
                        gw, gc = torch.autograd.grad(y, [wt_g, cd_g])
                        per_seed.setdefault(f"pred_{sc_name}", []).append(float(y.detach()))
                        per_seed.setdefault(f"gw_{sc_name}", []).append(gw.clone())
                        per_seed.setdefault(f"gc_{sc_name}", []).append(float(gc))
                    mdl = make_model(wid, wid, om.shape[0], "bilinear")
                    per_seed["pred"] = per_seed["pred_bilinear"]
                    per_seed["gw"] = per_seed["gw_bilinear"]
                    per_seed["gc"] = per_seed["gc_bilinear"]
                    with torch.no_grad():
                        hs, pn, sn = mdl.embed(x, p, s, s_.n_elec)
                        P, S = mdl.attn._factor_terms(hs, pn, sn)
                        q = mdl.attn.q(hs).view(hs.shape[0], 3, -1)
                        k = mdl.attn.k(hs).view(hs.shape[0], 3, -1)
                        sc = (torch.einsum("ihd,jhd->hij", q, k)
                              * torch.einsum("ihd,jhd->hij", P, S))
                        per_seed["attn"].append(sc.flatten())
            row = dict(anchor=[nt, nb], variant=vname, m=m, width=width,
                       factor_ms=t_fac * 1e3,
                       reciprocity=recip, reconstruction=recon)
            row["chan_spread"] = seed_spread(per_seed["chan"])
            if want_model:
                pr = np.array(per_seed["pred"])
                row["pred_spread_rel"] = float(pr.std() / max(abs(pr.mean()), 1e-30))
                row["gw_spread"] = seed_spread(per_seed["gw"])
                gc = np.array(per_seed["gc"])
                row["gc_spread_rel"] = float(gc.std() / max(abs(gc.mean()), 1e-30))
                row["attn_spread"] = seed_spread(per_seed["attn"])
                for sc_name in ("bilinear", "dynamic_kernel"):
                    pr = np.array(per_seed[f"pred_{sc_name}"])
                    gc2 = np.array(per_seed[f"gc_{sc_name}"])
                    row[f"pred_spread_{sc_name}"] = float(pr.std() / max(abs(pr.mean()), 1e-30))
                    row[f"gw_spread_{sc_name}"] = seed_spread(per_seed[f"gw_{sc_name}"])
                    row[f"gc_spread_{sc_name}"] = float(gc2.std() / max(abs(gc2.mean()), 1e-30))
            rows.append(row)
    return rows


def seed_spread(vs):
    """Max pairwise relative difference across probe seeds."""
    if len(vs) < 2:
        return float("nan")
    st = torch.stack([v.flatten().to(DT) for v in vs])
    ref = st.abs().max().clamp_min(1e-300)
    return float((st.max(0).values - st.min(0).values).abs().max() / ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-freq", type=int, default=3)
    args = ap.parse_args()
    om = omegas(args.n_freq)
    print(f"probe seeds {SEEDS}   frequencies {[f'{float(w):.2e}' for w in om]}")

    all_rows = []
    # small circuit: exact rank reachable -> variant D reference
    for nt, nb, ranks in ((3, 7, (8, 16, 32, 55)), (7, 13, (8, 16, 32))):
        g, s_, *_ = setup(nt, nb)
        print(f"\n=== anchor ({nt},{nb})  n_free={s_.n_free} ===")
        rows = audit_circuit(nt, nb, ranks, om)
        all_rows += rows
        hdr = (f"{'variant':>14} {'m':>4} {'wid':>4} {'ms':>6} | "
               f"{'recipAC':>9} {'reconAC':>9} | {'chan':>8} | "
               f"{'d/dww bil':>9} {'d/dww dyn':>9} | {'d/dC bil':>8} {'d/dC dyn':>8}")
        print(hdr)
        for r in rows:
            rc, rn = r["reciprocity"], r["reconstruction"]
            ac_rc = max(rc[1:]) if len(rc) > 1 else float("nan")
            ac_rn = max(rn[1:]) if len(rn) > 1 else float("nan")
            tag = f"{r['variant']}"
            if r["m"] == s_.n_free:
                tag = "D_exact"
            print(f"{tag:>14} {r['m']:>4} {r['width']:>4} {r['factor_ms']:>6.0f} | "
                  f"{ac_rc:>9.1e} {ac_rn:>9.1e} | {r['chan_spread']:>8.1e} | "
                  f"{r['gw_spread_bilinear']:>9.1e} {r['gw_spread_dynamic_kernel']:>9.1e} | "
                  f"{r['gc_spread_bilinear']:>8.1e} {r['gc_spread_dynamic_kernel']:>8.1e}")

    print("\nreciprocity = ||Z-Z^T||/||Z||;  recon = vs dense inverse;")
    print("chan/pred/d-dww/d-dC/attn = max spread across the 4 probe seeds.")
    print("\nDecision rule: if the same circuit gives materially different")
    print("predictions or sensitivities only because the probe seed changed,")
    print("fix the physical representation before training.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"seeds": list(SEEDS), "omegas": [float(w) for w in om],
             "rows": all_rows}, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
