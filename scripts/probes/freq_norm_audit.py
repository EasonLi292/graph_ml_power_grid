"""Per-frequency normalization audit for the dynamic kernel.

Question: does dividing each frequency's factors by its own invariant
impedance scale s_w = sqrt(mean_loads |Z_ii(w)|^2) align the frequency
ranges without breaking anything? Measured, no training run.

  1 feature p5 / median / p95 / max, per frequency and topology
  2 relative scale of DC vs every AC frequency
  3 gradient norms w.r.t. wire width and capacitance
  4 stability across factor probe seeds
  5 factorized == direct after normalization
  6 scaling with graph size
  7 does the physics init let one head or frequency dominate immediately

Reported for freq_norm=False vs True on small and large topologies.

    python scripts/probes/freq_norm_audit.py
    python scripts/probes/freq_norm_audit.py --out docs/analysis/freq_norm_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.stats import spearmanr

from eason.impedance_attention_model import (ImpAttnConfig,
                                             ImpedanceAttentionRegressor)
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (branch_system, impedance_factors,
                                     knob_tensors, node_features)
from tools.sampler import (FIXED_DUTY, FIXED_FREQ, FIXED_I_PEAK, FIXED_PHASE,
                           FIXED_R_VIA, FIXED_RSHEET_BOT, FIXED_RSHEET_TOP)

DT = torch.float64
M, NFREQ = 16, 3
SEEDS = (0, 101, 202, 303)
ANCHORS = ((3, 7), (7, 13), (13, 25))


def omegas(n=NFREQ):
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


def model(freq_norm, seed=0, heads=4, mode="frob"):
    torch.manual_seed(seed)
    cfg = ImpAttnConfig(hidden_dim=32, heads=heads, d_v=16, n_freq=NFREQ,
                        m_factor=M, score="dynamic_kernel",
                        freq_norm=freq_norm, freq_norm_mode=mode)
    mdl = ImpedanceAttentionRegressor(cfg, init_bias=-3.6).to(DT)
    with torch.no_grad():                       # non-degenerate head
        for prm in mdl.decoder.parameters():
            prm.copy_(torch.randn_like(prm) * 0.2)
    return mdl


def block_names(mdl):
    out = []
    for c in range(mdl.n_inv):
        lbl = "dc" if c == 0 else f"{'re' if c % 2 else 'im'}_w{(c + 1) // 2}"
        for k in range(mdl.cfg.max_degree + 1):
            out.append(f"{lbl}^{k}")
    return out


def factors(g, s_, wt, wb, cd, seed=0):
    R, C = knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                        FIXED_R_VIA)
    return impedance_factors(s_, R, C, omegas(), m=M, n_power=2, seed=seed)


def block_pair_stats(mdl, x, p, s, n_elec, n_pairs=4000):
    """Per-(channel,degree) pairwise score contribution, on sampled pairs.

    O(N) features; only a random subset of PAIRS is materialised, so no
    [N, N] tensor is ever built.
    """
    hstate, chans = mdl.embed_invariant(x, p, s, n_elec)
    N = hstate.shape[0]
    gen = torch.Generator().manual_seed(7)
    ii = torch.randint(0, N, (n_pairs,), generator=gen)
    jj = torch.randint(0, N, (n_pairs,), generator=gen)
    stats = {}
    names = block_names(mdl)
    b = 0
    for A, B, _ in chans:
        z = (A[ii] * B[jj]).sum(-1)
        for k in range(mdl.cfg.max_degree + 1):
            v = torch.ones_like(z) if k == 0 else z ** k
            av = v.abs().detach().numpy()
            # clamped pad nodes have structurally zero factors, so a chunk of
            # sampled pairs is exactly 0. Those are not a numerical tail --
            # report their fraction and take percentiles over the live pairs.
            nz = av[av > 0]
            if nz.size == 0:
                nz = av
            stats[names[b]] = dict(
                zero_frac=float((av == 0).mean()),
                p5=float(np.percentile(nz, 5)),
                med=float(np.median(nz)),
                p95=float(np.percentile(nz, 95)),
                mx=float(nz.max()))
            b += 1
    return stats, hstate, chans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []

    for nt, nb in ANCHORS:
        g, s_, wt, wb, cd = setup(nt, nb)
        x = node_features(s_, torch.tensor(g.loads, dtype=DT))
        print(f"\n{'='*78}\nanchor ({nt},{nb})  n_free={s_.n_free}  "
              f"N={s_.n_elec + s_.n_loads}")
        p, s = factors(g, s_, wt, wb, cd)

        for fn, mode in ((False, "-"), (True, "diag"), (True, "frob")):
            mdl = model(fn, mode=mode)
            st, hstate, chans = block_pair_stats(mdl, x, p, s, s_.n_elec)
            names = block_names(mdl)
            tag = f"norm={mode:>4}" if fn else "norm= OFF"
            print(f"\n  {tag}  -- 1/2. per-block pairwise |contribution|")
            print(f"  {'block':>9} {'zero%':>6} {'p5':>10} {'median':>10} "
                  f"{'p95':>10} {'max':>10} {'p95/p5':>8} {'max/med':>8}")
            for n in names:
                if n.endswith("^0"):
                    continue                      # constant block, nothing to see
                d = st[n]
                print(f"  {n:>9} {100*d['zero_frac']:>6.1f} {d['p5']:>10.2e} "
                      f"{d['med']:>10.2e} {d['p95']:>10.2e} {d['mx']:>10.2e} "
                      f"{d['p95']/max(d['p5'],1e-300):>8.1f} "
                      f"{d['mx']/max(d['med'],1e-300):>8.1f}")
            # 2. cross-FREQUENCY alignment, like-for-like (same part, same degree)
            for part in ("re", "im"):
                for k in (1, 2):
                    vals = [st[n]["med"] for n in names
                            if n.startswith(part) and n.endswith(f"^{k}")]
                    if len(vals) > 1:
                        print(f"     2. {part}^{k} medians across frequencies: "
                              + ", ".join(f"{v:.2e}" for v in vals)
                              + f"  -> ratio {max(vals)/max(min(vals),1e-300):.1f}")
            print(f"     2. DC^1 {st['dc^1']['med']:.2e} vs re_w1^1 "
                  f"{st['re_w1^1']['med']:.2e}  -> ratio "
                  f"{st['dc^1']['med']/max(st['re_w1^1']['med'],1e-300):.2f}")

            # 3. gradient norms
            wt_g = wt.clone().requires_grad_(True)
            cd_g = cd.clone().requires_grad_(True)
            R, C = knob_tensors(g, wt_g, wb, cd_g, FIXED_RSHEET_TOP,
                               FIXED_RSHEET_BOT, FIXED_R_VIA)
            pg, sg = impedance_factors(s_, R, C, omegas(), m=M, n_power=2)
            y = (10.0 ** mdl(x, pg, sg, s_.n_elec)).max()
            gw, gc = torch.autograd.grad(y, [wt_g, cd_g])
            print(f"     3. |grad| ww {float(gw.norm()):.3e}   "
                  f"C {abs(float(gc)):.3e}   y {float(y):.3e}")

            # 4. probe-seed stability (prediction + gradient ranking)
            preds, gws = [], []
            for sd in SEEDS:
                wt_s = wt.clone().requires_grad_(True)
                cd_s = cd.clone().requires_grad_(True)
                Rs, Cs = knob_tensors(g, wt_s, wb, cd_s, FIXED_RSHEET_TOP,
                                     FIXED_RSHEET_BOT, FIXED_R_VIA)
                ps, ss = impedance_factors(s_, Rs, Cs, omegas(), m=M,
                                           n_power=2, seed=sd)
                ys = (10.0 ** mdl(x, ps, ss, s_.n_elec)).max()
                g1, = torch.autograd.grad(ys, [wt_s])
                preds.append(float(ys.detach())); gws.append(g1.numpy())
            pr = np.array(preds)
            rho = float(np.mean([spearmanr(gws[i], gws[j]).statistic
                                 for i in range(len(gws))
                                 for j in range(i + 1, len(gws))]))
            print(f"     4. across probe seeds: pred rel-sd "
                  f"{pr.std()/max(abs(pr.mean()),1e-30):.3e}, "
                  f"d/dww rank rho {rho:+.4f}")

            # 5. factorized == direct
            yf = mdl(x, p, s, s_.n_elec, naive=False)
            yn = mdl(x, p, s, s_.n_elec, naive=True)
            e = float((yf - yn).abs().max() / yn.abs().max())
            print(f"     5. factorized vs explicit [N,N]: {e:.2e}")

            # 7. head / frequency dominance at init
            Phi, Psi = mdl.attn._maps(hstate, chans)
            bo = mdl.attn.block_of
            share_h, share_b = [], np.zeros(mdl.attn.n_blocks)
            for h in range(mdl.cfg.heads):
                per_b = np.array([
                    float((Phi[:, h, bo == bi].abs().mean()
                           * Psi[:, h, bo == bi].abs().mean()))
                    for bi in range(mdl.attn.n_blocks)])
                share_h.append(per_b.sum()); share_b += per_b
            share_h = np.array(share_h) / max(np.sum(share_h), 1e-300)
            share_b = share_b / max(share_b.sum(), 1e-300)
            top = np.argsort(share_b)[::-1][:4]
            print(f"     7. head share {np.round(share_h, 3).tolist()}  "
                  f"(max {share_h.max():.3f})")
            print(f"        top blocks " + ", ".join(
                f"{names[t]}={share_b[t]:.3f}" for t in top))

            rows.append(dict(anchor=[nt, nb], freq_norm=fn, mode=mode, blocks=st,
                             grad_ww=float(gw.norm()), grad_C=abs(float(gc)),
                             pred_rel_sd=float(pr.std()/max(abs(pr.mean()),1e-30)),
                             grad_rank_rho=rho, factorized_err=e,
                             head_share=share_h.tolist(),
                             block_share=share_b.tolist(),
                             block_names=names))

    print(f"\n{'='*78}")
    print("6. scaling with graph size: read the per-anchor blocks above —")
    print("   with normalization ON the medians should be comparable across")
    print("   anchors; with it OFF they track absolute impedance, which grows")
    print("   with the grid.")
    print("\nCaveat to report separately: per-frequency scaling aligns the")
    print("frequency RANGES but does not remove the within-frequency tail.")
    print("No clipping or extra nonlinearity is applied — either would break")
    print("the factorization.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
