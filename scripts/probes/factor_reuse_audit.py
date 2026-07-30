"""How much accuracy does reusing the base circuit's factors cost?

The only way the surrogate is faster than simulating is amortization:
factorize the base circuit ONCE, then score many proposed changes using
those base factors plus updated local R/C node features. Per-proposal cost
then collapses to the model forward (132 ms dynamic / 32 ms simple at 1225
nodes) instead of factors + forward (3296 ms), which is 36x slower than the
simulator. See scripts/probes/inference_cost.py.

The cost of that trick is fidelity: with base factors, the modification is
visible to the model ONLY through the local R/C features. This measures how
much that costs, against re-simulation.

Two prediction paths for a modified circuit G':
  EXACT  — recompute factors from G'         (slow, what training saw)
  REUSE  — base factors from G, local R/C from G'   (fast, off-distribution)

Reported per modification family and magnitude:
  * absolute error of each path vs the simulator
  * the CHANGE  y' - y : predicted vs simulated, sign accuracy, correlation
  * degradation vs modification magnitude

    python scripts/probes/factor_reuse_audit.py --ckpt checkpoints/v9_dynamic_kernel_s0.pt
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
from tools.dataset_runner import SimConfig, run_one
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (branch_system, dc_symmetric_factor,
                                     impedance_factors, knob_tensors,
                                     local_rc_features, node_features)
from tools.sampler import (FIXED_CONSTANTS, FIXED_DUTY, FIXED_FREQ,
                           FIXED_I_PEAK, FIXED_PHASE, FIXED_R_VIA,
                           FIXED_RSHEET_BOT, FIXED_RSHEET_TOP,
                           GLOBAL_RANGES, sample_edge_widths)

FDT = torch.float64


def default_omegas(n_freq):
    b = 2 * np.pi * FIXED_FREQ
    return torch.tensor([0.0, b, 5 * b, 25 * b, 125 * b][:n_freq], dtype=FDT)


def load_model(ckpt):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfgd = dict(st.get("cfg", {}))
    cfgd.setdefault("invariant", False)
    cfgd.setdefault("local_rc", False)
    cfg = ImpAttnConfig(**cfgd)
    mdl = ImpedanceAttentionRegressor(cfg).to(FDT)
    mdl.load_state_dict(st["model"]); mdl.eval()
    for p in mdl.parameters():
        p.requires_grad = False
    return mdl, cfg, int(st.get("args", {}).get("n_power", 2))


def build(nt, nb, wt, wb, cd):
    proto = build_regular_pdn(n_top=nt, n_bot=nb)
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (proto.n_loads, 1))
    g = build_regular_pdn(n_top=nt, n_bot=nb, Rsheet_top=FIXED_RSHEET_TOP,
                          Rsheet_bot=FIXED_RSHEET_BOT, wire_width=0.5,
                          R_via=FIXED_R_VIA, C_decap=float(cd), freq=FIXED_FREQ,
                          loads=loads,
                          ww_top_edges=np.asarray(wt, dtype=float),
                          ww_bot_edges=np.asarray(wb, dtype=float))
    return g, branch_system(g), loads


def sim_loads(nt, nb, wt, wb, cd, n_loads):
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (n_loads, 1))
    p = dict(FIXED_CONSTANTS)
    p.update(n_top=nt, n_bot=nb, wire_width=0.5, C_decap=float(cd), loads=loads,
             ww_top_edges=np.asarray(wt, dtype=float),
             ww_bot_edges=np.asarray(wb, dtype=float))
    return np.asarray(run_one(p, cfg=SimConfig())["peak_droop_loads"], dtype=float)


def predict(mdl, cfg, s_, g, loads, wt, wb, cd, om, n_power, base=None):
    """base=None -> EXACT (recompute factors). base=(p,s,fdc) -> REUSE."""
    R, C = knob_tensors(g, torch.tensor(wt, dtype=FDT),
                        torch.tensor(wb, dtype=FDT),
                        torch.tensor(float(cd), dtype=FDT),
                        FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
    x = node_features(s_, torch.tensor(loads, dtype=FDT))
    if cfg.local_rc:
        x = torch.cat([x, local_rc_features(s_, R, C)], -1)
    if base is None:
        p, s = impedance_factors(s_, R, C, om, m=cfg.m_factor, n_power=n_power)
        fdc = dc_symmetric_factor(s_, R, C, m=cfg.m_factor, n_power=n_power)
    else:
        p, s, fdc = base
    with torch.no_grad():
        return (10.0 ** mdl(x, p, s, s_.n_elec, fdc=fdc)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints/v9_dynamic_kernel_s0.pt"))
    ap.add_argument("--anchors", nargs="*", default=["4,7", "7,13", "11,31"])
    ap.add_argument("--n-designs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    mdl, cfg, n_power = load_model(args.ckpt)
    om = default_omegas(cfg.n_freq)
    print(f"ckpt {args.ckpt.name}  score={cfg.score}  local_rc={cfg.local_rc}  "
          f"m={cfg.m_factor}")
    if not cfg.local_rc:
        print("  ! local_rc=False: with base factors the model cannot see the")
        print("    modification AT ALL, so REUSE is a lower bound, not a test.")

    rng = np.random.default_rng(0)
    cd_p = GLOBAL_RANGES.by_name("C_decap")
    MODS = [("ww_edge", 0.10), ("ww_edge", 0.25), ("ww_edge", 0.50),
            ("ww_strap", 0.25), ("decap", 2.0), ("decap", 0.5)]
    rows = []
    for a in args.anchors:
        nt, nb = (int(v) for v in a.split(","))
        print(f"\n=== anchor ({nt},{nb}) ===")
        print(f"{'mod':>10} {'mag':>6} | {'exact relerr':>13} {'reuse relerr':>13} | "
              f"{'d sign ex':>10} {'d sign re':>10} | {'d rho ex':>9} {'d rho re':>9}")
        acc = {}
        for kind, mag in MODS:
            ex_e, re_e, ds_ex, ds_re, dr_ex, dr_re = [], [], [], [], [], []
            for _ in range(args.n_designs):
                wt0, wb0 = sample_edge_widths(nt, rng, n_bot=nb)
                cd0 = float(np.exp(rng.uniform(np.log(cd_p.lo), np.log(cd_p.hi))))
                g, s_, loads = build(nt, nb, wt0, wb0, cd0)
                bR, bC = knob_tensors(g, torch.tensor(wt0, dtype=FDT),
                                      torch.tensor(wb0, dtype=FDT),
                                      torch.tensor(cd0, dtype=FDT),
                                      FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                                      FIXED_R_VIA)
                bp, bs = impedance_factors(s_, bR, bC, om, m=cfg.m_factor,
                                           n_power=n_power)
                bf = dc_symmetric_factor(s_, bR, bC, m=cfg.m_factor,
                                         n_power=n_power)
                base = (bp, bs, bf)

                y0 = sim_loads(nt, nb, wt0, wb0, cd0, g.n_loads)
                p0 = predict(mdl, cfg, s_, g, loads, wt0, wb0, cd0, om,
                             n_power, base=None)

                wt1, wb1, cd1 = wt0.copy(), wb0.copy(), cd0
                if kind == "ww_edge":
                    wb1[rng.integers(wb1.size)] *= (1 + mag)
                elif kind == "ww_strap":
                    col = rng.integers(nb)
                    src = g.bot_edges[:, 0]
                    vert = (g.bot_edges[:, 1] - src) == nb
                    idx = np.nonzero(vert & (src % nb == col))[0]
                    if idx.size == 0:
                        continue
                    wb1[idx] *= (1 + mag)
                else:
                    cd1 = cd0 * mag
                g1, s1, loads1 = build(nt, nb, wt1, wb1, cd1)
                y1 = sim_loads(nt, nb, wt1, wb1, cd1, g1.n_loads)
                p1ex = predict(mdl, cfg, s1, g1, loads1, wt1, wb1, cd1, om,
                               n_power, base=None)
                p1re = predict(mdl, cfg, s1, g1, loads1, wt1, wb1, cd1, om,
                               n_power, base=base)

                ex_e.append(np.abs(p1ex - y1).mean() / y1.mean())
                re_e.append(np.abs(p1re - y1).mean() / y1.mean())
                dtrue = y1 - y0
                live = np.abs(dtrue) / y0 > 1e-6
                if live.sum() >= 3:
                    for pv, sg, rr in ((p1ex, ds_ex, dr_ex), (p1re, ds_re, dr_re)):
                        dpred = pv - p0
                        sg.append(float((np.sign(dpred[live]) ==
                                         np.sign(dtrue[live])).mean()))
                        rr.append(float(spearmanr(dpred[live],
                                                  dtrue[live]).statistic))
            if not ex_e:
                continue
            f = lambda v: float(np.mean(v)) if v else float("nan")
            print(f"{kind:>10} {mag:>6.2f} | {f(ex_e):>13.4f} {f(re_e):>13.4f} | "
                  f"{f(ds_ex):>10.3f} {f(ds_re):>10.3f} | "
                  f"{f(dr_ex):>9.3f} {f(dr_re):>9.3f}")
            rows.append(dict(anchor=[nt, nb], mod=kind, mag=mag,
                             exact_relerr=f(ex_e), reuse_relerr=f(re_e),
                             d_sign_exact=f(ds_ex), d_sign_reuse=f(ds_re),
                             d_rho_exact=f(dr_ex), d_rho_reuse=f(dr_re)))

    print("\nrelerr = mean |pred - sim| / mean sim, per-load.")
    print("d sign / d rho = sign accuracy and Spearman of the predicted CHANGE")
    print("(pred' - pred) against the simulated change (sim' - sim), per load,")
    print("over loads whose simulated change clears 1e-6 relative.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
