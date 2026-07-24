"""Sensitivity gate: is the surrogate's ∂droop/∂knob trustworthy enough
for gradient-based repair?

The repair loop consumes derivatives, not absolute droop. This gate
measures, against re-simulation, the three properties backprop-repair
needs, per anchor (never aggregated — v7 lesson):

  1. SIGN     — sign(model Δ) vs sign(sim Δ) for single-edge width
                changes and global decap changes;
  2. RANKING  — within a design, Spearman across perturbed edges of
                (model Δ, sim Δ): does the model order repair sites
                correctly?
  3. MAGNITUDE — median |model Δ / sim Δ| (step-size calibration; least
                critical, a verifier/line-search absorbs it).

Plus a model-internal linearity check: autograd grad·ΔR vs the model's
own finite difference (is one backward pass a valid proposer?).

Pass bar (per anchor): sign ≥ 95 %, mean site-ranking Spearman ≥ 0.8.

    python scripts/sensitivity_gate.py --ckpt checkpoints/droop_v7_edgeconv.pt \\
        --conv-type edgeconv --out docs/analysis/sensitivity_gate_edgeconv.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eason import EncoderConfig, PDNDroopRegressor
from tools.dataset_runner import SimConfig, run_one
from tools.grid_construction import EDGE_ATTR_COLS, EDGE_ATTR_DIM, build_regular_pdn, to_hetero_data
from tools.sampler import (
    FIXED_CONSTANTS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    FIXED_R_VIA,
    FIXED_RSHEET_BOT,
    FIXED_RSHEET_TOP,
    GLOBAL_RANGES,
    sample_edge_widths,
)

R_COL = EDGE_ATTR_COLS.index("R")
C_COL = EDGE_ATTR_COLS.index("C")
I_COL = EDGE_ATTR_COLS.index("I_peak")
F_COL = EDGE_ATTR_COLS.index("freq")
D_COL = EDGE_ATTR_COLS.index("duty")
P_COL = EDGE_ATTR_COLS.index("phase")

SIM_DELTA_FLOOR = 1e-4   # |sim Δ|/droop below this: excluded (solver noise)


def build_batch(g, ww_top, ww_bot, C_decap):
    """HeteroData batch matching pyg_dataset conventions exactly."""
    data = to_hetero_data(g)
    R_top = FIXED_RSHEET_TOP * g.pitch_top / ww_top
    R_bot = FIXED_RSHEET_BOT * g.pitch_bot / ww_bot

    def strap(Rdir):
        a = torch.zeros(Rdir.shape[0], EDGE_ATTR_DIM)
        a[:, R_COL] = Rdir
        return a

    data["mesh_top", "strap", "mesh_top"].edge_attr = strap(torch.cat([R_top, R_top]))
    data["mesh_bot", "strap", "mesh_bot"].edge_attr = strap(torch.cat([R_bot, R_bot]))
    nvia = data["mesh_top", "via", "mesh_bot"].edge_index.size(1)
    rv = torch.full((nvia,), float(FIXED_R_VIA))
    data["mesh_top", "via", "mesh_bot"].edge_attr = strap(rv)
    data["mesh_bot", "via", "mesh_top"].edge_attr = strap(rv.clone())
    dec = torch.zeros(data["mesh_bot", "decap", "mesh_bot"].edge_index.size(1), EDGE_ATTR_DIM)
    dec[:, C_COL] = C_decap
    data["mesh_bot", "decap", "mesh_bot"].edge_attr = dec
    nload = data["mesh_bot", "load", "mesh_bot"].edge_index.size(1)
    ld = torch.zeros(nload, EDGE_ATTR_DIM)
    ld[:, I_COL] = FIXED_I_PEAK; ld[:, F_COL] = FIXED_FREQ
    ld[:, D_COL] = FIXED_DUTY;   ld[:, P_COL] = FIXED_PHASE
    data["mesh_bot", "load", "mesh_bot"].edge_attr = ld
    data["y"] = torch.zeros(g.n_loads, dtype=torch.float32)
    return Batch.from_data_list([data])


def model_worst(model, g, ww_top, ww_bot, C_decap):
    return (10.0 ** model(build_batch(g, ww_top, ww_bot, C_decap))).max()


def sim_worst(n_top, n_bot, ww_top, ww_bot, C_decap, n_loads):
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (n_loads, 1))
    p = dict(FIXED_CONSTANTS)
    p.update(n_top=n_top, n_bot=n_bot, wire_width=0.5,  # placeholder; per-edge wins
             C_decap=C_decap, loads=loads,
             ww_top_edges=np.asarray(ww_top, dtype=np.float64),
             ww_bot_edges=np.asarray(ww_bot, dtype=np.float64))
    return float(run_one(p, cfg=SimConfig())["peak_droop_loads"].max())


def gate_anchor(model, n_top, n_bot, n_designs, k_bot, k_top, delta_ww,
                rng, verbose=True):
    from scipy.stats import spearmanr

    g = build_regular_pdn(n_top=n_top, n_bot=n_bot)
    n_tp, n_bp = g.top_edges.shape[0], g.bot_edges.shape[0]
    cd_p = GLOBAL_RANGES.by_name("C_decap")

    rows = []
    for d_i in range(n_designs):
        wt, wb = sample_edge_widths(n_top, rng, n_bot=n_bot)
        cd = float(np.exp(rng.uniform(np.log(cd_p.lo), np.log(cd_p.hi))))
        wt_t, wb_t = torch.from_numpy(wt).float(), torch.from_numpy(wb).float()

        base_sim = sim_worst(n_top, n_bot, wt, wb, cd, g.n_loads)
        # autograd at baseline (for the linearity check)
        wt_g = wt_t.clone().requires_grad_(True)
        wb_g = wb_t.clone().requires_grad_(True)
        w0 = model_worst(model, g, wt_g, wb_g, cd)
        w0.backward()
        grad_top, grad_bot = wt_g.grad.numpy(), wb_g.grad.numpy()
        base_model = float(w0.detach())

        edges = ([("bot", int(e)) for e in rng.choice(n_bp, min(k_bot, n_bp), replace=False)]
                 + [("top", int(e)) for e in rng.choice(n_tp, min(k_top, n_tp), replace=False)])
        for tier, e in edges:
            wt2, wb2 = wt.copy(), wb.copy()
            (wt2 if tier == "top" else wb2)[e] *= (1.0 + delta_ww)
            with torch.no_grad():
                dm = float(model_worst(model, g, torch.from_numpy(wt2).float(),
                                       torch.from_numpy(wb2).float(), cd)) - base_model
            ds = sim_worst(n_top, n_bot, wt2, wb2, cd, g.n_loads) - base_sim
            glin = (grad_top if tier == "top" else grad_bot)[e] * \
                   ((wt2 if tier == "top" else wb2)[e] - (wt if tier == "top" else wb)[e])
            rows.append(dict(design=d_i, kind=f"ww_{tier}", edge=e,
                             d_model=dm, d_sim=ds, d_lin=float(glin),
                             base_sim=base_sim))
        for fac in (2.0, 0.5):
            with torch.no_grad():
                dm = float(model_worst(model, g, wt_t, wb_t, cd * fac)) - base_model
            ds = sim_worst(n_top, n_bot, wt, wb, cd * fac, g.n_loads) - base_sim
            rows.append(dict(design=d_i, kind=f"decap_x{fac}", edge=-1,
                             d_model=dm, d_sim=ds, d_lin=float("nan"),
                             base_sim=base_sim))

    # ---- metrics ----
    ww = [r for r in rows if r["kind"].startswith("ww")]
    dec = [r for r in rows if r["kind"].startswith("decap")]
    live = [r for r in ww if abs(r["d_sim"]) / r["base_sim"] > SIM_DELTA_FLOOR]
    sign_ok = [np.sign(r["d_model"]) == np.sign(r["d_sim"]) for r in live]
    dec_live = [r for r in dec if abs(r["d_sim"]) / r["base_sim"] > SIM_DELTA_FLOOR]
    dec_ok = [np.sign(r["d_model"]) == np.sign(r["d_sim"]) for r in dec_live]
    rhos, lin_errs = [], []
    for d_i in range(n_designs):
        sub = [r for r in ww if r["design"] == d_i]
        rhos.append(float(spearmanr([r["d_model"] for r in sub],
                                    [r["d_sim"] for r in sub]).statistic))
        fd = np.array([r["d_model"] for r in sub])
        ln = np.array([r["d_lin"] for r in sub])
        lin_errs.append(float(np.median(np.abs(ln - fd) / (np.abs(fd) + 1e-12))))
    ratios = [abs(r["d_model"]) / abs(r["d_sim"]) for r in live]
    out = {
        "anchor": [n_top, n_bot],
        "n_ww_perturb": len(ww), "n_ww_live": len(live),
        "ww_sign_acc": float(np.mean(sign_ok)) if sign_ok else None,
        "site_rank_spearman_mean": float(np.mean(rhos)),
        "site_rank_spearman_per_design": rhos,
        "magnitude_ratio_median": float(np.median(ratios)) if ratios else None,
        "decap_sign_acc": float(np.mean(dec_ok)) if dec_ok else None,
        "n_decap_live": len(dec_live),
        "autograd_linearity_medrelerr": float(np.median(lin_errs)),
    }
    if verbose:
        print(f"  ({n_top},{n_bot}): sign {out['ww_sign_acc']}, "
              f"rank-rho {out['site_rank_spearman_mean']:.3f} "
              f"(per-design {[f'{r:.2f}' for r in rhos]}), "
              f"mag-ratio {out['magnitude_ratio_median']:.2f}, "
              f"decap-sign {out['decap_sign_acc']}, "
              f"lin-err {out['autograd_linearity_medrelerr']:.2%}")
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/droop_v7_edgeconv.pt"))
    ap.add_argument("--conv-type", default="edgeconv",
                    choices=["admittance", "edgeconv", "edge_aware"])
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--anchors", nargs="*", default=["3,7", "4,7", "7,13"],
                    help="train (3,7); OOD interp (4,7); OOD transfer (7,13)")
    ap.add_argument("--n-designs", type=int, default=3)
    ap.add_argument("--k-bot", type=int, default=12, help="bot strap edges perturbed")
    ap.add_argument("--k-top", type=int, default=6, help="top strap edges perturbed")
    ap.add_argument("--delta-ww", type=float, default=0.25, help="width step (+25%%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = PDNDroopRegressor(
        EncoderConfig(hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                      conv_type=args.conv_type, drop_edge_p=0.0),
        target_space="log")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad = False

    rng = np.random.default_rng(args.seed)
    print(f"gate: {args.ckpt.name} ({args.conv_type}) | "
          f"{args.n_designs} designs x ({args.k_bot}+{args.k_top}) edges x "
          f"+{args.delta_ww:.0%} width, decap x2/x0.5")
    t0 = time.time()
    results = []
    for a in args.anchors:
        nt, nb = (int(v) for v in a.split(","))
        res, _ = gate_anchor(model, nt, nb, args.n_designs, args.k_bot,
                             args.k_top, args.delta_ww, rng)
        results.append(res)

    print(f"\n{'anchor':>8} | {'sign':>6} | {'rank-rho':>8} | {'mag':>5} | "
          f"{'decap':>6} | verdict")
    for r in results:
        ok = (r["ww_sign_acc"] or 0) >= 0.95 and r["site_rank_spearman_mean"] >= 0.8
        print(f"  {tuple(r['anchor'])!s:>7} | {r['ww_sign_acc']:.2f} | "
              f"{r['site_rank_spearman_mean']:>8.3f} | "
              f"{r['magnitude_ratio_median']:>5.2f} | {r['decap_sign_acc']:.2f} | "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"({time.time()-t0:.0f}s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"ckpt": str(args.ckpt), "conv_type": args.conv_type,
             "delta_ww": args.delta_ww, "results": results}, indent=2))
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
