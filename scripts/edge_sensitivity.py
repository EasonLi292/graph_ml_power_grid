"""Per-edge R sensitivity from the *current* surrogate — no retraining.

The claim under test (user's idea): given the trained model, the gradient
``∂(worst droop)/∂R_e`` already tells us how changing one edge's resistance
affects the predicted droop. Unlike the full per-edge optimizer (which drove many
edges far off-distribution and hallucinated), a single-edge sensitivity is
evaluated at a *uniform* in-distribution operating point and only takes a small
step, so it should stay trustworthy.

This script checks three things at a uniform baseline grid:

  (A) compute ∂(worst droop)/∂R_e for every strap edge (one backward pass);
  (B) self-consistency — does grad·ΔR match the surrogate's actual finite
      difference for a small one-edge change? (is the linearization valid?)
  (C) ground-truth anchor — the *aggregate* sensitivity ∂droop/∂(global width)
      from the surrogate vs the simulator's finite difference (the simulator
      only supports a global width, so this is the one piece we can validate
      against truth today; per-edge validation needs a per-edge solver).

Usage:
    python3.12 scripts/edge_sensitivity.py --n-top 3 --wire-width 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eason import EncoderConfig, PDNDroopRegressor
from tools.dataset_runner import SimConfig, run_one
from tools.grid_construction import (
    EDGE_ATTR_COLS,
    EDGE_ATTR_DIM,
    build_regular_pdn,
    to_hetero_data,
)
from tools.sampler import (
    FIXED_CONSTANTS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
)

R_COL = EDGE_ATTR_COLS.index("R")
C_COL = EDGE_ATTR_COLS.index("C")
I_COL = EDGE_ATTR_COLS.index("I_peak")
F_COL = EDGE_ATTR_COLS.index("freq")
D_COL = EDGE_ATTR_COLS.index("duty")
P_COL = EDGE_ATTR_COLS.index("phase")


def proto(n_top):
    return build_regular_pdn(
        n_top=n_top, n_bot=FIXED_CONSTANTS["n_bot"],
        Rsheet_top=FIXED_CONSTANTS["Rsheet_top"],
        Rsheet_bot=FIXED_CONSTANTS["Rsheet_bot"],
        wire_width=0.5, R_via=FIXED_CONSTANTS["R_via"],
        C_decap=1e-10, freq=FIXED_CONSTANTS["freq"],
    )


def build_from_R(g, R_top_phys, R_bot_phys, C_decap):
    """HeteroData batch with per-physical-edge strap R (bidir-tiled)."""
    data = to_hetero_data(g)
    nload = data["mesh_bot", "load", "mesh_bot"].edge_index.size(1)
    nvia = data["mesh_top", "via", "mesh_bot"].edge_index.size(1)

    def strap(Rdir):
        a = torch.zeros(Rdir.size(0), EDGE_ATTR_DIM)
        a[:, R_COL] = Rdir
        return a

    data["mesh_top", "strap", "mesh_top"].edge_attr = strap(torch.cat([R_top_phys, R_top_phys]))
    data["mesh_bot", "strap", "mesh_bot"].edge_attr = strap(torch.cat([R_bot_phys, R_bot_phys]))
    rv = torch.full((nvia,), float(FIXED_CONSTANTS["R_via"]))
    data["mesh_top", "via", "mesh_bot"].edge_attr = strap(rv)
    data["mesh_bot", "via", "mesh_top"].edge_attr = strap(rv.clone())

    dec = torch.zeros(data["mesh_bot", "decap", "mesh_bot"].edge_index.size(1), EDGE_ATTR_DIM)
    dec[:, C_COL] = C_decap
    data["mesh_bot", "decap", "mesh_bot"].edge_attr = dec

    ld = torch.zeros(nload, EDGE_ATTR_DIM)
    ld[:, I_COL] = FIXED_I_PEAK; ld[:, F_COL] = FIXED_FREQ
    ld[:, D_COL] = FIXED_DUTY;   ld[:, P_COL] = FIXED_PHASE
    data["mesh_bot", "load", "mesh_bot"].edge_attr = ld
    data["y"] = torch.zeros(nload, dtype=torch.float32)
    return Batch.from_data_list([data])


def worst_droop(model, g, R_top_phys, R_bot_phys, C_decap):
    return (10.0 ** model(build_from_R(g, R_top_phys, R_bot_phys, C_decap))).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/droop_v5_nocoord.pt"))
    ap.add_argument("--n-top", type=int, default=3)
    ap.add_argument("--wire-width", type=float, default=0.5)
    ap.add_argument("--C-decap", type=float, default=2e-10)
    ap.add_argument("--png", type=Path, default=Path("docs/figures/fig_edge_sensitivity.png"))
    args = ap.parse_args()

    model = PDNDroopRegressor(
        EncoderConfig(hidden_dim=64, n_layers=7, conv_type="admittance", drop_edge_p=0.0),
        target_space="log",
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad = False

    g = proto(args.n_top)
    n_tp, n_bp = g.top_edges.shape[0], g.bot_edges.shape[0]
    Rt0 = FIXED_CONSTANTS["Rsheet_top"] * g.pitch_top / args.wire_width
    Rb0 = FIXED_CONSTANTS["Rsheet_bot"] * g.pitch_bot / args.wire_width

    # ---- (A) per-edge gradient at the uniform operating point ----
    R_top = torch.full((n_tp,), Rt0, requires_grad=True)
    R_bot = torch.full((n_bp,), Rb0, requires_grad=True)
    w0 = worst_droop(model, g, R_top, R_bot, args.C_decap)
    w0.backward()
    g_top = R_top.grad.clone()       # ∂worst/∂R_e  (V/Ω)
    g_bot = R_bot.grad.clone()
    w0 = float(w0.detach())

    print(f"=== Per-edge R sensitivity (surrogate, no retrain) ===")
    print(f"n_top={args.n_top}  uniform width={args.wire_width}  "
          f"R_top0={Rt0:.3f}Ω R_bot0={Rb0:.3f}Ω  C_decap={args.C_decap:.1e}")
    print(f"baseline worst droop = {w0*1e3:.4f} mV")
    print(f"strap edges: {n_tp} top + {n_bp} bot\n")

    # rank bot edges by sensitivity (∂droop/∂R, positive = more R → more droop)
    order = np.argsort(-g_bot.numpy())[:6]
    print("most sensitive bot straps  (∂droop/∂R_e, mV per Ω):")
    for e in order:
        u, v = g.bot_edges[e]
        print(f"    edge {u:>2}-{v:<2}  ∂droop/∂R = {g_bot[e].item()*1e3:+.4f} mV/Ω")
    print()

    # ---- (B) self-consistency: grad·ΔR vs surrogate finite difference ----
    # perturb the single most-sensitive bot edge's R by a small relative step
    e = int(order[0])
    print(f"(B) one-edge finite-difference check on bot edge "
          f"{g.bot_edges[e][0]}-{g.bot_edges[e][1]} (most sensitive):")
    print(f"    {'ΔR/R':>8} | {'grad·ΔR (mV)':>13} | {'true Δ (mV)':>12} | {'rel.err':>7}")
    for frac in (0.02, 0.05, 0.10, 0.20):
        dR = frac * Rb0
        Rb = R_bot.detach().clone(); Rb[e] += dR
        with torch.no_grad():
            wp = float(worst_droop(model, g, R_top.detach(), Rb, args.C_decap))
        lin = g_bot[e].item() * dR
        true = wp - w0
        rel = abs(lin - true) / (abs(true) + 1e-12)
        print(f"    {frac*100:6.0f}% | {lin*1e3:>13.5f} | {true*1e3:>12.5f} | {rel*100:6.1f}%")
    print("    (small rel.err = the gradient predicts a single-R change well)\n")

    # ---- (C) aggregate vs simulator (the validatable piece today) ----
    # surrogate ∂droop/∂width at global uniform width:
    wvar = torch.tensor(float(args.wire_width), requires_grad=True)
    Rt = FIXED_CONSTANTS["Rsheet_top"] * g.pitch_top / wvar
    Rb = FIXED_CONSTANTS["Rsheet_bot"] * g.pitch_bot / wvar
    ws = worst_droop(model, g, Rt.expand(n_tp), Rb.expand(n_bp), args.C_decap)
    ws.backward()
    d_surr = float(wvar.grad)        # ∂droop/∂width, surrogate

    # simulator finite difference on global width:
    def sim_worst(width):
        loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                        (g.n_loads, 1))
        p = dict(FIXED_CONSTANTS)
        p.update(wire_width=width, C_decap=args.C_decap, n_top=args.n_top, loads=loads)
        return float(run_one(p, cfg=SimConfig())["peak_droop_loads"].max())
    dw = 0.02
    d_sim = (sim_worst(args.wire_width + dw) - sim_worst(args.wire_width - dw)) / (2 * dw)
    print("(C) aggregate sensitivity ∂droop/∂(global width), ground-truth anchor:")
    print(f"    surrogate  = {d_surr*1e3:+.4f} mV per unit width")
    print(f"    simulator  = {d_sim*1e3:+.4f} mV per unit width   "
          f"(rel.err {abs(d_surr-d_sim)/(abs(d_sim)+1e-12)*100:.1f}%)")
    print("    [per-edge sensitivities can't be validated until the solver takes per-edge R]")

    _plot(g, g_bot, args.n_top, args.wire_width, w0, args.png)


def _plot(g, g_bot, n_top, width, w0, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import cm, colors

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    s = g_bot.numpy() * 1e3
    vmax = float(np.abs(s).max())
    norm = colors.Normalize(vmin=0, vmax=vmax)
    segs = [[g.bot_pos[u], g.bot_pos[v]] for u, v in g.bot_edges]
    lc = LineCollection(segs, cmap=cm.inferno, norm=norm,
                        linewidths=2 + 6 * norm(np.abs(s)))
    lc.set_array(np.abs(s))
    ax.add_collection(lc)
    ax.scatter(g.bot_pos[:, 0], g.bot_pos[:, 1], s=8, c="0.6")
    fig.colorbar(lc, ax=ax, label="|∂droop/∂R_e|  (mV/Ω)", shrink=0.8)
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_title(f"Per-edge R sensitivity, M_bot  (n_top={n_top}, width={width})\n"
                 f"baseline worst droop {w0*1e3:.3f} mV — brighter/thicker = widening "
                 f"this edge helps most")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)
    print(f"\n  figure → {png}")


if __name__ == "__main__":
    main()
