"""Localized inverse design: optimize *per-edge* strap resistances.

Motivation
----------
The global design loop (``design_grad.py``) optimizes a single ``wire_width``
that is broadcast to **every** strap edge — to fix one droop hotspot it has to
widen the whole grid. Here we instead make the wire width of **each strap edge an
independent variable** and ask: can the optimizer fix the worst-load droop by
thickening only the few edges near the hotspot, spending far less metal than a
uniform widen?

    variables : per-edge wire width  w_e ∈ [0.2, 1.0]  (top + bot straps)
                R_e = Rsheet · pitch / w_e          (differentiable)
    loss      : ReLU(worst_droop / target − 1)²  +  λ · Σ_e metal(w_e)
                metal(w_e) = (w_e · pitch_layer)   ∝ copper volume of segment e

C_decap and via R are held fixed (we only touch strap R, per the brief).

⚠ OOD CAVEAT. The surrogate was trained only on grids with a *uniform* per-layer
R (one knob → all straps). A heterogeneous per-edge R pattern is out of
distribution, and the simulator (`build_regular_pdn`) only accepts a scalar
wire_width, so this design **cannot yet be validated against ground truth**. This
script is a surrogate-only probe: it shows whether the optimization is wired up
and what the *surrogate believes* about localized fixes. Trusting the result
requires retraining on per-edge-varied R (the planned model change).

Usage:
    python3.12 scripts/design_edge.py --n-top 4 --wire-width-init 0.4
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
    GLOBAL_RANGES,
)

R_COL = EDGE_ATTR_COLS.index("R")
C_COL = EDGE_ATTR_COLS.index("C")
I_COL = EDGE_ATTR_COLS.index("I_peak")
F_COL = EDGE_ATTR_COLS.index("freq")
D_COL = EDGE_ATTR_COLS.index("duty")
P_COL = EDGE_ATTR_COLS.index("phase")

WW = GLOBAL_RANGES.by_name("wire_width")  # .lo / .hi


# -----------------------------------------------------------------------------
# Reparameterization: unconstrained z → per-edge wire width in [lo, hi]
# -----------------------------------------------------------------------------

def widths_from_z(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z) * (WW.hi - WW.lo) + WW.lo


def z_from_width(w: float) -> float:
    """Inverse of widths_from_z for a scalar (to initialize at a uniform width)."""
    frac = (w - WW.lo) / (WW.hi - WW.lo)
    frac = min(max(frac, 1e-4), 1 - 1e-4)
    return float(np.log(frac / (1 - frac)))


# -----------------------------------------------------------------------------
# Differentiable graph builder with PER-EDGE strap R
# -----------------------------------------------------------------------------

def _proto(n_top: int):
    """Build a prototype grid once to read topology + geometry."""
    g = build_regular_pdn(
        n_top=n_top,
        n_bot=FIXED_CONSTANTS["n_bot"],
        Rsheet_top=FIXED_CONSTANTS["Rsheet_top"],
        Rsheet_bot=FIXED_CONSTANTS["Rsheet_bot"],
        wire_width=0.5,
        R_via=FIXED_CONSTANTS["R_via"],
        C_decap=1e-10,
        freq=FIXED_CONSTANTS["freq"],
    )
    return g


def build_diff_data_edge(
    n_top: int,
    w_top: torch.Tensor,   # [n_top_phys]
    w_bot: torch.Tensor,   # [n_bot_phys]
    C_decap: float,
    proto=None,
) -> Batch:
    """Build a single-graph batch whose strap R is a per-edge tensor."""
    g = proto if proto is not None else _proto(n_top)
    data = to_hetero_data(g)

    n_load = data["mesh_bot", "load", "mesh_bot"].edge_index.size(1)

    def _strap_attr(R_dir: torch.Tensor) -> torch.Tensor:
        a = torch.zeros(R_dir.size(0), EDGE_ATTR_DIM)
        a[:, R_COL] = R_dir
        return a

    # R per physical edge; edge_index is bidir-packed [u→v ; v→u], so the
    # directed R column is the physical R tiled twice.
    R_top_phys = FIXED_CONSTANTS["Rsheet_top"] * g.pitch_top / w_top
    R_bot_phys = FIXED_CONSTANTS["Rsheet_bot"] * g.pitch_bot / w_bot
    R_top_dir = torch.cat([R_top_phys, R_top_phys])
    R_bot_dir = torch.cat([R_bot_phys, R_bot_phys])
    R_via = torch.full((data["mesh_top", "via", "mesh_bot"].edge_index.size(1),),
                       float(FIXED_CONSTANTS["R_via"]))

    data["mesh_top", "strap", "mesh_top"].edge_attr = _strap_attr(R_top_dir)
    data["mesh_bot", "strap", "mesh_bot"].edge_attr = _strap_attr(R_bot_dir)
    data["mesh_top", "via", "mesh_bot"].edge_attr = _strap_attr(R_via)
    data["mesh_bot", "via", "mesh_top"].edge_attr = _strap_attr(R_via.clone())

    n_decap = data["mesh_bot", "decap", "mesh_bot"].edge_index.size(1)
    dec = torch.zeros(n_decap, EDGE_ATTR_DIM)
    dec[:, C_COL] = C_decap
    data["mesh_bot", "decap", "mesh_bot"].edge_attr = dec

    ld = torch.zeros(n_load, EDGE_ATTR_DIM)
    ld[:, I_COL] = FIXED_I_PEAK
    ld[:, F_COL] = FIXED_FREQ
    ld[:, D_COL] = FIXED_DUTY
    ld[:, P_COL] = FIXED_PHASE
    data["mesh_bot", "load", "mesh_bot"].edge_attr = ld

    data["y"] = torch.zeros(n_load, dtype=torch.float32)
    return Batch.from_data_list([data])


def predict_loads(model, n_top, w_top, w_bot, C_decap, proto=None) -> torch.Tensor:
    batch = build_diff_data_edge(n_top, w_top, w_bot, C_decap, proto=proto)
    return 10.0 ** model(batch)   # [n_loads] in volts


# -----------------------------------------------------------------------------
# Cost = total copper volume ∝ Σ width·pitch
# -----------------------------------------------------------------------------

def metal(w_top, w_bot, pitch_top, pitch_bot) -> torch.Tensor:
    return (w_top * pitch_top).sum() + (w_bot * pitch_bot).sum()


def metal_uniform(width, n_top_phys, n_bot_phys, pitch_top, pitch_bot) -> float:
    return width * pitch_top * n_top_phys + width * pitch_bot * n_bot_phys


# -----------------------------------------------------------------------------
# Uniform-width baseline: smallest uniform width hitting the target (1-D search)
# -----------------------------------------------------------------------------

def uniform_width_for_target(model, n_top, target_v, C_decap, proto):
    n_top_phys = proto.top_edges.shape[0]
    n_bot_phys = proto.bot_edges.shape[0]
    lo, hi = WW.lo, WW.hi
    # worst droop is monotone-decreasing in width (more copper → less droop)
    def worst(width):
        wt = torch.full((n_top_phys,), float(width))
        wb = torch.full((n_bot_phys,), float(width))
        with torch.no_grad():
            return float(predict_loads(model, n_top, wt, wb, C_decap, proto).max())
    if worst(hi) > target_v:
        return None, worst(hi)          # infeasible even at max width
    for _ in range(40):                 # bisection on width
        mid = 0.5 * (lo + hi)
        if worst(mid) <= target_v:
            hi = mid
        else:
            lo = mid
    return hi, worst(hi)


# -----------------------------------------------------------------------------
# Per-edge optimization
# -----------------------------------------------------------------------------

def optimize(model, n_top, target_v, C_decap, init_width,
             lambda_cost, n_steps, lr):
    proto = _proto(n_top)
    n_top_phys = proto.top_edges.shape[0]
    n_bot_phys = proto.bot_edges.shape[0]
    pt, pb = proto.pitch_top, proto.pitch_bot

    z0 = z_from_width(init_width)
    z_top = torch.full((n_top_phys,), z0, requires_grad=True)
    z_bot = torch.full((n_bot_phys,), z0, requires_grad=True)
    opt = torch.optim.Adam([z_top, z_bot], lr=lr)

    for step in range(n_steps):
        w_top = widths_from_z(z_top)
        w_bot = widths_from_z(z_bot)
        loads = predict_loads(model, n_top, w_top, w_bot, C_decap, proto)
        worst = loads.max()
        spec = torch.relu(worst / target_v - 1.0) ** 2
        m = metal(w_top, w_bot, pt, pb)
        # normalize cost by the all-thin metal so λ is dimensionless-ish
        m0 = metal_uniform(WW.lo, n_top_phys, n_bot_phys, pt, pb)
        loss = spec + lambda_cost * (m / m0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        w_top = widths_from_z(z_top)
        w_bot = widths_from_z(z_bot)
        loads = predict_loads(model, n_top, w_top, w_bot, C_decap, proto)
    return {
        "proto": proto,
        "w_top": w_top.detach(),
        "w_bot": w_bot.detach(),
        "loads": loads.detach(),
        "worst": float(loads.max()),
        "metal": float(metal(w_top, w_bot, pt, pb)),
        "n_top_phys": n_top_phys,
        "n_bot_phys": n_bot_phys,
        "pitch_top": pt,
        "pitch_bot": pb,
    }


def plot(out, n_top, target_v, init_width, hotspot_k, png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import cm, colors

    g = out["proto"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    norm = colors.Normalize(vmin=WW.lo, vmax=WW.hi)
    cmap = cm.viridis

    for k, (pos, edges, w, title) in enumerate([
        (g.bot_pos, g.bot_edges, out["w_bot"].numpy(), "M_bot straps"),
        (g.top_pos, g.top_edges, out["w_top"].numpy(), "M_top straps"),
    ]):
        segs = [[pos[u], pos[v]] for u, v in edges]
        lc = LineCollection(segs, cmap=cmap, norm=norm,
                            linewidths=2 + 6 * norm(w))
        lc.set_array(w)
        ax[k].add_collection(lc)
        ax[k].scatter(pos[:, 0], pos[:, 1], s=8, c="0.7", zorder=1)
        ax[k].set_title(f"{title} — width (thicker/brighter = more copper)")
        ax[k].set_aspect("equal")
        ax[k].autoscale()
        fig.colorbar(lc, ax=ax[k], label="wire_width", shrink=0.8)

    # mark the original (pre-optimization) hotspot we set out to fix
    hu, hv = g.load_pairs[hotspot_k]
    hp = 0.5 * (g.bot_pos[hu] + g.bot_pos[hv])
    ax[0].scatter([hp[0]], [hp[1]], s=260, marker="*", c="red",
                  edgecolors="k", zorder=5, label="target hotspot")
    ax[0].legend(loc="upper right")

    fig.suptitle(
        f"Per-edge strap optimization  (n_top={n_top}, init width={init_width}, "
        f"target={target_v*1e3:.3f} mV)\n"
        f"final worst droop = {out['worst']*1e3:.3f} mV   "
        f"metal = {out['metal']:.2f}  [SURROGATE-ONLY / OOD]",
        fontsize=11,
    )
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)
    print(f"  figure → {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/droop_v5_nocoord.pt"))
    ap.add_argument("--n-top", type=int, default=4)
    ap.add_argument("--wire-width-init", type=float, default=0.4)
    ap.add_argument("--C-decap", type=float, default=2e-10)
    ap.add_argument("--target-mV", type=float, default=None,
                    help="default: 0.85 × baseline worst droop (forces a real fix)")
    ap.add_argument("--lambda-cost", type=float, default=1e-2)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--png", type=Path, default=Path("docs/figures/fig_edge_design.png"))
    args = ap.parse_args()

    model = PDNDroopRegressor(
        EncoderConfig(hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                      conv_type="admittance", drop_edge_p=0.0),
        target_space="log",
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    proto = _proto(args.n_top)
    n_top_phys = proto.top_edges.shape[0]
    n_bot_phys = proto.bot_edges.shape[0]

    # ---- baseline: uniform init width ----
    wt0 = torch.full((n_top_phys,), args.wire_width_init)
    wb0 = torch.full((n_bot_phys,), args.wire_width_init)
    with torch.no_grad():
        base_loads = predict_loads(model, args.n_top, wt0, wb0, args.C_decap, proto)
    base_worst = float(base_loads.max())
    worst_k = int(torch.argmax(base_loads))

    target_v = args.target_mV * 1e-3 if args.target_mV is not None else 0.85 * base_worst

    print(f"=== Localized (per-edge) strap optimization — SURROGATE ONLY (OOD) ===")
    print(f"ckpt={args.ckpt.name}  n_top={args.n_top}  init width={args.wire_width_init}  "
          f"C_decap={args.C_decap:.1e}")
    print(f"strap edges: {n_top_phys} top + {n_bot_phys} bot = {n_top_phys+n_bot_phys}")
    print(f"baseline (uniform {args.wire_width_init}): worst droop = {base_worst*1e3:.4f} mV "
          f"at load #{worst_k}")
    print(f"target: {target_v*1e3:.4f} mV")
    print()

    # ---- uniform-width solution for the same target ----
    uw, uw_worst = uniform_width_for_target(model, args.n_top, target_v, args.C_decap, proto)
    if uw is None:
        print(f"[uniform] infeasible: even width={WW.hi} gives {uw_worst*1e3:.4f} mV")
        uniform_metal = float("inf")
    else:
        uniform_metal = metal_uniform(uw, n_top_phys, n_bot_phys,
                                       proto.pitch_top, proto.pitch_bot)
        print(f"[uniform baseline] width {uw:.4f} → worst {uw_worst*1e3:.4f} mV, "
              f"metal = {uniform_metal:.2f}")

    # ---- per-edge optimization ----
    out = optimize(model, args.n_top, target_v, args.C_decap, args.wire_width_init,
                   args.lambda_cost, args.n_steps, args.lr)
    # physical lower bound: all edges at max width is the lowest-droop design
    # any per-edge assignment can reach. Beating it is physically impossible.
    wt_max = torch.full((n_top_phys,), WW.hi)
    wb_max = torch.full((n_bot_phys,), WW.hi)
    with torch.no_grad():
        floor = float(predict_loads(model, args.n_top, wt_max, wb_max, args.C_decap, proto).max())
    print(f"[physical floor]   all edges @ {WW.hi} → worst {floor*1e3:.4f} mV "
          f"(lowest droop any per-edge design can reach)")

    print(f"[per-edge]         → worst {out['worst']*1e3:.4f} mV, metal = {out['metal']:.2f}")
    if out["worst"] < floor - 1e-9:
        print(f"  ⚠ per-edge droop is BELOW the all-max-copper floor — physically "
              f"impossible. The surrogate is hallucinating on this OOD R pattern.")
    if uw is not None and out["worst"] <= target_v * 1.02:
        save = 100 * (1 - out["metal"] / uniform_metal)
        print(f"                   metal vs uniform: {save:+.1f}%  "
              f"({'cheaper' if save > 0 else 'more expensive'})")
    print()

    # ---- localization: how concentrated is the added copper? ----
    w_all = torch.cat([out["w_top"], out["w_bot"]])
    added = (w_all - WW.lo)
    frac_edges_used = float((added > 0.05 * (WW.hi - WW.lo)).float().mean())
    print(f"localization: {frac_edges_used*100:.0f}% of strap edges thickened "
          f">5% above the floor")
    # top-5 widest bot edges + distance to the hotspot
    hu, hv = proto.load_pairs[worst_k]
    hot_xy = 0.5 * (proto.bot_pos[hu] + proto.bot_pos[hv])
    wb = out["w_bot"].numpy()
    order = np.argsort(-wb)[:5]
    print(f"hotspot load #{worst_k} at bot-xy ({hot_xy[0]:.1f},{hot_xy[1]:.1f}); "
          f"widest bot straps:")
    for e in order:
        u, v = proto.bot_edges[e]
        mid = 0.5 * (proto.bot_pos[u] + proto.bot_pos[v])
        d = float(np.hypot(*(mid - hot_xy)))
        print(f"    edge {u:>2}-{v:<2}  width={wb[e]:.3f}  dist_to_hotspot={d:.2f}")

    plot(out, args.n_top, target_v, args.wire_width_init, worst_k, args.png)


if __name__ == "__main__":
    main()
