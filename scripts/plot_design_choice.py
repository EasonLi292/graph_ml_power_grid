"""Figures about the *design the optimizer chooses* (not droop accuracy).

Three views, all on the coordinate-free 7-hop surrogate:

  1. fig_design_trajectory.png — the gradient-descent path through the
     (wire_width, C_decap) design space, over a contour map of predicted droop,
     with the spec boundary drawn. Shows the design being chosen and where it
     lands (cheapest feasible point — or, for an infeasible budget, the
     surrogate's stuck minimum).
  2. fig_design_convergence.png — wire_width, C_decap, predicted droop, and the
     loss vs optimization step for one (budget, topology): the choosing process.
  3. fig_design_atlas.png — every chosen design (3 budgets × 3 topologies) as a
     point in design space, over the droop landscape: how the pick migrates with
     the spec and the topology.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eason import EncoderConfig, PDNDroopRegressor
from tools.grid_construction import build_regular_pdn, to_hetero_data
from tools.sampler import (FIXED_CONSTANTS, FIXED_I_PEAK, FIXED_FREQ,
                           FIXED_DUTY, FIXED_PHASE, GLOBAL_RANGES)
from scripts.design_grad import design_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [0.10, 0.15, 0.20]      # mV
NTOPS = [3, 4, 7]
COLOR = {3: "#c0392b", 4: "#8e44ad", 7: "#2c6fbb"}
MARK = {3: "o", 4: "D", 7: "s"}
NG = 36                            # contour grid resolution
LAMBDA = 1e-3
N_STEPS = 200


def load_model():
    m = PDNDroopRegressor(
        EncoderConfig(hidden_dim=64, n_layers=7, conv_type="admittance",
                      drop_edge_p=0.0), target_space="log")
    m.load_state_dict(torch.load(ROOT / "checkpoints" / "droop_v5_nocoord.pt",
                                 map_location="cpu", weights_only=False)["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def sdroop(model, nt, ww, cd):
    loads = np.tile([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]],
                    (build_regular_pdn(n_top=nt).n_loads, 1))
    g = build_regular_pdn(
        n_top=nt, n_bot=FIXED_CONSTANTS["n_bot"],
        Rsheet_top=FIXED_CONSTANTS["Rsheet_top"],
        Rsheet_bot=FIXED_CONSTANTS["Rsheet_bot"],
        wire_width=ww, R_via=FIXED_CONSTANTS["R_via"], C_decap=cd,
        freq=FIXED_CONSTANTS["freq"], loads=loads)
    d = to_hetero_data(g)
    d["y"] = torch.zeros(g.n_loads)
    return float(10 ** model(Batch.from_data_list([d])).max()) * 1e3


def droop_grid(model, nt, wws, cds):
    Z = np.zeros((len(cds), len(wws)))
    for i, cd in enumerate(cds):
        for j, ww in enumerate(wws):
            Z[i, j] = sdroop(model, nt, float(ww), float(cd))
    return Z


def trajectory(model, nt, budget_mV, seed=0):
    """Adam descent, recording the visited (ww, cd, droop, loss) each step."""
    target_v = budget_mV * 1e-3
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(2, requires_grad=True)
    z.data = torch.randn(2, generator=g) * 0.5
    opt = torch.optim.Adam([z], lr=0.1)
    ww, cd, drp, loss_hist = [], [], [], []
    for _ in range(N_STEPS):
        loss, info = design_loss(model, z, nt, target_v, LAMBDA)
        opt.zero_grad(); loss.backward(); opt.step()
        ww.append(info["wire_width"]); cd.append(info["C_decap"])
        drp.append(info["worst_pred"] * 1e3); loss_hist.append(info["loss"])
    return (np.array(ww), np.array(cd), np.array(drp), np.array(loss_hist))


def fig_trajectory(model, wws, cds):
    nt = 4
    Z = droop_grid(model, nt, wws, cds)
    WW, CD = np.meshgrid(wws, cds)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), sharey=True,
                             constrained_layout=True)
    for ax, bud in zip(axes, BUDGETS):
        cf = ax.contourf(WW, CD, Z, levels=20, cmap="magma_r", alpha=0.9)
        # spec boundary: predicted droop == budget
        cs = ax.contour(WW, CD, Z, levels=[bud], colors="cyan", linewidths=2.2)
        ax.clabel(cs, fmt=lambda v: f"{v:.2f} mV spec", fontsize=8)
        ww, cd, drp, _ = trajectory(model, nt, bud)
        ax.plot(ww, cd, "-", color="white", lw=1.2, alpha=0.8)
        ax.scatter(ww[0], cd[0], s=90, color="white", edgecolor="k",
                   zorder=5, label="start")
        ax.scatter(ww[-1], cd[-1], s=240, marker="*", color="lime",
                   edgecolor="k", zorder=6, label="chosen design")
        feasible = drp[-1] <= bud + 1e-6
        ax.set_title(f"budget {bud:.2f} mV  →  ww={ww[-1]:.3f}, C={cd[-1]:.1e}\n"
                     f"{'lands on spec boundary' if feasible else 'INFEASIBLE (stuck at surrogate min)'}",
                     fontsize=9)
        ax.set_xlabel("wire_width")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("C_decap (F)")
    fig.colorbar(cf, ax=axes, fraction=0.025, pad=0.02,
                 label="predicted worst droop (mV)")
    fig.suptitle("What design the optimizer chooses — gradient descent over the "
                 "droop landscape (n_top = 4, OOD).  "
                 "cyan = spec boundary,  ★ = chosen design", fontsize=12)
    fig.savefig(FIG_DIR / "fig_design_trajectory.png", dpi=125)
    plt.close(fig)


def fig_convergence(model):
    nt, bud = 4, 0.15
    ww, cd, drp, loss = trajectory(model, nt, bud)
    steps = np.arange(len(ww))
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(steps, ww, color="#c0392b"); axes[0].set_ylabel("wire_width")
    axes[0].set_title(f"Convergence of the chosen design (n_top={nt}, "
                      f"budget={bud} mV)")
    axes[1].plot(steps, cd, color="#8e44ad"); axes[1].set_ylabel("C_decap (F)")
    axes[1].set_yscale("log")
    axes[2].plot(steps, drp, color="#2c6fbb")
    axes[2].axhline(bud, color="k", ls="--", lw=1, label=f"budget {bud} mV")
    axes[2].set_ylabel("pred droop (mV)"); axes[2].legend(fontsize=8)
    axes[3].plot(steps, loss, color="#16a085"); axes[3].set_ylabel("loss")
    axes[3].set_yscale("log"); axes[3].set_xlabel("optimization step")
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_design_convergence.png", dpi=130)
    plt.close(fig)


def fig_atlas(model, wws, cds):
    # backdrop: n_top=4 droop landscape
    Z = droop_grid(model, 4, wws, cds)
    WW, CD = np.meshgrid(wws, cds)
    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(WW, CD, Z, levels=20, cmap="Greys", alpha=0.55)
    fig.colorbar(cf, ax=ax, label="predicted droop, n_top=4 backdrop (mV)")
    for nt in NTOPS:
        xs, ys = [], []
        for bud in BUDGETS:
            ww, cd, drp, _ = trajectory(model, nt, bud)
            xs.append(ww[-1]); ys.append(cd[-1])
        ax.plot(xs, ys, "-", color=COLOR[nt], alpha=0.5)
        ax.scatter(xs, ys, marker=MARK[nt], s=150, color=COLOR[nt],
                   edgecolor="k", zorder=5,
                   label=f"n_top={nt}" + (" (OOD)" if nt == 4 else ""))
        for bud, x, y in zip(BUDGETS, xs, ys):
            ax.annotate(f"{bud:.2f}", (x, y), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
    ax.set_xlabel("chosen wire_width"); ax.set_ylabel("chosen C_decap (F)")
    ax.set_yscale("log")
    ax.set_title("Design atlas — the design chosen for every (budget, topology)\n"
                 "labels = droop budget in mV; looser budget → less copper, "
                 "fewer pads → more copper")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_design_atlas.png", dpi=130)
    plt.close(fig)


def main():
    m = load_model()
    ww_p = GLOBAL_RANGES.by_name("wire_width")
    cd_p = GLOBAL_RANGES.by_name("C_decap")
    wws = np.linspace(ww_p.lo, ww_p.hi, NG)
    cds = np.geomspace(cd_p.lo, cd_p.hi, NG)
    fig_trajectory(m, wws, cds)
    fig_convergence(m)
    fig_atlas(m, wws, cds)
    print("wrote fig_design_trajectory.png, fig_design_convergence.png, "
          "fig_design_atlas.png")


if __name__ == "__main__":
    main()
