"""Inferred (surrogate) vs actual (simulator) droop.

Two views, both for the coordinate-free 7-hop surrogate:
  1. wire-width sweeps at fixed (near-ceiling) decap — exposes the surrogate's
     non-monotonicity on the held-out topology n_top=4 vs the clean monotonic
     in-distribution n_top=3, against the simulator ground truth.
  2. a direct inferred-vs-actual scatter over all swept points, with y=x.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eason import EncoderConfig, PDNDroopRegressor
from tools.grid_construction import build_regular_pdn, to_hetero_data
from tools.dataset_runner import SimConfig, run_one
from tools.sampler import (FIXED_CONSTANTS, FIXED_I_PEAK, FIXED_FREQ,
                           FIXED_DUTY, FIXED_PHASE)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
CD = 7.7e-10  # near the decap ceiling, matching the 0.10 mV design point
SPEC = 0.10   # mV


def load_model():
    m = PDNDroopRegressor(
        EncoderConfig(hidden_dim=64, n_layers=7, conv_type="admittance",
                      drop_edge_p=0.0), target_space="log")
    m.load_state_dict(torch.load(ROOT / "checkpoints" / "droop_v5_nocoord.pt",
                                 map_location="cpu", weights_only=False)["model"])
    m.eval()
    return m


def _graph(nt, ww, cd):
    loads = np.tile([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]],
                    (build_regular_pdn(n_top=nt).n_loads, 1))
    return build_regular_pdn(
        n_top=nt, n_bot=FIXED_CONSTANTS["n_bot"],
        Rsheet_top=FIXED_CONSTANTS["Rsheet_top"],
        Rsheet_bot=FIXED_CONSTANTS["Rsheet_bot"],
        wire_width=ww, R_via=FIXED_CONSTANTS["R_via"], C_decap=cd,
        freq=FIXED_CONSTANTS["freq"], loads=loads)


@torch.no_grad()
def surrogate(m, nt, ww, cd):
    g = _graph(nt, ww, cd)
    d = to_hetero_data(g)
    d["y"] = torch.zeros(g.n_loads)
    return float(10 ** m(Batch.from_data_list([d])).max()) * 1e3


def simulator(nt, ww, cd):
    loads = np.tile([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]],
                    (build_regular_pdn(n_top=nt).n_loads, 1))
    p = dict(FIXED_CONSTANTS)
    p.update(wire_width=ww, C_decap=cd, n_top=nt, loads=loads)
    return float(run_one(p, cfg=SimConfig())["peak_droop_loads"].max()) * 1e3


def main():
    m = load_model()
    wws = np.linspace(0.2, 1.0, 17)
    data = {}
    for nt in (3, 4):
        sur = np.array([surrogate(m, nt, float(w), CD) for w in wws])
        sim = np.array([simulator(nt, float(w), CD) for w in wws])
        data[nt] = (sur, sim)

    # ---- view 1: sweeps ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    titles = {4: "n_top = 4  (HELD-OUT / OOD)", 3: "n_top = 3  (in-distribution)"}
    for ax, nt in zip(axes, (4, 3)):
        sur, sim = data[nt]
        ax.plot(wws, sur, "o-", color="#c0392b", label="surrogate (inferred)")
        ax.plot(wws, sim, "s-", color="#2c6fbb", label="simulator (actual)")
        ax.axhline(SPEC, color="green", ls=":", lw=1.5, label=f"spec {SPEC} mV")
        if nt == 4:
            j = int(np.argmin(sur))
            ax.scatter([wws[j]], [sur[j]], s=200, facecolor="none",
                       edgecolor="k", lw=2, zorder=5)
            ax.annotate("optimizer stops here\n(surrogate's minimum)",
                        (wws[j], sur[j]), (0.45, 0.17), fontsize=9,
                        arrowprops=dict(arrowstyle="->"))
            ax.annotate("simulator: spec IS met\nat full wire",
                        (wws[-1], sim[-1]), (0.62, 0.085), fontsize=9,
                        color="#2c6fbb",
                        arrowprops=dict(arrowstyle="->", color="#2c6fbb"))
        ax.set_title(titles[nt])
        ax.set_xlabel("wire_width (copper)")
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("worst-load droop (mV)")
    fig.suptitle("Inferred vs actual droop vs wire width "
                 f"(decap fixed at {CD:.1e} F)\n"
                 "OOD: surrogate is non-monotonic and over-pessimistic; "
                 "in-dist: surrogate tracks the simulator", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_surrogate_vs_sim_sweep.png", dpi=130)
    plt.close(fig)

    # ---- view 2: inferred vs actual scatter ----
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for nt, c, mk in ((3, "#2c6fbb", "s"), (4, "#c0392b", "o")):
        sur, sim = data[nt]
        ax.scatter(sim, sur, s=55, color=c, marker=mk,
                   label=f"n_top={nt}" + (" (OOD)" if nt == 4 else " (in-dist)"))
    lo, hi = 0.08, 0.36
    ax.plot([lo, hi], [lo, hi], "k--", label="inferred = actual")
    ax.set_xlabel("actual (simulator) worst droop (mV)")
    ax.set_ylabel("inferred (surrogate) worst droop (mV)")
    ax.set_title("Inferred vs actual droop along the wire-width sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_inferred_vs_actual.png", dpi=130)
    plt.close(fig)

    print("wrote fig_surrogate_vs_sim_sweep.png, fig_inferred_vs_actual.png")
    j = int(np.argmin(data[4][0]))
    print(f"n_top=4 surrogate min at ww={wws[j]:.3f} (pred {data[4][0][j]:.4f} mV); "
          f"sim at full wire = {data[4][1][-1]:.4f} mV (spec {SPEC})")


if __name__ == "__main__":
    main()
