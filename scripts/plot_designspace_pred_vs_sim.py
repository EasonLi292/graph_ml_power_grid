"""Predicted vs actual droop across the *full 2-D design space*.

Sweeps both knobs — wire_width and C_decap — and compares the surrogate's
predicted worst-load droop to the simulator's, as paired heatmaps plus a
signed-error map, for the held-out topology n_top=4 (OOD) and the
in-distribution n_top=3. Also emits 1-D capacitance sweeps to complement the
existing wire-width sweep.
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
from tools.dataset_runner import SimConfig, run_one
from tools.sampler import (FIXED_CONSTANTS, FIXED_I_PEAK, FIXED_FREQ,
                           FIXED_DUTY, FIXED_PHASE, GLOBAL_RANGES)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
NWW, NCD = 16, 16


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


def grids(m, nt, wws, cds):
    pred = np.zeros((NCD, NWW))
    sim = np.zeros((NCD, NWW))
    for i, cd in enumerate(cds):
        for j, ww in enumerate(wws):
            pred[i, j] = surrogate(m, nt, float(ww), float(cd))
            sim[i, j] = simulator(nt, float(ww), float(cd))
    return pred, sim


def fig_2d(m, wws, cds):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ext = [wws[0], wws[-1], 0, NCD]
    cd_ticks = np.linspace(0.5, NCD - 0.5, 5)
    cd_lab = [f"{v:.1e}" for v in np.geomspace(cds[0], cds[-1], 5)]
    for row, nt in enumerate((4, 3)):
        pred, sim = grids(m, nt, wws, cds)
        vmax = max(pred.max(), sim.max())
        vmin = min(pred.min(), sim.min())
        err = pred - sim
        emax = np.abs(err).max()
        tag = "n_top = 4  (OOD)" if nt == 4 else "n_top = 3  (in-dist)"
        for col, (data, title, cmap, vlo, vhi) in enumerate([
            (pred, f"{tag}\nPREDICTED droop (mV)", "magma", vmin, vmax),
            (sim, f"{tag}\nACTUAL droop (mV)", "magma", vmin, vmax),
            (err, f"{tag}\nPREDICTED − ACTUAL (mV)", "RdBu_r", -emax, emax),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data, origin="lower", aspect="auto", extent=ext,
                           cmap=cmap, vmin=vlo, vmax=vhi)
            cs = ax.contour(data, levels=6, colors="k", linewidths=0.5,
                            extent=ext, origin="lower")
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("wire_width")
            ax.set_yticks(cd_ticks)
            ax.set_yticklabels(cd_lab)
            ax.set_ylabel("C_decap (F)")
    fig.suptitle("Predicted vs actual worst-load droop across the 2-D design "
                 "space\nIn-dist (bottom) matches; OOD (top) is over-pessimistic "
                 "and distorted at the fat-wire edge", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_designspace_pred_vs_sim.png", dpi=120)
    plt.close(fig)


def fig_cap_sweep(m, cds):
    ww_fixed = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, nt in zip(axes, (4, 3)):
        sur = np.array([surrogate(m, nt, ww_fixed, float(c)) for c in cds])
        sim = np.array([simulator(nt, ww_fixed, float(c)) for c in cds])
        ax.plot(cds, sur, "o-", color="#c0392b", label="surrogate (inferred)")
        ax.plot(cds, sim, "s-", color="#2c6fbb", label="simulator (actual)")
        ax.set_xscale("log")
        ax.set_title(("n_top = 4  (OOD)" if nt == 4 else
                      "n_top = 3  (in-dist)"))
        ax.set_xlabel("C_decap (F)")
        ax.legend()
    axes[0].set_ylabel("worst-load droop (mV)")
    fig.suptitle(f"Inferred vs actual droop vs decap (wire_width = {ww_fixed})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cap_sweep_pred_vs_sim.png", dpi=130)
    plt.close(fig)


def main():
    m = load_model()
    ww_p = GLOBAL_RANGES.by_name("wire_width")
    cd_p = GLOBAL_RANGES.by_name("C_decap")
    wws = np.linspace(ww_p.lo, ww_p.hi, NWW)
    cds = np.geomspace(cd_p.lo, cd_p.hi, NCD)
    fig_2d(m, wws, cds)
    fig_cap_sweep(m, cds)
    print("wrote fig_designspace_pred_vs_sim.png, fig_cap_sweep_pred_vs_sim.png")


if __name__ == "__main__":
    main()
