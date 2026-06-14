"""Per-chip spatial droop map: predicted vs actual at every load site.

For ONE design instance (one wire_width / C_decap / n_top), lay out all load
sources at their physical mesh locations and color them by droop — surrogate
prediction vs simulator ground truth, side by side — plus a per-load
predicted-vs-actual scatter. Coordinates are used only for plotting; the model
never sees them.

Default chip: the held-out topology n_top=4, an aggressive (thin-wire,
low-cap) design so there is real spatial droop spread to see.
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
                           FIXED_DUTY, FIXED_PHASE)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

N_TOP = 4
WIRE_WIDTH = 0.35
C_DECAP = 8e-11


def load_model():
    m = PDNDroopRegressor(
        EncoderConfig(hidden_dim=64, n_layers=7, conv_type="admittance",
                      drop_edge_p=0.0), target_space="log")
    m.load_state_dict(torch.load(ROOT / "checkpoints" / "droop_v5_nocoord.pt",
                                 map_location="cpu", weights_only=False)["model"])
    m.eval()
    return m


def main():
    m = load_model()
    g = build_regular_pdn(
        n_top=N_TOP, n_bot=FIXED_CONSTANTS["n_bot"],
        Rsheet_top=FIXED_CONSTANTS["Rsheet_top"],
        Rsheet_bot=FIXED_CONSTANTS["Rsheet_bot"],
        wire_width=WIRE_WIDTH, R_via=FIXED_CONSTANTS["R_via"], C_decap=C_DECAP,
        freq=FIXED_CONSTANTS["freq"],
        loads=np.tile([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]],
                      (build_regular_pdn(n_top=N_TOP).n_loads, 1)))

    # ----- predicted (per-load, in mesh order) -----
    d = to_hetero_data(g)
    d["y"] = torch.zeros(g.n_loads)
    with torch.no_grad():
        pred = (10.0 ** m(Batch.from_data_list([d]))).numpy() * 1e3  # mV, len 14

    # ----- actual (simulator, per-load) -----
    p = dict(FIXED_CONSTANTS)
    p.update(wire_width=WIRE_WIDTH, C_decap=C_DECAP, n_top=N_TOP, loads=g.loads)
    actual = run_one(p, cfg=SimConfig())["peak_droop_loads"] * 1e3  # mV, len 14

    # ----- load-site locations (midpoint of the Vdd/Vss node pair) -----
    bp = g.bot_pos
    loc = np.array([(bp[u] + bp[v]) / 2 for u, v in g.load_pairs])  # [14, 2]
    lx, ly = loc[:, 0], loc[:, 1]

    vmin = min(pred.min(), actual.min())
    vmax = max(pred.max(), actual.max())

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(18, 6.2), gridspec_kw={"width_ratios": [1, 1, 0.95]},
        constrained_layout=True)

    def spatial(ax, vals, title):
        # faint bot-mesh nodes for context
        ax.scatter(bp[:, 0], bp[:, 1], s=8, color="lightgray", zorder=1)
        sc = ax.scatter(lx, ly, c=vals, s=900, cmap="inferno_r",
                        vmin=vmin, vmax=vmax, edgecolor="k", linewidth=0.8,
                        zorder=3)
        for x, y, v in zip(lx, ly, vals):
            ax.text(x, y, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white", zorder=4)
        ax.set_title(title)
        ax.set_xlabel("x (mesh units)")
        ax.set_ylabel("y (mesh units)")
        ax.set_aspect("equal")
        return sc

    spatial(ax0, pred, "PREDICTED droop per load (mV)")
    sc = spatial(ax1, actual, "ACTUAL droop per load (mV)")
    fig.colorbar(sc, ax=(ax0, ax1), fraction=0.04, pad=0.02, label="droop (mV)")

    ax2.scatter(actual, pred, s=90, color="#c0392b", edgecolor="k", zorder=3)
    lo, hi = vmin * 0.97, vmax * 1.03
    ax2.plot([lo, hi], [lo, hi], "k--", label="pred = actual")
    for i, (a, pv) in enumerate(zip(actual, pred)):
        ax2.annotate(str(i), (a, pv), fontsize=7, xytext=(3, 3),
                     textcoords="offset points")
    ss_res = float(np.sum((pred - actual) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    mae = float(np.mean(np.abs(pred - actual)))
    ax2.set_title(f"Per-load pred vs actual\nR² = {r2:.3f}   MAE = {mae:.4f} mV")
    ax2.set_xlabel("actual droop (mV)")
    ax2.set_ylabel("predicted droop (mV)")
    ax2.legend(loc="upper left")

    fig.suptitle(
        f"Single chip — n_top={N_TOP} (OOD), wire_width={WIRE_WIDTH}, "
        f"C_decap={C_DECAP:.0e} F: droop distribution over the 14 load sites",
        fontsize=13)
    fig.savefig(FIG_DIR / "fig_single_chip_droop.png", dpi=130)
    plt.close(fig)
    print(f"worst-load: pred {pred.max():.4f} / actual {actual.max():.4f} mV  "
          f"(per-load R²={r2:.3f}, MAE={mae:.4f} mV)")
    print(f"wrote {FIG_DIR / 'fig_single_chip_droop.png'}")


if __name__ == "__main__":
    main()
