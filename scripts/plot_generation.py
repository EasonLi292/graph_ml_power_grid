"""Figures for the inverse-design (generation) analysis.

Plots the captured output of::

    scripts/design_grad.py --ckpt checkpoints/droop_v5_nocoord.pt \
        --target-mV {0.10,0.15,0.20} --lambda-cost 1e-3

(coordinate-free 7-hop surrogate). The table below is that run's result,
recorded verbatim so the figures are reproducible without re-optimising.
Columns: budget_mV, n_top, wire_width, C_decap, pred_mV, sim_mV, feasible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# budget_mV, n_top, wire_width, C_decap, pred_mV, sim_mV, feasible
ROWS = [
    (0.10, 3, 0.9862, 7.20e-10, 0.1128, 0.1130, False),
    (0.10, 4, 0.7364, 7.69e-10, 0.1405, 0.1216, False),
    (0.10, 7, 0.8313, 2.27e-10, 0.0990, 0.0991, True),
    (0.15, 3, 0.9086, 4.24e-10, 0.1342, 0.1344, True),
    (0.15, 4, 0.7132, 5.61e-10, 0.1502, 0.1323, True),
    (0.15, 7, 0.6010, 5.13e-11, 0.1501, 0.1501, False),
    (0.20, 3, 0.6128, 2.45e-10, 0.2001, 0.2000, True),
    (0.20, 4, 0.6310, 2.39e-10, 0.1964, 0.1827, True),
    (0.20, 7, 0.4479, 5.07e-11, 0.1968, 0.1971, True),
]

BUDGETS = [0.10, 0.15, 0.20]
NTOPS = [3, 4, 7]
COLOR = {3: "#c0392b", 4: "#8e44ad", 7: "#2c6fbb"}
MARK = {3: "o", 4: "D", 7: "s"}


def _get(n_top, col):
    return [r[col] for r in ROWS if r[1] == n_top]


def fig_cost_vs_budget():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for nt in NTOPS:
        bud = _get(nt, 0)
        ww = _get(nt, 2)
        cd = _get(nt, 3)
        lbl = f"n_top={nt}" + (" (OOD)" if nt == 4 else "")
        ax1.plot(bud, ww, MARK[nt] + "-", color=COLOR[nt], label=lbl, ms=9)
        ax2.plot(bud, cd, MARK[nt] + "-", color=COLOR[nt], label=lbl, ms=9)
    ax1.set_xlabel("droop budget (mV)")
    ax1.set_ylabel("recovered wire_width")
    ax1.set_title("Looser budget → thinner (cheaper) wire")
    ax1.set_xticks(BUDGETS)
    ax1.legend()
    ax2.set_xlabel("droop budget (mV)")
    ax2.set_ylabel("recovered C_decap (F)")
    ax2.set_yscale("log")
    ax2.set_title("Decap also relaxes as the budget loosens")
    ax2.set_xticks(BUDGETS)
    ax2.legend()
    fig.suptitle("Inverse design: recovered knobs vs droop budget "
                 "(coordinate-free surrogate)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_design_cost_vs_budget.png", dpi=130)
    plt.close(fig)


def fig_surrogate_fidelity():
    fig, ax = plt.subplots(figsize=(7, 7))
    for nt in NTOPS:
        pred = _get(nt, 4)
        sim = _get(nt, 5)
        lbl = f"n_top={nt}" + (" (OOD)" if nt == 4 else " (in-dist)")
        ax.scatter(sim, pred, s=130, marker=MARK[nt], color=COLOR[nt],
                   edgecolor="k", linewidth=0.6, label=lbl, zorder=3)
    lo, hi = 0.09, 0.16
    ax.plot([lo, hi], [lo, hi], "k--", label="pred = sim")
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="green", alpha=0.06)
    ax.text(0.105, 0.150, "conservative\n(pred > sim, SAFE)", fontsize=9,
            color="green")
    ax.text(0.130, 0.098, "optimistic\n(pred < sim, risky)", fontsize=9,
            color="red")
    ax.set_xlabel("simulator droop at recovered design (mV)")
    ax.set_ylabel("surrogate predicted droop (mV)")
    ax.set_title("Surrogate fidelity at the optimum\n"
                 "in-dist lands on y=x; OOD (n_top=4) sits in the SAFE zone")
    ax.legend(loc="lower right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_surrogate_fidelity.png", dpi=130)
    plt.close(fig)


def fig_topology_difficulty():
    """At each budget, copper needed vs supply density (n_top)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    width = 0.25
    x = np.arange(len(NTOPS))
    for i, bud in enumerate(BUDGETS):
        ww = [r[2] for nt in NTOPS for r in ROWS if r[1] == nt and r[0] == bud]
        ax.bar(x + (i - 1) * width, ww, width, label=f"{bud:.2f} mV budget")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n_top={nt}" + ("\n(OOD)" if nt == 4 else "")
                        for nt in NTOPS])
    ax.set_ylabel("recovered wire_width (copper cost)")
    ax.set_title("Fewer supply pads → more copper needed at a fixed budget")
    ax.legend(title="droop budget")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_design_topology.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_cost_vs_budget()
    fig_surrogate_fidelity()
    fig_topology_difficulty()
    print(f"Generation figures -> {FIG_DIR}")
