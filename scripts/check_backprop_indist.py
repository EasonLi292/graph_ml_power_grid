"""How well does the gradient inverse design work in-distribution (n_top 3, 7)?

For a sweep of droop targets on each in-distribution topology, run the
backprop design loop and check a 3-way agreement at the recovered design:

    target budget  ≈  surrogate prediction  ≈  simulator ground truth

If backprop + the (in-dist, near-exact) surrogate are working, all three lie on
y = x: the optimizer lands a design whose *real* droop equals what was asked.
Also reports recovered knobs and convergence quality. Figure +
docs/analysis/backprop_indist.json.
"""
from __future__ import annotations

import json
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
from scripts.design_grad import design_loss, to_physical

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "docs" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

NTOPS = [3, 7]
N_BUDGETS = 9
N_STEPS = 300
LAMBDA = 1e-3


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


def _loads(nt):
    return np.tile([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]],
                   (build_regular_pdn(n_top=nt).n_loads, 1))


@torch.no_grad()
def surrogate(model, nt, ww, cd):
    g = build_regular_pdn(
        n_top=nt, n_bot=FIXED_CONSTANTS["n_bot"],
        Rsheet_top=FIXED_CONSTANTS["Rsheet_top"],
        Rsheet_bot=FIXED_CONSTANTS["Rsheet_bot"],
        wire_width=ww, R_via=FIXED_CONSTANTS["R_via"], C_decap=cd,
        freq=FIXED_CONSTANTS["freq"], loads=_loads(nt))
    d = to_hetero_data(g)
    d["y"] = torch.zeros(g.n_loads)
    return float(10 ** model(Batch.from_data_list([d])).max()) * 1e3


def simulator(nt, ww, cd):
    p = dict(FIXED_CONSTANTS)
    p.update(wire_width=ww, C_decap=cd, n_top=nt, loads=_loads(nt))
    return float(run_one(p, cfg=SimConfig())["peak_droop_loads"].max()) * 1e3


def optimize(model, nt, budget_mV, seed=0):
    target_v = budget_mV * 1e-3
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(2, requires_grad=True)
    z.data = torch.randn(2, generator=g) * 0.5
    opt = torch.optim.Adam([z], lr=0.1)
    last = None
    for _ in range(N_STEPS):
        loss, info = design_loss(model, z, nt, target_v, LAMBDA)
        opt.zero_grad(); loss.backward(); opt.step()
        last = info
    return last


def main():
    m = load_model()
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, nt in enumerate(NTOPS):
        # achievable droop range (max wire+cap = lowest droop; min = highest)
        lo = surrogate(m, nt, 1.0, 8e-10)
        hi = surrogate(m, nt, 0.2, 5e-11)
        budgets = np.linspace(lo * 1.05, hi * 0.92, N_BUDGETS)

        rows = []
        for b in budgets:
            info = optimize(m, nt, float(b))
            ww, cd = info["wire_width"], info["C_decap"]
            pred = info["worst_pred"] * 1e3
            sim = simulator(nt, ww, cd)
            rows.append(dict(budget=float(b), wire_width=ww, C_decap=cd,
                             pred=pred, sim=sim, loss=info["loss"]))
        results[f"n_top_{nt}"] = rows
        bud = np.array([r["budget"] for r in rows])
        pred = np.array([r["pred"] for r in rows])
        sim = np.array([r["sim"] for r in rows])
        ww = np.array([r["wire_width"] for r in rows])
        cd = np.array([r["C_decap"] for r in rows])

        # panel 1: 3-way agreement (target vs achieved)
        ax = axes[row, 0]
        lohi = [min(bud.min(), sim.min()) * 0.98, max(bud.max(), sim.max()) * 1.02]
        ax.plot(lohi, lohi, "k--", label="target = achieved (ideal)")
        ax.scatter(bud, sim, s=70, color="#2c6fbb", label="simulator droop", zorder=4)
        ax.scatter(bud, pred, s=30, color="#c0392b", marker="x",
                   label="surrogate droop", zorder=5)
        mae_sim = float(np.mean(np.abs(sim - bud)))
        ax.set_title(f"n_top = {nt} (in-dist): asked vs achieved\n"
                     f"|simulated − target| MAE = {mae_sim:.4f} mV")
        ax.set_xlabel("requested droop budget (mV)")
        ax.set_ylabel("droop at recovered design (mV)")
        ax.legend(fontsize=8)

        # panel 2: recovered knobs vs budget
        ax2 = axes[row, 1]
        ax2.plot(bud, ww, "o-", color="#c0392b", label="wire_width")
        ax2.set_xlabel("requested droop budget (mV)")
        ax2.set_ylabel("recovered wire_width", color="#c0392b")
        ax2.tick_params(axis="y", labelcolor="#c0392b")
        ax2b = ax2.twinx()
        ax2b.plot(bud, cd, "s-", color="#8e44ad", label="C_decap")
        ax2b.set_ylabel("recovered C_decap (F)", color="#8e44ad")
        ax2b.set_yscale("log")
        ax2b.tick_params(axis="y", labelcolor="#8e44ad")
        ax2.set_title(f"n_top = {nt}: recovered design vs budget "
                      "(looser → less metal)")

    fig.suptitle("In-distribution inverse design (n_top 3 & 7): does backprop "
                 "land the requested droop?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig_backprop_indist.png", dpi=130)
    plt.close(fig)

    (DATA_DIR / "backprop_indist.json").write_text(json.dumps(results, indent=2))

    # console summary
    for nt in NTOPS:
        rows = results[f"n_top_{nt}"]
        sim = np.array([r["sim"] for r in rows])
        bud = np.array([r["budget"] for r in rows])
        pred = np.array([r["pred"] for r in rows])
        print(f"\nn_top={nt}:  |sim−target| MAE={np.mean(np.abs(sim-bud)):.4f} mV   "
              f"|pred−sim| MAE={np.mean(np.abs(pred-sim)):.4f} mV")
        print(f"  {'budget':>8} {'pred':>8} {'sim':>8} {'ww':>7} {'C_decap':>9}")
        for r in rows:
            print(f"  {r['budget']:>8.4f} {r['pred']:>8.4f} {r['sim']:>8.4f} "
                  f"{r['wire_width']:>7.3f} {r['C_decap']:>9.2e}")
    print(f"\nfig -> {FIG_DIR/'fig_backprop_indist.png'}")


if __name__ == "__main__":
    main()
