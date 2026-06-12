"""Inverse-design demo: given a target worst-load droop spec, find the
cheapest (wire_width, C_decap, n_top) that meets it.

Uses the trained surrogate as a fast feasibility-scanner, then
validates the recommended design point against the transient simulator.

Usage:
    python scripts/design.py \\
        --ckpt checkpoints/droop_v5_conductance.pt \\
        --target-mV 0.10

The output is a table per n_top showing the Pareto frontier of feasible
designs, and a single recommendation per topology (smallest
wire_width that achieves the spec at any C_decap in range).
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
from tools.grid_construction import build_regular_pdn, to_hetero_data
from tools.pyg_dataset import LOG_FLOOR
from tools.sampler import (
    ALL_N_TOP,
    FIXED_CONSTANTS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    GLOBAL_RANGES,
)


def build_one(wire_width: float, C_decap: float, n_top: int):
    """Build one HeteroData sample with the given knobs."""
    loads = np.tile(
        np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
        (build_regular_pdn(n_top=n_top).n_loads, 1),
    )
    g = build_regular_pdn(
        n_top=n_top,
        n_bot=FIXED_CONSTANTS["n_bot"],
        Rsheet_top=FIXED_CONSTANTS["Rsheet_top"],
        Rsheet_bot=FIXED_CONSTANTS["Rsheet_bot"],
        wire_width=wire_width,
        R_via=FIXED_CONSTANTS["R_via"],
        C_decap=C_decap,
        freq=FIXED_CONSTANTS["freq"],
        loads=loads,
    )
    data = to_hetero_data(g)
    # Make the y-attribute present so PyG batching is happy.
    data["y"] = torch.zeros(g.n_loads, dtype=torch.float32)
    return data


@torch.no_grad()
def predict_worst_droop(
    model: PDNDroopRegressor,
    wire_width: float,
    C_decap: float,
    n_top: int,
) -> float:
    """Surrogate-predicted worst-load droop, in volts."""
    data = build_one(wire_width, C_decap, n_top)
    batch = Batch.from_data_list([data])
    pred = model(batch).cpu()
    return float(10.0 ** pred.max().item())


def simulate_worst_droop(wire_width: float, C_decap: float, n_top: int) -> float:
    """Ground-truth worst-load droop from the transient simulator."""
    loads = np.tile(
        np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
        (build_regular_pdn(n_top=n_top).n_loads, 1),
    )
    p = dict(FIXED_CONSTANTS)
    p.update(wire_width=wire_width, C_decap=C_decap, n_top=n_top, loads=loads)
    res = run_one(p, cfg=SimConfig())
    return float(res["peak_droop_loads"].max())


def feasibility_grid(
    model: PDNDroopRegressor,
    n_top: int,
    target_v: float,
    n_ww: int = 25,
    n_cd: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ww grid, cd grid, predicted droop[n_ww, n_cd], feasible mask)."""
    ww_p = GLOBAL_RANGES.by_name("wire_width")
    cd_p = GLOBAL_RANGES.by_name("C_decap")
    ww = np.geomspace(ww_p.lo, ww_p.hi, n_ww)
    cd = np.geomspace(cd_p.lo, cd_p.hi, n_cd)

    droop = np.zeros((n_ww, n_cd), dtype=np.float32)
    for i, w in enumerate(ww):
        for j, c in enumerate(cd):
            droop[i, j] = predict_worst_droop(model, float(w), float(c), n_top)
    feasible = droop <= target_v
    return ww, cd, droop, feasible


def recommend(
    ww: np.ndarray, cd: np.ndarray, feasible: np.ndarray
) -> tuple[float, float] | None:
    """Pick the minimum-wire_width design that has *any* feasible C_decap.

    Among that minimum-ww column, take the smallest C_decap that still
    satisfies the spec (cheapest cap that works at the narrowest wire).
    Returns ``None`` if no design in the box meets the spec.
    """
    cols_feasible = feasible.any(axis=1)
    if not cols_feasible.any():
        return None
    i = int(np.argmax(cols_feasible))  # first feasible wire_width index
    j_options = np.where(feasible[i])[0]
    j = int(j_options.min())           # smallest C_decap among feasible at this ww
    return float(ww[i]), float(cd[j])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/droop_v5_conductance.pt"),
    )
    ap.add_argument("--target-mV", type=float, default=0.10,
                    help="Worst-load droop spec, in mV.")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=7)
    args = ap.parse_args()

    target_v = args.target_mV * 1e-3

    model = PDNDroopRegressor(
        EncoderConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            conv_type="admittance",
            drop_edge_p=0.0,
        ),
        target_space="log",
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    print(f"Target: worst-load droop ≤ {args.target_mV:.3f} mV")
    print(f"Surrogate: {args.ckpt.name}\n")

    print(f"{'n_top':>6} | {'rec ww':>10} | {'rec C_decap':>12} | "
          f"{'pred drp':>10} | {'sim drp':>10} | err  ")
    print("-" * 70)
    for n_top in ALL_N_TOP:
        ww, cd, droop_grid, feas = feasibility_grid(model, int(n_top), target_v)
        rec = recommend(ww, cd, feas)
        if rec is None:
            print(f"{n_top:>6} | (no feasible design in box; box max  droop "
                  f"{droop_grid.min()*1e3:.3f} mV at min cost corner)")
            continue
        rec_ww, rec_cd = rec
        pred = predict_worst_droop(model, rec_ww, rec_cd, int(n_top)) * 1e3
        truth = simulate_worst_droop(rec_ww, rec_cd, int(n_top)) * 1e3
        err = (pred - truth) / max(truth, LOG_FLOOR)
        feasibility_pct = 100.0 * feas.mean()
        margin = "✓" if truth <= args.target_mV else "✗"
        print(f"{n_top:>6} | {rec_ww:>10.4f} | {rec_cd:>12.2e} | "
              f"{pred:>9.4f} mV | {truth:>9.4f} mV | "
              f"{err*100:+5.1f}%  {margin}  ({feasibility_pct:.0f}% of box feasible)")

    print()
    print("Notes:")
    print("  * 'rec' = recommended design: smallest wire_width that has any")
    print("    feasible C_decap, with the smallest such C_decap.")
    print("  * 'pred drp' is the surrogate prediction at that point;")
    print("    'sim drp' is the transient-solver ground truth.")
    print("  * '✓' = simulator confirms the spec; '✗' = surrogate optimistically")
    print("    recommended a point that the simulator says fails.")


if __name__ == "__main__":
    main()
