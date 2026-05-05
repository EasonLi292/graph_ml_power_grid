"""Build the canonical regular PDN, run a transient solve, print peak droop.

Sanity check that the data pipeline + ground-truth solver are wired up.
Expected output on default parameters: peak droop on the order of a few
millivolts at the centre of M_bot, ~0 mV at the four corner pads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.grid_construction import build_regular_pdn
from tools.transient_solver import simulate


def main() -> None:
    g = build_regular_pdn()
    res = simulate(g, t_end=5e-9, dt=1e-11)
    V_bot = res["V_bot"]

    droop = g.Vdd - V_bot.min(axis=0)

    print(f"mesh: {g.n_top}x{g.n_top} top + {g.n_bot}x{g.n_bot} bot")
    print(f"loads: {g.n_loads}, decaps: {g.n_decaps}, vias: {g.via_pairs.shape[0]}")
    print(f"timesteps: {res['t'].size} (t_end=5 ns, dt=10 ps)")
    print(
        f"peak droop on M_bot: max {droop.max() * 1e3:.2f} mV, "
        f"mean {droop.mean() * 1e3:.2f} mV"
    )
    print("droop map on M_bot (mV):")
    print(np.round(droop.reshape(g.n_bot, g.n_bot) * 1e3, 2))


if __name__ == "__main__":
    main()
