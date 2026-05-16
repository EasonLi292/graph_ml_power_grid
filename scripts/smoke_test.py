"""Build canonical PDNs across pad patterns, run transient + DC, print maps.

Sanity check that the data pipeline + ground-truth solvers are wired up.
For each of the four pad patterns:

* shows the topology (n_loads, n_decaps, derived per-segment R values)
* runs a short transient (5 ns @ 10 ps) at the default homogeneous load
* prints peak droop AND static IR drop on M_bot
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.grid_construction import PAD_PATTERNS, build_regular_pdn
from tools.transient_solver import simulate, solve_static_dc


def _droop_for(pad_pattern: str) -> None:
    g = build_regular_pdn(pad_pattern=pad_pattern)
    res = simulate(g, t_end=5e-9, dt=1e-11)
    dc = solve_static_dc(g)
    peak = g.Vdd - res["V_bot"].min(axis=0)
    static = g.Vdd - dc["V_bot"]

    print(f"\n-- pad_pattern={pad_pattern} --")
    print(f"  mesh: {g.n_top}x{g.n_top} top + {g.n_bot}x{g.n_bot} bot")
    print(f"  loads: {g.n_loads}, decaps: {g.n_decaps}, vias: {g.via_pairs.shape[0]}")
    print(f"  derived  R_top={g.R_top:.3f}Ω  R_bot={g.R_bot:.3f}Ω  "
          f"(pitch_top={g.pitch_top}, pitch_bot={g.pitch_bot}, width={g.wire_width})")
    print(f"  peak droop on M_bot (mV):  max {peak.max()*1e3:.2f}  mean {peak.mean()*1e3:.2f}")
    print(f"  static IR-drop on M_bot (mV): max {static.max()*1e3:.2f}  mean {static.mean()*1e3:.2f}")
    print("  peak droop map:")
    print(np.round(peak.reshape(g.n_bot, g.n_bot) * 1e3, 2))


def main() -> None:
    for pp in PAD_PATTERNS:
        _droop_for(pp)


if __name__ == "__main__":
    main()
