"""Validate the SPICE parser + transient solver against an IBM .output file.

Parses ``ibmpg<k>t.spice``, runs our backward-Euler MNA solver, and compares
the per-probe-node v(t) to the shipped ``ibmpg<k>t.output`` ground truth.

Usage:
    python3.12 scripts/validate_ibmpg.py --bench ibmpg1t
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.spice_parser import parse_netlist
from tools.spice_solver import solve_transient

IBMPG = Path("datasets/ibmpg")


def _unzip(stem: str, ext: str) -> Path:
    out = IBMPG / f"{stem}.{ext}"
    if out.exists():
        return out
    import bz2
    import gzip
    for comp, opener in ((".bz2", bz2.open), (".gz", gzip.open)):
        z = IBMPG / f"{stem}.{ext}{comp}"
        if z.exists():
            with opener(z, "rt") as fi, open(out, "w") as fo:
                fo.write(fi.read())
            return out
    raise FileNotFoundError(f"{stem}.{ext}[.bz2/.gz] not found in {IBMPG}")


def parse_output(path: Path) -> dict[str, np.ndarray]:
    """Parse an IBM .output: blocks of 'Node: <name>' then 't value' rows."""
    series: dict[str, list[tuple[float, float]]] = {}
    cur = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("node:"):
            cur = s.split(":", 1)[1].strip()
            series[cur] = []
        elif cur is not None:
            parts = s.split()
            if len(parts) == 2:
                try:
                    series[cur].append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    return {k: np.array(v) for k, v in series.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ibmpg1t")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    spice = _unzip(args.bench, "spice")
    out_f = _unzip(args.bench, "output")

    print(f"parsing {spice.name} ...")
    circ = parse_netlist(spice)
    print("  ", circ.summary())

    print("solving (backward-Euler MNA) ...")
    t0 = time.time()
    res = solve_transient(circ)
    print(f"  done in {time.time() - t0:.1f}s, {len(res['t'])} steps")

    ref = parse_output(out_f)
    t_sim = res["t"]
    pred = {n: v for n, v in zip(res["probe_names"], res["probe_V"].T)}

    print(f"\ncomparing {len(ref)} probe nodes vs .output:")
    print(f"  {'node':<24} {'max|err| (V)':>13} {'rel (vs swing)':>15}")
    abs_errs, rel_errs = [], []
    for name, arr in sorted(ref.items()):
        t_ref, v_ref = arr[:, 0], arr[:, 1]
        if name not in pred:
            print(f"  {name:<24} (not probed by solver)")
            continue
        v_interp = np.interp(t_ref, t_sim, pred[name])
        ae = np.max(np.abs(v_interp - v_ref))
        swing = max(v_ref.max() - v_ref.min(), 1e-12)
        re = ae / swing
        abs_errs.append(ae); rel_errs.append(re)
        if len(abs_errs) <= 12:
            print(f"  {name:<24} {ae:>13.3e} {re:>14.1%}")
    if abs_errs:
        print(f"\n  median max|err| = {np.median(abs_errs):.3e} V | "
              f"median rel = {np.median(rel_errs):.1%} | "
              f"worst rel = {np.max(rel_errs):.1%}  over {len(abs_errs)} nodes")

    if args.plot and pred:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [n for n in res["probe_names"] if n in ref][:6]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
        for ax, name in zip(axes.ravel(), names):
            arr = ref[name]
            ax.plot(arr[:, 0] * 1e9, arr[:, 1] * 1e3, "k-", lw=2, label="IBM .output")
            ax.plot(t_sim * 1e9, pred[name] * 1e3, "r--", lw=1, label="our solver")
            ax.set_title(name, fontsize=9); ax.set_xlabel("t (ns)"); ax.set_ylabel("V (mV)")
            ax.legend(fontsize=7)
        png = Path("docs/figures/fig_ibmpg_validate.png")
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.suptitle(f"{args.bench}: solver vs shipped .output")
        fig.savefig(png, dpi=120)
        print(f"  figure → {png}")


if __name__ == "__main__":
    main()
