"""Measure retraining (seed) variance of the sensitivity gate.

Item-2 exposed the problem this script exists to fix: two runs of the *same*
config, both gated at n=12, disagree far outside the gate's own CIs.

    L7 @ (4,7), n=12:  0.86 [0.79,0.91] / rho +0.654   (original checkpoint)
                       0.59 [0.50,0.67] / rho +0.109   (retrained, item 2)

The CIs do not overlap. The gate's intervals are a design-level bootstrap
for a *fixed* checkpoint, so they quantify perturbation sampling only —
they say nothing about the spread induced by retraining. Every architectural
conclusion on this track (the depth trade-off, the (4,7) collapse, the whole
premise for impedance attention) currently rests on single draws of a
quantity whose spread is unmeasured.

This runs the same config across several seeds and reports the spread, so
future comparisons can be judged against it.

    # 1. print the training commands (run them on the box; nothing launches here)
    python scripts/seed_variance_study.py --emit --seeds 5 --n-layers 7

    # 2. aggregate whatever gate JSONs exist
    python scripts/seed_variance_study.py --aggregate --n-layers 7

A comparison is only meaningful if it exceeds the seed spread reported here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ANCHORS = ("3,7", "4,7", "7,13")


def emit(seeds, n_layers, conv_type, data, epochs, outdir):
    print(f"# {seeds} seeds x n_layers={n_layers}; ~30 min/seed at L7, ~60 at L20")
    print(f"# safe to run concurrently (distinct --ckpt / --out paths)")
    for s in range(seeds):
        ck = f"checkpoints/seedvar_l{n_layers}_s{s}.pt"
        out = f"{outdir}/seedvar_l{n_layers}_s{s}.json"
        print(f"\npython scripts/train_droop.py --data {data} "
              f"--conv-type {conv_type} --n-layers {n_layers} --epochs {epochs} "
              f"--seed {s} --device cuda --ckpt {ck}")
        print(f"python scripts/sensitivity_gate.py --ckpt {ck} "
              f"--conv-type {conv_type} --n-layers {n_layers} "
              f"--n-designs 12 --seed 0 --out {out}")
    print(f"\n# then:  python scripts/seed_variance_study.py --aggregate "
          f"--n-layers {n_layers}")


def aggregate(n_layers, outdir):
    paths = sorted(Path(outdir).glob(f"seedvar_l{n_layers}_s*.json"))
    if not paths:
        print(f"no gate JSONs matching {outdir}/seedvar_l{n_layers}_s*.json")
        return
    rows = {a: {"sign": [], "rho": [], "mag": []} for a in ANCHORS}
    for p in paths:
        d = json.loads(p.read_text())
        for r in d["results"]:
            a = f"{r['anchor'][0]},{r['anchor'][1]}"
            if a in rows:
                rows[a]["sign"].append(r["ww_sign_acc"])
                rows[a]["rho"].append(r["site_rank_spearman_mean"])
                rows[a]["mag"].append(r["magnitude_ratio_median"])

    print(f"n_layers={n_layers}, {len(paths)} seeds: {[p.stem[-2:] for p in paths]}")
    print(f"{'anchor':>8} {'sign mean+-sd':>18} {'range':>16} "
          f"{'rho mean+-sd':>18} {'range':>16}")
    summary = {}
    for a in ANCHORS:
        s, r = np.array(rows[a]["sign"]), np.array(rows[a]["rho"])
        if s.size == 0:
            continue
        print(f"{a:>8} {s.mean():>9.3f}+-{s.std(ddof=1) if s.size>1 else 0:.3f}"
              f" {f'[{s.min():.2f},{s.max():.2f}]':>16}"
              f" {r.mean():>+9.3f}+-{r.std(ddof=1) if r.size>1 else 0:.3f}"
              f" {f'[{r.min():+.2f},{r.max():+.2f}]':>16}")
        summary[a] = {
            "n_seeds": int(s.size),
            "sign_mean": float(s.mean()), "sign_sd": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "sign_range": [float(s.min()), float(s.max())],
            "rho_mean": float(r.mean()), "rho_sd": float(r.std(ddof=1)) if r.size > 1 else 0.0,
            "rho_range": [float(r.min()), float(r.max())],
            "mag_mean": float(np.mean(rows[a]["mag"])),
        }
    out = Path(outdir) / f"seedvar_summary_l{n_layers}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out}")
    print("Treat any architecture comparison smaller than these ranges as "
          "undecided, regardless of the gate's own CIs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--conv-type", default="edgeconv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--data", default="datasets/regular_v7_anchors/dataset.h5")
    ap.add_argument("--outdir", default="docs/analysis")
    args = ap.parse_args()
    if args.emit:
        emit(args.seeds, args.n_layers, args.conv_type, args.data,
             args.epochs, args.outdir)
    if args.aggregate or not args.emit:
        aggregate(args.n_layers, args.outdir)


if __name__ == "__main__":
    main()
