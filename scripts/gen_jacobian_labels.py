"""Generate exact Jacobian labels for Sobolev (gradient-supervised) training.

For each dataset design, one adjoint/backprop pass through the
differentiable sim (`tools/torch_sim.py`) yields the full exact gradient
of the worst-load peak droop w.r.t. every design knob:

    jac_lnww_top [MAX_E]  = ∂droop/∂(ln ww_e), top straps  (nan-padded)
    jac_lnww_bot [MAX_E]  = ∂droop/∂(ln ww_e), bot straps
    jac_lnC      [MAX_D]  = ∂droop/∂(ln C_site), per decap site
    droop_sim             = worst droop recomputed by the torch sim
                            (cross-check vs the stored dataset label)

Cost ~0.3 s (small die) – 1.4 s (13×13) per design; 16 k train designs
≈ 1–2 h on 8 workers. Output: ``jacobians.h5`` next to the dataset,
groups per split, index-aligned with the dataset arrays.

    python scripts/gen_jacobian_labels.py --split train --limit 200
    python scripts/gen_jacobian_labels.py --split train          # full (GPU box)
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.grid_construction import build_regular_pdn
from tools.sampler import (
    ALL_ANCHORS,
    FIXED_CONSTANTS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    FIXED_RSHEET_BOT,
    FIXED_RSHEET_TOP,
)

_G_CACHE: dict = {}


def _graph(n_top: int, n_bot: int, C_decap: float, wt, wb):
    key = (n_top, n_bot)
    if key not in _G_CACHE:
        proto = build_regular_pdn(n_top=n_top, n_bot=n_bot)
        _G_CACHE[key] = proto.n_loads
    n_loads = _G_CACHE[key]
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (n_loads, 1))
    return build_regular_pdn(
        n_top=n_top, n_bot=n_bot,
        Rsheet_top=FIXED_RSHEET_TOP, Rsheet_bot=FIXED_RSHEET_BOT,
        wire_width=0.5, R_via=FIXED_CONSTANTS["R_via"],
        C_decap=C_decap, freq=FIXED_FREQ, loads=loads,
        ww_top_edges=wt, ww_bot_edges=wb,
    )


def _worker(task):
    import torch
    torch.set_num_threads(1)
    from tools.torch_sim import worst_droop_jacobian

    idx, n_top, n_bot, C_decap, wt, wb = task
    g = _graph(n_top, n_bot, C_decap, wt, wb)
    droop, jt, jb, jc, wi = worst_droop_jacobian(
        g, wt.astype(np.float64), wb.astype(np.float64),
        np.full(g.n_decaps, C_decap, dtype=np.float64),
        FIXED_RSHEET_TOP, FIXED_RSHEET_BOT)
    return idx, float(droop.max()), jt, jb, jc, wi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("datasets/regular_v7_anchors/dataset.h5"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: jacobians.h5 next to --data")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_path = args.out or args.data.parent / "jacobians.h5"

    with h5py.File(args.data, "r") as f:
        grp = f["bulk"][args.split]
        n_all = grp["n_top"].shape[0]
        sl = slice(args.start, min(n_all, args.start + (args.limit or n_all)))
        n_top = grp["n_top"][sl]; n_bot = grp["n_bot"][sl]
        gp = grp["global_params"][sl]
        wwt = grp["ww_top_edges"][sl]; wwb = grp["ww_bot_edges"][sl]
        droop_ref = grp["worst_load_droop"][sl]
        max_e = wwt.shape[1]

    # per-anchor edge/decap counts for un-padding
    counts = {}
    for nt, nb in ALL_ANCHORS:
        g = build_regular_pdn(n_top=int(nt), n_bot=int(nb))
        counts[(int(nt), int(nb))] = (g.top_edges.shape[0], g.bot_edges.shape[0],
                                      g.n_decaps)
    max_d = max(c[2] for c in counts.values())

    tasks = []
    for i in range(n_top.shape[0]):
        a = (int(n_top[i]), int(n_bot[i]))
        te, be, _ = counts[a]
        tasks.append((args.start + i, a[0], a[1], float(gp[i, 1]),
                      wwt[i, :te].astype(np.float64),
                      wwb[i, :be].astype(np.float64)))

    print(f"{args.split}[{sl.start}:{sl.stop}] — {len(tasks)} designs, "
          f"{args.workers} workers → {out_path}")
    t0 = time.time()
    jac_t = np.full((len(tasks), max_e), np.nan, dtype=np.float32)
    jac_b = np.full((len(tasks), max_e), np.nan, dtype=np.float32)
    jac_c = np.full((len(tasks), max_d), np.nan, dtype=np.float32)
    droop_sim = np.zeros(len(tasks), dtype=np.float32)
    worst_idx = np.zeros(len(tasks), dtype=np.int32)

    with Pool(args.workers) as pool:
        for k, (idx, dmax, jt, jb, jc, wi) in enumerate(
                pool.imap(_worker, tasks, chunksize=4)):
            i = idx - args.start
            jac_t[i, :jt.size] = jt; jac_b[i, :jb.size] = jb
            jac_c[i, :jc.size] = jc
            droop_sim[i] = dmax; worst_idx[i] = wi
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(tasks)} ({el:.0f}s, {el/(k+1):.2f}s/design)")

    rel = np.abs(droop_sim - droop_ref) / np.maximum(np.abs(droop_ref), 1e-12)
    print(f"cross-check vs dataset droop: median rel {np.median(rel):.2e}, "
          f"max {rel.max():.2e}")

    with h5py.File(out_path, "a") as f:
        gr = f.require_group(args.split)
        for name, arr in (("jac_lnww_top", jac_t), ("jac_lnww_bot", jac_b),
                          ("jac_lnC", jac_c), ("droop_sim", droop_sim),
                          ("worst_idx", worst_idx)):
            if name not in gr:
                full_shape = (n_all,) + arr.shape[1:]
                gr.create_dataset(name, shape=full_shape, dtype=arr.dtype,
                                  fillvalue=np.nan if arr.dtype.kind == "f" else 0)
            gr[name][sl] = arr
        done = gr.require_dataset("done", shape=(n_all,), dtype=bool,
                                  fillvalue=False)
        done[sl] = True
    print(f"→ {out_path} [{args.split}] done "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
