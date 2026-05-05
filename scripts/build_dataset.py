"""Sample LHS parameters, run transient sims, write a single dataset HDF5.

Usage:
    python scripts/build_dataset.py \
        --out datasets/regular_v1/dataset.h5 \
        --n-train 8000 --n-val 1000 --n-test 1000 \
        --seed 42

Adding an extrapolation split (off by default) is one flag away — see
``--n-extrapolation`` and ``EXTRAPOLATION_RANGES`` in ``tools/sampler``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dataset_runner import SimConfig, run_many
from tools.grid_construction import build_regular_pdn
from tools.sampler import DEFAULT_RANGES, EXTRAPOLATION_RANGES, ParamRanges


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _stack_results(results: list[dict]) -> dict[str, np.ndarray]:
    return {
        "peak_droop_bot": np.stack([r["peak_droop_bot"] for r in results]),
        "peak_droop_top": np.stack([r["peak_droop_top"] for r in results]),
        "worst_node_idx": np.array([r["worst_node_idx"] for r in results], dtype=np.int32),
        "worst_node_droop": np.array([r["worst_node_droop"] for r in results], dtype=np.float32),
    }


def _collect_traj_subset(
    results: list[dict], subset_idx: list[int]
) -> dict[str, np.ndarray] | None:
    kept = [results[i] for i in subset_idx if "V_bot_full" in results[i]]
    if not kept:
        return None
    return {
        "indices": np.array(subset_idx, dtype=np.int32),
        "V_bot": np.stack([r["V_bot_full"] for r in kept]).astype(np.float32),
        "V_top": np.stack([r["V_top_full"] for r in kept]).astype(np.float32),
        "t": kept[0]["t_full"].astype(np.float32),
    }


def generate_split(
    name: str,
    n: int,
    ranges: ParamRanges,
    seed: int,
    cfg: SimConfig,
    n_workers: int,
    subset_size: int = 0,
) -> tuple[np.ndarray, dict, dict | None]:
    """Generate one split. Returns (params, stacked_results, optional_traj_subset)."""
    print(f"[{name}] sampling {n} points (seed={seed})...")
    samples = ranges.lhs(n, seed=seed)
    sample_dicts = ranges.to_dict_list(samples)

    subset_idx = list(range(min(subset_size, n))) if subset_size > 0 else []

    print(f"[{name}] simulating on {n_workers or 'all'} workers...")
    t0 = time.time()
    results = run_many(
        sample_dicts,
        keep_traj_idx=set(subset_idx),
        cfg=cfg,
        n_workers=n_workers,
    )
    dt = time.time() - t0
    print(f"[{name}] done in {dt:.1f}s ({1000 * dt / n:.1f} ms/sample)")

    stacked = _stack_results(results)
    traj_subset = _collect_traj_subset(results, subset_idx) if subset_idx else None
    return samples.astype(np.float32), stacked, traj_subset


def write_dataset(out_path: Path, splits: dict[str, tuple], ranges: ParamRanges, cfg: SimConfig, seed: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g_template = build_regular_pdn()

    with h5py.File(out_path, "w") as f:
        f.attrs["version"] = 1
        f.attrs["seed"] = seed
        f.attrs["created_at"] = datetime.now().isoformat()
        f.attrs["git_sha"] = _git_sha()
        f.attrs["param_names"] = json.dumps(list(ranges.names))
        f.attrs["param_ranges"] = json.dumps(
            [(p.lo, p.hi, p.scale) for p in ranges.params]
        )
        f.attrs["sim_config"] = json.dumps(
            {
                "n_periods": cfg.n_periods,
                "steps_per_period": cfg.steps_per_period,
                "warmup_periods": cfg.warmup_periods,
                "Vdd": cfg.Vdd,
                "phase": cfg.phase,
            }
        )
        f.attrs["topology"] = json.dumps(
            {
                "n_top": g_template.n_top,
                "n_bot": g_template.n_bot,
                "n_loads": g_template.n_loads,
                "n_decaps": g_template.n_decaps,
                "n_vias": int(g_template.via_pairs.shape[0]),
                "vdd_pad_top_idx": g_template.vdd_pad_top_idx.tolist(),
                "load_attach_bot_idx": g_template.load_attach_bot_idx.tolist(),
                "decap_attach_bot_idx": g_template.decap_attach_bot_idx.tolist(),
            }
        )

        for name, (params, stacked, traj_subset) in splits.items():
            grp = f.create_group(name)
            grp.create_dataset("params", data=params, compression="gzip")
            for k, v in stacked.items():
                grp.create_dataset(k, data=v, compression="gzip")
            if traj_subset is not None:
                sub = grp.create_group("V_subset")
                for k, v in traj_subset.items():
                    sub.create_dataset(k, data=v, compression="gzip")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/regular_v1/dataset.h5"))
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-extrapolation", type=int, default=0,
                    help="Off by default. Sample from EXTRAPOLATION_RANGES.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--subset-size", type=int, default=100,
                    help="Per-split: how many samples retain full V(t) for diagnostics.")
    args = ap.parse_args()

    cfg = SimConfig()
    splits: dict[str, tuple] = {}

    splits["train"] = generate_split(
        "train", args.n_train, DEFAULT_RANGES,
        seed=args.seed, cfg=cfg, n_workers=args.n_workers,
        subset_size=args.subset_size,
    )
    splits["val"] = generate_split(
        "val", args.n_val, DEFAULT_RANGES,
        seed=args.seed + 1, cfg=cfg, n_workers=args.n_workers,
        subset_size=0,
    )
    splits["test"] = generate_split(
        "test", args.n_test, DEFAULT_RANGES,
        seed=args.seed + 2, cfg=cfg, n_workers=args.n_workers,
        subset_size=0,
    )
    if args.n_extrapolation > 0:
        splits["extrapolation"] = generate_split(
            "extrapolation", args.n_extrapolation, EXTRAPOLATION_RANGES,
            seed=args.seed + 3, cfg=cfg, n_workers=args.n_workers,
            subset_size=0,
        )

    write_dataset(args.out, splits, DEFAULT_RANGES, cfg, seed=args.seed)


if __name__ == "__main__":
    main()
