"""Sample 3-knob parameters, run transient + DC sims, write the dataset HDF5.

Groups written:

* ``/bulk/{train,val,test}/`` — LHS over ``(wire_width, C_decap)``.
  Train + val sample ``n_top ∈ TRAIN_N_TOP``; test samples
  ``n_top ∈ TEST_N_TOP``. Because test draws topologies never seen in
  training, a passing test score requires the model to read the graph
  rather than memorize a per-n_top mapping.
* ``/analysis/sweeps/<axis>/n_top_<N>/`` — 1-D sweep along a continuous
  axis (``wire_width`` or ``C_decap``), the other axis held at its
  median, repeated for every ``n_top`` in ``ALL_N_TOP``. Used for
  latent-space / sensitivity analysis, not training.

Usage:
    python scripts/build_dataset.py \\
        --out datasets/regular_v5/dataset.h5 \\
        --n-train 16000 --n-val 2000 --n-test 2000 \\
        --sweep-points 50 --seed 42
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
from tools.grid_construction import BOT_COL_PATTERN, build_regular_pdn
from tools.sampler import (
    ALL_N_TOP,
    FIXED_CONSTANTS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    GLOBAL_RANGES,
    TEST_N_TOP,
    TRAIN_N_TOP,
    axis_sweep,
    sample_n_top,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _check_invariant_n_loads() -> int:
    """``n_loads`` is constant across every valid ``n_top`` (loads sit
    on M_bot at fixed (row, boundary) positions). Sanity-check at import
    time so a future schema change can't silently let it drift."""
    counts = {nt: build_regular_pdn(n_top=nt).n_loads for nt in ALL_N_TOP}
    if len(set(counts.values())) != 1:
        raise RuntimeError(
            f"n_loads should be invariant across ALL_N_TOP, got {counts}"
        )
    return next(iter(counts.values()))


N_LOADS: int = _check_invariant_n_loads()


# ---------------------------------------------------------------------------
# Per-sample assembly
# ---------------------------------------------------------------------------

def _assemble_sample_dicts(
    global_samples: np.ndarray,
    n_top_per_sample: np.ndarray,
) -> list[dict]:
    """Build the per-sample param dict ``run_one`` consumes."""
    out = []
    name_idx = {n: i for i, n in enumerate(GLOBAL_RANGES.names)}
    for i in range(global_samples.shape[0]):
        loads = np.tile(
            np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]], dtype=np.float64),
            (N_LOADS, 1),
        )
        d: dict = dict(FIXED_CONSTANTS)
        d["wire_width"] = float(global_samples[i, name_idx["wire_width"]])
        d["C_decap"]    = float(global_samples[i, name_idx["C_decap"]])
        d["n_top"]      = int(n_top_per_sample[i])
        d["loads"]      = loads
        out.append(d)
    return out


def _stack_results(results: list[dict]) -> dict[str, np.ndarray]:
    return {
        "peak_droop_loads":   np.stack([r["peak_droop_loads"]   for r in results]),
        "static_droop_loads": np.stack([r["static_droop_loads"] for r in results]),
        "worst_load_idx":     np.array([r["worst_load_idx"]     for r in results], dtype=np.int32),
        "worst_load_droop":   np.array([r["worst_load_droop"]   for r in results], dtype=np.float32),
    }


def _collect_traj_subset(results: list[dict], subset_idx: list[int]) -> dict | None:
    kept = [results[i] for i in subset_idx if "V_loads_full" in results[i]]
    if not kept:
        return None
    return {
        "indices": np.array(subset_idx, dtype=np.int32),
        "V_loads": np.stack([r["V_loads_full"] for r in kept]).astype(np.float32),
        "t":       kept[0]["t_full"].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_split(
    name: str,
    n: int,
    n_top_choices: tuple[int, ...],
    seed: int,
    cfg: SimConfig,
    n_workers: int,
    subset_size: int = 0,
) -> dict:
    print(f"[{name}] sampling {n} points (seed={seed}) over n_top={n_top_choices}...")
    global_samples = GLOBAL_RANGES.lhs(n, seed=seed)
    n_top_per = sample_n_top(n, seed=seed + 100_000, choices=n_top_choices)

    sample_dicts = _assemble_sample_dicts(global_samples, n_top_per)
    subset_idx = list(range(min(subset_size, n))) if subset_size > 0 else []

    print(f"[{name}] simulating on {n_workers or 'all'} workers...")
    t0 = time.time()
    results = run_many(sample_dicts, keep_traj_idx=set(subset_idx), cfg=cfg, n_workers=n_workers)
    dt = time.time() - t0
    print(f"[{name}] done in {dt:.1f}s ({1000 * dt / max(n, 1):.1f} ms/sample)")

    return {
        "global_params": global_samples.astype(np.float32),
        "n_top":         n_top_per.astype(np.int16),
        "results":       _stack_results(results),
        "V_subset":      _collect_traj_subset(results, subset_idx) if subset_idx else None,
    }


def generate_sweep(
    axis: str,
    n_top: int,
    n_points: int,
    cfg: SimConfig,
    n_workers: int,
) -> dict:
    axis_vals, medians = axis_sweep(axis, n_points)

    global_samples = np.zeros((n_points, GLOBAL_RANGES.d), dtype=np.float64)
    for j, name in enumerate(GLOBAL_RANGES.names):
        global_samples[:, j] = medians[name]
    global_samples[:, GLOBAL_RANGES.names.index(axis)] = axis_vals

    n_top_per = np.full(n_points, n_top, dtype=np.int16)
    sample_dicts = _assemble_sample_dicts(global_samples, n_top_per)

    t0 = time.time()
    results = run_many(sample_dicts, cfg=cfg, n_workers=n_workers)
    dt = time.time() - t0
    print(f"  sweep[{axis}|n_top={n_top}] {n_points}pts in {dt:.1f}s")

    return {
        "axis_values":   axis_vals.astype(np.float32),
        "global_params": global_samples.astype(np.float32),
        "n_top":         n_top_per,
        "results":       _stack_results(results),
    }


# ---------------------------------------------------------------------------
# H5 writer
# ---------------------------------------------------------------------------

def _write_split(grp: h5py.Group, payload: dict) -> None:
    grp.create_dataset("global_params", data=payload["global_params"], compression="gzip")
    grp.create_dataset("n_top",         data=payload["n_top"],         compression="gzip")
    for k, v in payload["results"].items():
        grp.create_dataset(k, data=v, compression="gzip")
    if "axis_values" in payload:
        grp.create_dataset("axis_values", data=payload["axis_values"], compression="gzip")
    if payload.get("V_subset") is not None:
        sub = grp.create_group("V_subset")
        for k, v in payload["V_subset"].items():
            sub.create_dataset(k, data=v, compression="gzip")


def write_dataset(
    out_path: Path,
    bulk: dict[str, dict],
    sweeps: dict[str, dict[int, dict]],
    cfg: SimConfig,
    seed: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["version"]    = 5
        f.attrs["seed"]       = seed
        f.attrs["created_at"] = datetime.now().isoformat()
        f.attrs["git_sha"]    = _git_sha()

        f.attrs["global_param_names"] = json.dumps(list(GLOBAL_RANGES.names))
        f.attrs["train_n_top"]        = json.dumps(list(TRAIN_N_TOP))
        f.attrs["test_n_top"]         = json.dumps(list(TEST_N_TOP))
        f.attrs["bot_col_pattern"]    = json.dumps(list(BOT_COL_PATTERN))
        f.attrs["n_loads"]            = int(N_LOADS)
        f.attrs["load_attr_row"]      = json.dumps(
            {"I_peak": FIXED_I_PEAK, "freq": FIXED_FREQ, "duty": FIXED_DUTY, "phase": FIXED_PHASE}
        )
        f.attrs["fixed_constants"]    = json.dumps(FIXED_CONSTANTS)
        f.attrs["param_ranges"]       = json.dumps(
            {"global": [(p.lo, p.hi, p.scale) for p in GLOBAL_RANGES.params]}
        )
        f.attrs["sim_config"] = json.dumps(
            {
                "steps_per_period":    cfg.steps_per_period,
                "measure_periods":     cfg.measure_periods,
                "min_warmup_periods":  cfg.min_warmup_periods,
                "settling_tau_factor": cfg.settling_tau_factor,
                "Vdd":                 cfg.Vdd,
            }
        )

        topo = {}
        for nt in ALL_N_TOP:
            g_t = build_regular_pdn(n_top=nt)
            topo[str(nt)] = {
                "n_top":              nt,
                "n_bot":              g_t.n_bot,
                "pitch_top":          g_t.pitch_top,
                "pitch_bot":          g_t.pitch_bot,
                "n_loads":            g_t.n_loads,
                "n_decaps":           g_t.n_decaps,
                "n_vias":             int(g_t.via_pairs.shape[0]),
                "vdd_pad_top_idx":    g_t.vdd_pad_top_idx.tolist(),
                "vss_pad_top_idx":    g_t.vss_pad_top_idx.tolist(),
                "top_is_vdd":         g_t.top_is_vdd.tolist(),
                "load_pairs":         g_t.load_pairs.tolist(),
                "decap_pairs":        g_t.decap_pairs.tolist(),
            }
        f.attrs["topology"] = json.dumps(topo)

        bulk_grp = f.create_group("bulk")
        for split_name, payload in bulk.items():
            _write_split(bulk_grp.create_group(split_name), payload)

        if sweeps:
            sweep_grp = f.create_group("analysis").create_group("sweeps")
            for axis, by_n_top in sweeps.items():
                ax_grp = sweep_grp.create_group(axis)
                for nt, payload in by_n_top.items():
                    _write_split(ax_grp.create_group(f"n_top_{nt}"), payload)

    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/regular_v5/dataset.h5"))
    ap.add_argument("--n-train",      type=int, default=16000)
    ap.add_argument("--n-val",        type=int, default=2000)
    ap.add_argument("--n-test",       type=int, default=2000)
    ap.add_argument("--sweep-points", type=int, default=50,
                    help="Per-axis-per-n_top; 0 disables sweeps.")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--n-workers",    type=int, default=None)
    ap.add_argument("--subset-size",  type=int, default=200,
                    help="Train-only: how many samples retain full V(t).")
    args = ap.parse_args()

    cfg = SimConfig()

    bulk = {
        "train": generate_split(
            "train", args.n_train, TRAIN_N_TOP,
            seed=args.seed, cfg=cfg, n_workers=args.n_workers,
            subset_size=args.subset_size,
        ),
        "val": generate_split(
            "val", args.n_val, TRAIN_N_TOP,
            seed=args.seed + 1, cfg=cfg, n_workers=args.n_workers,
        ),
        "test": generate_split(
            "test", args.n_test, TEST_N_TOP,
            seed=args.seed + 2, cfg=cfg, n_workers=args.n_workers,
        ),
    }

    sweeps: dict[str, dict[int, dict]] = {}
    if args.sweep_points > 0:
        sweep_axes = list(GLOBAL_RANGES.names)
        print(f"[sweeps] {len(sweep_axes)} axes × {len(ALL_N_TOP)} n_top × "
              f"{args.sweep_points} points")
        for axis in sweep_axes:
            sweeps[axis] = {}
            for nt in ALL_N_TOP:
                sweeps[axis][nt] = generate_sweep(
                    axis, nt, args.sweep_points,
                    cfg=cfg, n_workers=args.n_workers,
                )

    write_dataset(args.out, bulk, sweeps, cfg, seed=args.seed)


if __name__ == "__main__":
    main()
