"""Sample LHS parameters, run transient + DC sims, write the dataset HDF5.

Three groups are written:

* ``/bulk/{train,val,test}/`` — LHS samples over global continuous params,
  iid per-load draws, drawn from the *training* pad patterns. The model
  trains and is validated/tested here.
* ``/ood/<pattern>/`` — same distribution but drawn exclusively from the
  held-out pattern(s). Test-only: probes pad-pattern generalization.
* ``/analysis/sweeps/<axis>/<pattern>/`` — 1-D parameter sweeps with all
  other knobs held at their medians, per training pad pattern. Used for
  latent-space / sensitivity analysis, not training.

Usage:
    python scripts/build_dataset.py \\
        --out datasets/regular_v2/dataset.h5 \\
        --n-train 16000 --n-val 2000 --n-test 2000 --n-ood 2000 \\
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
from tools.grid_construction import PAD_PATTERNS, build_regular_pdn
from tools.sampler import (
    GLOBAL_RANGES,
    OOD_PAD_PATTERNS,
    PER_LOAD_RANGES,
    TRAIN_PAD_PATTERNS,
    ParamRanges,
    axis_sweep,
    sample_pad_patterns,
    sample_per_load,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _n_loads_for(pattern: str) -> int:
    return int(build_regular_pdn(pad_pattern=pattern).n_loads)


def _max_n_loads() -> int:
    return max(_n_loads_for(p) for p in PAD_PATTERNS)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _assemble_sample_dicts(
    global_samples: np.ndarray,
    pattern_idx: np.ndarray,
    per_load: np.ndarray,
) -> list[dict]:
    """Bundle into the per-sample dict that ``run_one`` consumes."""
    out = []
    for i, pp_idx in enumerate(pattern_idx):
        pattern = PAD_PATTERNS[int(pp_idx)]
        n_loads = _n_loads_for(pattern)
        # broadcast global freq into the 4-col loads array
        loads = np.zeros((n_loads, 4), dtype=np.float64)
        loads[:, 0] = per_load[i, :n_loads, 0]                   # I_peak
        loads[:, 1] = float(global_samples[i, GLOBAL_RANGES.names.index("freq")])
        loads[:, 2] = per_load[i, :n_loads, 1]                   # duty
        loads[:, 3] = per_load[i, :n_loads, 2]                   # phase
        d: dict = {
            name: float(global_samples[i, j])
            for j, name in enumerate(GLOBAL_RANGES.names)
        }
        d["pad_pattern"] = pattern
        d["loads"] = loads
        out.append(d)
    return out


def _stack_results(results: list[dict], n_bot_sq: int, n_top_sq: int) -> dict[str, np.ndarray]:
    return {
        "peak_droop_bot": np.stack([r["peak_droop_bot"] for r in results]),
        "peak_droop_top": np.stack([r["peak_droop_top"] for r in results]),
        "static_droop_bot": np.stack([r["static_droop_bot"] for r in results]),
        "static_droop_top": np.stack([r["static_droop_top"] for r in results]),
        "worst_node_idx": np.array([r["worst_node_idx"] for r in results], dtype=np.int32),
        "worst_node_droop": np.array([r["worst_node_droop"] for r in results], dtype=np.float32),
    }


def _collect_traj_subset(results: list[dict], subset_idx: list[int]) -> dict | None:
    kept = [results[i] for i in subset_idx if "V_bot_full" in results[i]]
    if not kept:
        return None
    return {
        "indices": np.array(subset_idx, dtype=np.int32),
        "V_bot": np.stack([r["V_bot_full"] for r in kept]).astype(np.float32),
        "V_top": np.stack([r["V_top_full"] for r in kept]).astype(np.float32),
        "t": kept[0]["t_full"].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def _pad_load_x(per_load: np.ndarray, freq_col: np.ndarray, max_n_loads: int) -> np.ndarray:
    """Convert ``per_load[N, K, 3]`` (I_peak, duty, phase) into the 4-col
    storage layout ``[N, max_n_loads, 4]`` = (I_peak, freq, duty, phase),
    zero-padded beyond each sample's actual load count.
    """
    N = per_load.shape[0]
    out = np.zeros((N, max_n_loads, 4), dtype=np.float32)
    K = min(per_load.shape[1], max_n_loads)
    out[:, :K, 0] = per_load[:, :K, 0]
    out[:, :K, 1] = freq_col[:, None]
    out[:, :K, 2] = per_load[:, :K, 1]
    out[:, :K, 3] = per_load[:, :K, 2]
    return out


def generate_split(
    name: str,
    n: int,
    pattern_choices: tuple[str, ...],
    seed: int,
    cfg: SimConfig,
    n_workers: int,
    subset_size: int = 0,
    max_n_loads: int | None = None,
) -> dict:
    """Generate one split. Returns a dict ready to write to H5.

    Each sample draws:
      * Global continuous params via LHS over ``GLOBAL_RANGES``
      * Pad pattern uniformly from ``pattern_choices``
      * Per-load (I_peak, duty, phase) iid from ``PER_LOAD_RANGES`` (one
        triple per load instance — count depends on the pattern)
    """
    if max_n_loads is None:
        max_n_loads = _max_n_loads()

    print(f"[{name}] sampling {n} points (seed={seed}) over patterns={pattern_choices}...")
    global_samples = GLOBAL_RANGES.lhs(n, seed=seed)
    pattern_idx = sample_pad_patterns(n, seed=seed + 100_000, choices=pattern_choices)

    n_loads_per = [_n_loads_for(PAD_PATTERNS[int(i)]) for i in pattern_idx]
    per_load = sample_per_load(
        n_loads_per, seed=seed + 200_000, ranges=PER_LOAD_RANGES, max_n_loads=max_n_loads
    )

    sample_dicts = _assemble_sample_dicts(global_samples, pattern_idx, per_load)

    subset_idx = list(range(min(subset_size, n))) if subset_size > 0 else []

    print(f"[{name}] simulating on {n_workers or 'all'} workers...")
    t0 = time.time()
    results = run_many(sample_dicts, keep_traj_idx=set(subset_idx), cfg=cfg, n_workers=n_workers)
    dt = time.time() - t0
    print(f"[{name}] done in {dt:.1f}s ({1000 * dt / n:.1f} ms/sample)")

    g_template = build_regular_pdn()
    stacked = _stack_results(results, g_template.n_bot_nodes, g_template.n_top_nodes)

    freq_col = global_samples[:, GLOBAL_RANGES.names.index("freq")].astype(np.float32)
    load_x = _pad_load_x(per_load, freq_col, max_n_loads)

    return {
        "global_params": global_samples.astype(np.float32),
        "pad_pattern_idx": pattern_idx.astype(np.int8),
        "load_x": load_x,
        "n_loads": np.asarray(n_loads_per, dtype=np.int16),
        "results": stacked,
        "V_subset": _collect_traj_subset(results, subset_idx) if subset_idx else None,
    }


def generate_sweep(
    axis: str,
    pattern: str,
    n_points: int,
    seed: int,
    cfg: SimConfig,
    n_workers: int,
    max_n_loads: int,
) -> dict:
    """1-D sweep along ``axis`` at all-other-medians, fixed pad pattern."""
    axis_vals, g_med, l_med = axis_sweep(axis, n_points)
    n_loads = _n_loads_for(pattern)

    global_names = GLOBAL_RANGES.names
    global_samples = np.zeros((n_points, GLOBAL_RANGES.d), dtype=np.float64)
    for j, name in enumerate(global_names):
        global_samples[:, j] = g_med[name]
    if axis in global_names:
        global_samples[:, global_names.index(axis)] = axis_vals

    per_load = np.zeros((n_points, max_n_loads, PER_LOAD_RANGES.d), dtype=np.float32)
    for j, name in enumerate(PER_LOAD_RANGES.names):
        per_load[:, :n_loads, j] = l_med[name]
    if axis in PER_LOAD_RANGES.names:
        per_load[:, :n_loads, PER_LOAD_RANGES.names.index(axis)] = axis_vals[:, None]

    pattern_idx = np.full(n_points, PAD_PATTERNS.index(pattern), dtype=np.int8)
    sample_dicts = _assemble_sample_dicts(global_samples, pattern_idx, per_load)

    t0 = time.time()
    results = run_many(sample_dicts, cfg=cfg, n_workers=n_workers)
    dt = time.time() - t0
    print(f"  sweep[{axis}|{pattern}] {n_points}pts in {dt:.1f}s")

    g_template = build_regular_pdn(pad_pattern=pattern)
    stacked = _stack_results(results, g_template.n_bot_nodes, g_template.n_top_nodes)
    freq_col = global_samples[:, global_names.index("freq")].astype(np.float32)

    return {
        "axis_values": axis_vals.astype(np.float32),
        "global_params": global_samples.astype(np.float32),
        "pad_pattern_idx": pattern_idx,
        "load_x": _pad_load_x(per_load, freq_col, max_n_loads),
        "n_loads": np.full(n_points, n_loads, dtype=np.int16),
        "results": stacked,
    }


# ---------------------------------------------------------------------------
# H5 writer
# ---------------------------------------------------------------------------

def _write_split(grp: h5py.Group, payload: dict) -> None:
    grp.create_dataset("global_params", data=payload["global_params"], compression="gzip")
    grp.create_dataset("pad_pattern_idx", data=payload["pad_pattern_idx"], compression="gzip")
    grp.create_dataset("load_x", data=payload["load_x"], compression="gzip")
    grp.create_dataset("n_loads", data=payload["n_loads"], compression="gzip")
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
    ood: dict[str, dict],
    sweeps: dict[str, dict[str, dict]],
    cfg: SimConfig,
    seed: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_n = _max_n_loads()

    with h5py.File(out_path, "w") as f:
        f.attrs["version"] = 3
        f.attrs["seed"] = seed
        f.attrs["created_at"] = datetime.now().isoformat()
        f.attrs["git_sha"] = _git_sha()
        f.attrs["global_param_names"] = json.dumps(list(GLOBAL_RANGES.names))
        f.attrs["per_load_param_names"] = json.dumps(list(PER_LOAD_RANGES.names))
        f.attrs["load_x_columns"] = json.dumps(["I_peak", "freq", "duty", "phase"])
        f.attrs["pad_patterns"] = json.dumps(list(PAD_PATTERNS))
        f.attrs["train_pad_patterns"] = json.dumps(list(TRAIN_PAD_PATTERNS))
        f.attrs["ood_pad_patterns"] = json.dumps(list(OOD_PAD_PATTERNS))
        f.attrs["max_n_loads"] = int(max_n)
        f.attrs["param_ranges"] = json.dumps(
            {
                "global": [(p.lo, p.hi, p.scale) for p in GLOBAL_RANGES.params],
                "per_load": [(p.lo, p.hi, p.scale) for p in PER_LOAD_RANGES.params],
            }
        )
        f.attrs["sim_config"] = json.dumps(
            {
                "steps_per_period": cfg.steps_per_period,
                "measure_periods": cfg.measure_periods,
                "min_warmup_periods": cfg.min_warmup_periods,
                "settling_tau_factor": cfg.settling_tau_factor,
                "Vdd": cfg.Vdd,
            }
        )
        topo_blocks = {}
        for pp in PAD_PATTERNS:
            g_t = build_regular_pdn(pad_pattern=pp)
            topo_blocks[pp] = {
                "n_top": g_t.n_top,
                "n_bot": g_t.n_bot,
                "pitch_top": g_t.pitch_top,
                "pitch_bot": g_t.pitch_bot,
                "n_loads": g_t.n_loads,
                "n_decaps": g_t.n_decaps,
                "n_vias": int(g_t.via_pairs.shape[0]),
                "vdd_pad_top_idx": g_t.vdd_pad_top_idx.tolist(),
                "load_attach_bot_idx": g_t.load_attach_bot_idx.tolist(),
                "decap_attach_bot_idx": g_t.decap_attach_bot_idx.tolist(),
            }
        f.attrs["topology"] = json.dumps(topo_blocks)

        bulk_grp = f.create_group("bulk")
        for split_name, payload in bulk.items():
            _write_split(bulk_grp.create_group(split_name), payload)

        ood_grp = f.create_group("ood")
        for pattern, payload in ood.items():
            _write_split(ood_grp.create_group(pattern), payload)

        if sweeps:
            sweep_grp = f.create_group("analysis").create_group("sweeps")
            for axis, by_pattern in sweeps.items():
                ax_grp = sweep_grp.create_group(axis)
                for pattern, payload in by_pattern.items():
                    _write_split(ax_grp.create_group(pattern), payload)

    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/regular_v2/dataset.h5"))
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-ood", type=int, default=2000)
    ap.add_argument("--sweep-points", type=int, default=50,
                    help="Per-axis-per-pattern; 0 disables sweeps.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--subset-size", type=int, default=200,
                    help="Per-split: how many train samples retain full V(t).")
    args = ap.parse_args()

    cfg = SimConfig()
    max_n = _max_n_loads()

    bulk = {
        "train": generate_split(
            "train", args.n_train, TRAIN_PAD_PATTERNS,
            seed=args.seed, cfg=cfg, n_workers=args.n_workers,
            subset_size=args.subset_size, max_n_loads=max_n,
        ),
        "val": generate_split(
            "val", args.n_val, TRAIN_PAD_PATTERNS,
            seed=args.seed + 1, cfg=cfg, n_workers=args.n_workers,
            subset_size=0, max_n_loads=max_n,
        ),
        "test": generate_split(
            "test", args.n_test, TRAIN_PAD_PATTERNS,
            seed=args.seed + 2, cfg=cfg, n_workers=args.n_workers,
            subset_size=0, max_n_loads=max_n,
        ),
    }

    ood = {}
    for pp in OOD_PAD_PATTERNS:
        ood[pp] = generate_split(
            f"ood_{pp}", args.n_ood, (pp,),
            seed=args.seed + 1000, cfg=cfg, n_workers=args.n_workers,
            subset_size=0, max_n_loads=max_n,
        )

    sweeps: dict[str, dict[str, dict]] = {}
    if args.sweep_points > 0:
        sweep_axes = list(GLOBAL_RANGES.names) + list(PER_LOAD_RANGES.names)
        print(f"[sweeps] {len(sweep_axes)} axes × {len(TRAIN_PAD_PATTERNS)} patterns × "
              f"{args.sweep_points} points")
        for axis in sweep_axes:
            sweeps[axis] = {}
            for pattern in TRAIN_PAD_PATTERNS:
                sweeps[axis][pattern] = generate_sweep(
                    axis, pattern, args.sweep_points,
                    seed=args.seed + 7000, cfg=cfg, n_workers=args.n_workers,
                    max_n_loads=max_n,
                )

    write_dataset(args.out, bulk, ood, sweeps, cfg, seed=args.seed)


if __name__ == "__main__":
    main()
