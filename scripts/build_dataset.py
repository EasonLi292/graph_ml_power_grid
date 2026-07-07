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
from tools.grid_construction import BOT_COL_PATTERNS, build_regular_pdn
from tools.sampler import (
    ALL_ANCHORS,
    FIXED_CONSTANTS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    GLOBAL_RANGES,
    TEST_ANCHORS,
    TRAIN_ANCHORS,
    axis_sweep,
    sample_anchor,
    sample_edge_widths,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


# Per-anchor topology counts. ``n_loads`` is invariant across n_top
# *within* a die size (loads sit on M_bot), but differs across n_bot —
# so droop arrays and per-edge widths are stored NaN-padded to the max
# over ALL_ANCHORS and the loader reads the first ``n_loads(anchor)`` /
# ``edges(anchor)`` columns of each row.
_N_LOADS_BY_ANCHOR: dict[tuple[int, int], int] = {}
_TOP_EDGE_COUNTS: dict[tuple[int, int], int] = {}
_BOT_EDGE_COUNTS: dict[tuple[int, int], int] = {}
for _a in ALL_ANCHORS:
    _g = build_regular_pdn(n_top=_a[0], n_bot=_a[1])
    _N_LOADS_BY_ANCHOR[_a] = int(_g.n_loads)
    _TOP_EDGE_COUNTS[_a] = int(_g.top_edges.shape[0])
    _BOT_EDGE_COUNTS[_a] = int(_g.bot_edges.shape[0])
MAX_N_LOADS:   int = max(_N_LOADS_BY_ANCHOR.values())
MAX_TOP_EDGES: int = max(_TOP_EDGE_COUNTS.values())
MAX_BOT_EDGES: int = max(_BOT_EDGE_COUNTS.values())


# ---------------------------------------------------------------------------
# Per-sample assembly
# ---------------------------------------------------------------------------

def _assemble_sample_dicts(
    global_samples: np.ndarray,
    anchors: np.ndarray,
    edge_widths: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[dict]:
    """Build the per-sample param dict ``run_one`` consumes.

    ``anchors`` is ``[n, 2]`` of (n_top, n_bot). When ``edge_widths`` is
    given (per-edge dataset), each sample also carries ``ww_top_edges`` /
    ``ww_bot_edges`` so the solver stamps a heterogeneous per-edge R
    instead of a uniform ``wire_width``.
    """
    out = []
    name_idx = {n: i for i, n in enumerate(GLOBAL_RANGES.names)}
    for i in range(global_samples.shape[0]):
        nt, nb = int(anchors[i, 0]), int(anchors[i, 1])
        loads = np.tile(
            np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]], dtype=np.float64),
            (_N_LOADS_BY_ANCHOR[(nt, nb)], 1),
        )
        d: dict = dict(FIXED_CONSTANTS)
        d["wire_width"] = float(global_samples[i, name_idx["wire_width"]])
        d["C_decap"]    = float(global_samples[i, name_idx["C_decap"]])
        d["n_top"]      = nt
        d["n_bot"]      = nb
        d["loads"]      = loads
        if edge_widths is not None:
            wt, wb = edge_widths[i]
            d["ww_top_edges"] = wt
            d["ww_bot_edges"] = wb
        out.append(d)
    return out


def _pack_edge_widths(
    edge_widths: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Pad per-edge widths into rectangular arrays for HDF5.

    Both layers are NaN-padded to their max across ALL_ANCHORS; the
    loader reads only the first ``edges(anchor)`` columns of each row.
    """
    n = len(edge_widths)
    top = np.full((n, MAX_TOP_EDGES), np.nan, dtype=np.float32)
    bot = np.full((n, MAX_BOT_EDGES), np.nan, dtype=np.float32)
    for i, (wt, wb) in enumerate(edge_widths):
        top[i, : wt.shape[0]] = wt
        bot[i, : wb.shape[0]] = wb
    return {"ww_top_edges": top, "ww_bot_edges": bot}


def _pad_loads(rows: list[np.ndarray]) -> np.ndarray:
    """Stack per-sample load vectors, NaN-padding to MAX_N_LOADS."""
    out = np.full((len(rows), MAX_N_LOADS), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, : r.shape[0]] = r
    return out


def _stack_results(results: list[dict]) -> dict[str, np.ndarray]:
    return {
        "peak_droop_loads":   _pad_loads([r["peak_droop_loads"]   for r in results]),
        "static_droop_loads": _pad_loads([r["static_droop_loads"] for r in results]),
        "worst_load_idx":     np.array([r["worst_load_idx"]     for r in results], dtype=np.int32),
        "worst_load_droop":   np.array([r["worst_load_droop"]   for r in results], dtype=np.float32),
    }


def _collect_traj_subset(results: list[dict], subset_idx: list[int]) -> dict | None:
    kept = [results[i] for i in subset_idx if "V_loads_full" in results[i]]
    if not kept:
        return None
    # V_loads_full is time-major [T, n_loads]. Both axes vary per sample
    # (T with the RC-dependent warmup, n_loads with the anchor) — NaN-pad
    # both and store each sample's time vector alongside.
    T_max = max(r["V_loads_full"].shape[0] for r in kept)
    V = np.full((len(kept), T_max, MAX_N_LOADS), np.nan, dtype=np.float32)
    t = np.full((len(kept), T_max), np.nan, dtype=np.float32)
    for i, r in enumerate(kept):
        Ti, nl = r["V_loads_full"].shape
        V[i, :Ti, :nl] = r["V_loads_full"]
        t[i, :Ti] = r["t_full"]
    return {
        "indices": np.array(subset_idx, dtype=np.int32),
        "V_loads": V,
        "t":       t,
    }


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_split(
    name: str,
    n: int,
    anchor_choices: tuple[tuple[int, int], ...],
    seed: int,
    cfg: SimConfig,
    n_workers: int,
    subset_size: int = 0,
    per_edge: bool = False,
) -> dict:
    mode = "per-edge R" if per_edge else "uniform R"
    print(f"[{name}] sampling {n} points (seed={seed}) over anchors={anchor_choices} [{mode}]...")
    global_samples = GLOBAL_RANGES.lhs(n, seed=seed)
    anchors = sample_anchor(n, seed=seed + 100_000, choices=anchor_choices)

    edge_widths = None
    if per_edge:
        rng = np.random.default_rng(seed + 200_000)
        edge_widths = [
            sample_edge_widths(int(nt), rng, n_bot=int(nb)) for nt, nb in anchors
        ]

    sample_dicts = _assemble_sample_dicts(global_samples, anchors, edge_widths)
    subset_idx = list(range(min(subset_size, n))) if subset_size > 0 else []

    print(f"[{name}] simulating on {n_workers or 'all'} workers...")
    t0 = time.time()
    results = run_many(sample_dicts, keep_traj_idx=set(subset_idx), cfg=cfg, n_workers=n_workers)
    dt = time.time() - t0
    print(f"[{name}] done in {dt:.1f}s ({1000 * dt / max(n, 1):.1f} ms/sample)")

    payload = {
        "global_params": global_samples.astype(np.float32),
        "n_top":         anchors[:, 0].astype(np.int16),
        "n_bot":         anchors[:, 1].astype(np.int16),
        "results":       _stack_results(results),
        "V_subset":      _collect_traj_subset(results, subset_idx) if subset_idx else None,
    }
    if per_edge:
        payload.update(_pack_edge_widths(edge_widths))
    return payload


def generate_sweep(
    axis: str,
    anchor: tuple[int, int],
    n_points: int,
    cfg: SimConfig,
    n_workers: int,
) -> dict:
    axis_vals, medians = axis_sweep(axis, n_points)

    global_samples = np.zeros((n_points, GLOBAL_RANGES.d), dtype=np.float64)
    for j, name in enumerate(GLOBAL_RANGES.names):
        global_samples[:, j] = medians[name]
    global_samples[:, GLOBAL_RANGES.names.index(axis)] = axis_vals

    anchors = np.tile(np.asarray(anchor, dtype=np.int16), (n_points, 1))
    sample_dicts = _assemble_sample_dicts(global_samples, anchors)

    t0 = time.time()
    results = run_many(sample_dicts, cfg=cfg, n_workers=n_workers)
    dt = time.time() - t0
    print(f"  sweep[{axis}|anchor={anchor}] {n_points}pts in {dt:.1f}s")

    return {
        "axis_values":   axis_vals.astype(np.float32),
        "global_params": global_samples.astype(np.float32),
        "n_top":         anchors[:, 0],
        "n_bot":         anchors[:, 1],
        "results":       _stack_results(results),
    }


# ---------------------------------------------------------------------------
# H5 writer
# ---------------------------------------------------------------------------

def _write_split(grp: h5py.Group, payload: dict) -> None:
    grp.create_dataset("global_params", data=payload["global_params"], compression="gzip")
    grp.create_dataset("n_top",         data=payload["n_top"],         compression="gzip")
    grp.create_dataset("n_bot",         data=payload["n_bot"],         compression="gzip")
    for k, v in payload["results"].items():
        grp.create_dataset(k, data=v, compression="gzip")
    for k in ("ww_top_edges", "ww_bot_edges"):
        if k in payload:
            grp.create_dataset(k, data=payload[k], compression="gzip")
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
    per_edge: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["version"]    = 6
        f.attrs["seed"]       = seed
        f.attrs["created_at"] = datetime.now().isoformat()
        f.attrs["git_sha"]    = _git_sha()
        f.attrs["per_edge"]   = bool(per_edge)
        f.attrs["max_top_edges"] = int(MAX_TOP_EDGES)
        f.attrs["max_bot_edges"] = int(MAX_BOT_EDGES)
        f.attrs["max_n_loads"]   = int(MAX_N_LOADS)

        f.attrs["global_param_names"] = json.dumps(list(GLOBAL_RANGES.names))
        f.attrs["train_anchors"]      = json.dumps([list(a) for a in TRAIN_ANCHORS])
        f.attrs["test_anchors"]       = json.dumps([list(a) for a in TEST_ANCHORS])
        f.attrs["bot_col_patterns"]   = json.dumps(
            {str(nb): list(pat) for nb, pat in BOT_COL_PATTERNS.items()}
        )
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
        for nt, nb in ALL_ANCHORS:
            g_t = build_regular_pdn(n_top=nt, n_bot=nb)
            topo[f"{nt}_{nb}"] = {
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
            for axis, by_anchor in sweeps.items():
                ax_grp = sweep_grp.create_group(axis)
                for (nt, nb), payload in by_anchor.items():
                    _write_split(ax_grp.create_group(f"anchor_{nt}_{nb}"), payload)

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
    ap.add_argument("--per-edge", action="store_true",
                    help="Heterogeneous per-strap-edge wire width (decorrelates "
                         "R from topology; needed for per-edge sensitivity).")
    args = ap.parse_args()

    cfg = SimConfig()

    bulk = {
        "train": generate_split(
            "train", args.n_train, TRAIN_ANCHORS,
            seed=args.seed, cfg=cfg, n_workers=args.n_workers,
            subset_size=args.subset_size, per_edge=args.per_edge,
        ),
        "val": generate_split(
            "val", args.n_val, TRAIN_ANCHORS,
            seed=args.seed + 1, cfg=cfg, n_workers=args.n_workers,
            per_edge=args.per_edge,
        ),
        "test": generate_split(
            "test", args.n_test, TEST_ANCHORS,
            seed=args.seed + 2, cfg=cfg, n_workers=args.n_workers,
            per_edge=args.per_edge,
        ),
    }

    sweeps: dict[str, dict[tuple[int, int], dict]] = {}
    if args.sweep_points > 0:
        sweep_axes = list(GLOBAL_RANGES.names)
        print(f"[sweeps] {len(sweep_axes)} axes × {len(ALL_ANCHORS)} anchors × "
              f"{args.sweep_points} points")
        for axis in sweep_axes:
            sweeps[axis] = {}
            for anchor in ALL_ANCHORS:
                sweeps[axis][anchor] = generate_sweep(
                    axis, anchor, args.sweep_points,
                    cfg=cfg, n_workers=args.n_workers,
                )

    write_dataset(args.out, bulk, sweeps, cfg, seed=args.seed, per_edge=args.per_edge)


if __name__ == "__main__":
    main()
