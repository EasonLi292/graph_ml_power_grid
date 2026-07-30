"""Build a CONTROLLED-VARIATION paired dataset: same base circuit, one change.

Why this exists. The v8 bulk dataset draws every per-edge width
independently, so **no two samples share a base circuit** — at (13,13) the
minimum number of differing edges between any two samples is 260 of 260. And
the sample budget is 0.60–2.3 samples per wire dimension at the larger
anchors. Per-edge sensitivity is therefore not merely hard to learn from it,
it is *unidentifiable*: no architecture or loss can recover 2000 partial
derivatives from 1209 points that vary in all 2000 coordinates at once.

That is why the trained model predicts a decap change almost perfectly
(sign 1.000, rank rho 0.95 — one global scalar, ~1200 samples) and a wire
change at barely above chance (sign 0.58-0.77 vs a ~0.54 majority baseline).
See docs/analysis/factor_reuse_*.json.

So: for each base circuit, apply ONE parameter change, hold everything else
fixed, and simulate both. That is a controlled derivative measurement.

    G  -> simulator -> y      (per-load droop)
    G' -> simulator -> y'     (one parameter changed)

Both are stored as full circuits — the modification is represented by the
resulting circuit, never by an abstract action label. Perturbations are
stored as a sparse delta against their base to keep the file small; the
loader reconstructs G'.

Coverage is tracked and reported across topology, size, family, effect sign,
effect magnitude and modification locality. Splits are assigned by whole
topology family so no modified version of a base can leak across splits.

    python scripts/build_paired_dataset.py --out datasets/paired_v1/pairs.h5 \\
        --bases-per-anchor 150 --n-workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dataset_runner import SimConfig, run_many
from tools.grid_construction import build_regular_pdn
from tools.sampler import (FIXED_CONSTANTS, FIXED_DUTY, FIXED_FREQ,
                           FIXED_I_PEAK, FIXED_PHASE, GLOBAL_RANGES,
                           TEST_ANCHORS, TRAIN_ANCHORS, sample_edge_widths)

# Held-out die size for validation: training keeps the largest grids (25) so
# size extrapolation survives, selection happens on an unseen middle size.
VAL_NBOT = 19
FAMILIES = ("ww_edge", "ww_section", "ww_strap", "ww_multi", "ww_top_strap",
            "decap_global", "load_one", "load_freq")
# both directions, several magnitudes; kept away from 1.0 so the change is real
MAGS = (0.5, 0.75, 0.9, 1.1, 1.33, 2.0)


def column_groups(edges: np.ndarray, n: int) -> list[np.ndarray]:
    """Vertical-edge indices grouped by column (one full strap per group)."""
    src, dst = edges[:, 0], edges[:, 1]
    vert = (dst - src) == n
    col = src % n
    out = []
    for c in range(n):
        idx = np.nonzero(vert & (col == c))[0]
        if idx.size:
            out.append(idx)
    return out


def make_perturbation(rng, g, nt, nb, wt, wb, cd, loads, family, mag):
    """Return (wt', wb', cd', loads', touched_layer, touched_idx) or None."""
    wt2, wb2, cd2, ld2 = wt.copy(), wb.copy(), float(cd), loads.copy()
    layer, touched = -1, np.zeros(0, dtype=np.int32)
    if family == "ww_edge":
        j = int(rng.integers(wb2.size)); wb2[j] *= mag
        layer, touched = 1, np.array([j], dtype=np.int32)
    elif family == "ww_section":
        grp = column_groups(g.bot_edges, nb)
        if not grp:
            return None
        col = grp[int(rng.integers(len(grp)))]
        k = max(2, col.size // 3)
        st = int(rng.integers(max(1, col.size - k + 1)))
        idx = col[st:st + k]
        wb2[idx] *= mag
        layer, touched = 1, idx.astype(np.int32)
    elif family == "ww_strap":
        grp = column_groups(g.bot_edges, nb)
        if not grp:
            return None
        idx = grp[int(rng.integers(len(grp)))]
        wb2[idx] *= mag
        layer, touched = 1, idx.astype(np.int32)
    elif family == "ww_multi":
        grp = column_groups(g.bot_edges, nb)
        if len(grp) < 3:
            return None
        pick = rng.choice(len(grp), size=3, replace=False)
        idx = np.concatenate([grp[p] for p in pick])
        wb2[idx] *= mag
        layer, touched = 1, idx.astype(np.int32)
    elif family == "ww_top_strap":
        grp = column_groups(g.top_edges, nt)
        if not grp:
            return None
        idx = grp[int(rng.integers(len(grp)))]
        wt2[idx] *= mag
        layer, touched = 0, idx.astype(np.int32)
    elif family == "decap_global":
        cd2 = float(cd) * mag
    elif family == "load_one":
        j = int(rng.integers(ld2.shape[0])); ld2[j, 0] *= mag
        layer, touched = 2, np.array([j], dtype=np.int32)
    elif family == "load_freq":
        ld2[:, 1] *= mag
    else:
        raise ValueError(family)
    # keep widths inside the band the analytic normaliser assumes
    lo, hi = GLOBAL_RANGES.by_name("wire_width").lo, GLOBAL_RANGES.by_name("wire_width").hi
    if wt2.min() < lo * 0.5 or wt2.max() > hi * 2 or \
       wb2.min() < lo * 0.5 or wb2.max() > hi * 2:
        return None
    return wt2, wb2, cd2, ld2, layer, touched


def sim_spec(nt, nb, wt, wb, cd, loads):
    p = dict(FIXED_CONSTANTS)
    p.update(n_top=int(nt), n_bot=int(nb), wire_width=0.5, C_decap=float(cd),
             loads=np.asarray(loads, dtype=float),
             ww_top_edges=np.asarray(wt, dtype=float),
             ww_bot_edges=np.asarray(wb, dtype=float))
    return p


def locality(family, layer, touched, g, nb):
    """Crude local/distant label for COVERAGE ONLY (never a model input)."""
    if family in ("decap_global", "load_freq"):
        return "global"
    if layer == 1 and touched.size:
        frac = touched.size / max(g.bot_edges.shape[0], 1)
        return "local" if frac < 0.05 else ("regional" if frac < 0.25 else "global")
    if layer == 0:
        return "regional"
    return "local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/paired_v1/pairs.h5"))
    ap.add_argument("--bases-per-anchor", type=int, default=150)
    ap.add_argument("--perturbs-per-base", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-nearzero-frac", type=float, default=0.15,
                    help="cap the share of pairs whose simulated worst-droop "
                         "change is under 1e-4 relative — some are kept for "
                         "calibration, they must not dominate")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ww_p = GLOBAL_RANGES.by_name("wire_width")
    cd_p = GLOBAL_RANGES.by_name("C_decap")
    anchors = sorted(set(TRAIN_ANCHORS) | set(TEST_ANCHORS))

    def split_of(a):
        if a in TEST_ANCHORS:
            return "test"
        return "val" if a[1] == VAL_NBOT else "train"

    print(f"anchors {len(anchors)}; splits by whole topology family")
    for a in anchors:
        print(f"   {str(a):>9} -> {split_of(a)}")

    # ---- build the work list: one base sim + K perturbation sims per base
    specs, meta = [], []
    t0 = time.time()
    for a in anchors:
        nt, nb = a
        g0 = build_regular_pdn(n_top=nt, n_bot=nb)
        nl = g0.n_loads
        # test anchors get half as many bases as train/val -- never more,
        # and no absolute floor (a floor silently inverted the split at small
        # --bases-per-anchor)
        nb_bases = (args.bases_per_anchor if split_of(a) != "test"
                    else max(1, args.bases_per_anchor // 2))
        for b in range(nb_bases):
            wt, wb = sample_edge_widths(nt, rng, n_bot=nb)
            cd = float(np.exp(rng.uniform(np.log(cd_p.lo), np.log(cd_p.hi))))
            ip = float(np.exp(rng.uniform(np.log(FIXED_I_PEAK * 0.5),
                                          np.log(FIXED_I_PEAK * 2.0))))
            loads = np.tile(np.array([[ip, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                            (nl, 1))
            g = build_regular_pdn(
                n_top=nt, n_bot=nb, wire_width=0.5, C_decap=cd, loads=loads,
                ww_top_edges=wt, ww_bot_edges=wb)
            bid = len(meta)
            specs.append(sim_spec(nt, nb, wt, wb, cd, loads))
            meta.append(dict(kind="base", anchor=a, split=split_of(a),
                             base_of=bid, family="-", mag=1.0, layer=-1,
                             touched=np.zeros(0, np.int32), loc="-",
                             wt=wt, wb=wb, cd=cd, loads=loads, n_loads=nl))
            # rotate the family order per base so that even with
            # perturbs_per_base < len(FAMILIES) every family gets covered
            # across bases rather than the first few dominating
            fams = (np.arange(len(FAMILIES)) + b) % len(FAMILIES)
            for t in range(args.perturbs_per_base):
                fam = FAMILIES[int(fams[t % len(FAMILIES)])]
                mag = float(MAGS[int(rng.integers(len(MAGS)))])
                out = make_perturbation(rng, g, nt, nb, wt, wb, cd, loads,
                                        fam, mag)
                if out is None:
                    continue
                wt2, wb2, cd2, ld2, layer, touched = out
                specs.append(sim_spec(nt, nb, wt2, wb2, cd2, ld2))
                meta.append(dict(kind="pert", anchor=a, split=split_of(a),
                                 base_of=bid, family=fam, mag=mag, layer=layer,
                                 touched=touched,
                                 loc=locality(fam, layer, touched, g, nb),
                                 wt=wt2, wb=wb2, cd=cd2, loads=ld2, n_loads=nl))
    print(f"\n{len(specs)} simulations "
          f"({sum(m['kind']=='base' for m in meta)} bases, "
          f"{sum(m['kind']=='pert' for m in meta)} perturbations) "
          f"queued in {time.time()-t0:.0f}s")

    print(f"simulating on {args.n_workers} workers...")
    t0 = time.time()
    res = run_many(specs, cfg=SimConfig(), n_workers=args.n_workers,
                   chunksize=8)
    print(f"  done in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/max(len(specs),1)*1e3:.0f} ms/sim)")
    for m, r in zip(meta, res):
        m["y"] = np.asarray(r["peak_droop_loads"], dtype=float)

    # ---- stratify: cap the near-zero-effect share, keep some for calibration
    ybase = {m["base_of"]: m["y"] for m in meta if m["kind"] == "base"}
    keep, dropped_nz = [], 0
    nz_budget = Counter()
    tot = Counter()
    for i, m in enumerate(meta):
        if m["kind"] == "base":
            keep.append(i); continue
        y0, y1 = ybase[m["base_of"]], m["y"]
        rel = abs(y1.max() - y0.max()) / max(y0.max(), 1e-300)
        m["rel_effect"] = rel
        m["sign"] = int(np.sign(y1.max() - y0.max()))
        key = (m["anchor"], m["family"])
        tot[key] += 1
        if rel < 1e-4:
            if nz_budget[key] >= args.max_nearzero_frac * max(tot[key], 1):
                dropped_nz += 1
                continue
            nz_budget[key] += 1
        keep.append(i)
    meta = [meta[i] for i in keep]
    print(f"  dropped {dropped_nz} near-zero-effect pairs to cap their share "
          f"at {args.max_nearzero_frac:.0%} per (anchor, family)")

    # ---- write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    max_l = max(m["n_loads"] for m in meta)
    max_t = max(m["wt"].size for m in meta)
    max_b = max(m["wb"].size for m in meta)
    max_touch = max(1, max(m["touched"].size for m in meta))
    fam_code = {f: i for i, f in enumerate(FAMILIES)}
    with h5py.File(args.out, "w") as f:
        f.attrs["created_at"] = datetime.now().isoformat()
        f.attrs["families"] = json.dumps(list(FAMILIES))
        f.attrs["mags"] = json.dumps(list(MAGS))
        f.attrs["val_nbot"] = VAL_NBOT
        f.attrs["train_anchors"] = json.dumps([list(a) for a in TRAIN_ANCHORS])
        f.attrs["test_anchors"] = json.dumps([list(a) for a in TEST_ANCHORS])
        f.attrs["note"] = ("controlled variation: each perturbation shares its "
                           "base circuit and changes ONE parameter")
        for sp in ("train", "val", "test"):
            rows = [m for m in meta if m["split"] == sp]
            if not rows:
                continue
            grp = f.create_group(f"pairs/{sp}")
            n = len(rows)
            idx_of = {}
            for k, m in enumerate(rows):
                idx_of.setdefault(m["base_of"], None)
            d = lambda name, shape, dt: grp.create_dataset(
                name, shape=shape, dtype=dt, compression="gzip",
                compression_opts=4)
            ds = {
                "n_top": d("n_top", (n,), "i2"), "n_bot": d("n_bot", (n,), "i2"),
                "is_base": d("is_base", (n,), "i1"),
                "base_of": d("base_of", (n,), "i4"),
                "family": d("family", (n,), "i1"),
                "mag": d("mag", (n,), "f4"),
                "layer": d("layer", (n,), "i1"),
                "n_touched": d("n_touched", (n,), "i4"),
                "touched": d("touched", (n, max_touch), "i4"),
                "C_decap": d("C_decap", (n,), "f8"),
                "ww_top_edges": d("ww_top_edges", (n, max_t), "f4"),
                "ww_bot_edges": d("ww_bot_edges", (n, max_b), "f4"),
                "loads": d("loads", (n, max_l, 4), "f4"),
                "peak_droop_loads": d("peak_droop_loads", (n, max_l), "f8"),
                "rel_effect": d("rel_effect", (n,), "f4"),
                "effect_sign": d("effect_sign", (n,), "i1"),
            }
            for k, m in enumerate(rows):
                ds["n_top"][k] = m["anchor"][0]; ds["n_bot"][k] = m["anchor"][1]
                ds["is_base"][k] = 1 if m["kind"] == "base" else 0
                ds["base_of"][k] = m["base_of"]
                ds["family"][k] = -1 if m["kind"] == "base" else fam_code[m["family"]]
                ds["mag"][k] = m["mag"]; ds["layer"][k] = m["layer"]
                ds["n_touched"][k] = m["touched"].size
                if m["touched"].size:
                    ds["touched"][k, :m["touched"].size] = m["touched"]
                ds["C_decap"][k] = m["cd"]
                ds["ww_top_edges"][k, :m["wt"].size] = m["wt"]
                ds["ww_bot_edges"][k, :m["wb"].size] = m["wb"]
                ds["loads"][k, :m["n_loads"]] = m["loads"]
                ds["peak_droop_loads"][k, :m["n_loads"]] = m["y"]
                ds["rel_effect"][k] = m.get("rel_effect", 0.0)
                ds["effect_sign"][k] = m.get("sign", 0)
            print(f"  {sp}: {n} rows "
                  f"({sum(m['kind']=='base' for m in rows)} bases)")
    print(f"\nwrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")

    # ---- coverage report: counts, never just totals
    print("\ncoverage (perturbations only)")
    pert = [m for m in meta if m["kind"] == "pert"]
    for dim, fn in (("split", lambda m: m["split"]),
                    ("family", lambda m: m["family"]),
                    ("locality", lambda m: m["loc"]),
                    ("effect sign", lambda m: {1: "worse", -1: "better",
                                               0: "flat"}[m["sign"]]),
                    ("effect magnitude", lambda m: (
                        "<1e-4" if m["rel_effect"] < 1e-4 else
                        "1e-4..1e-3" if m["rel_effect"] < 1e-3 else
                        "1e-3..1e-2" if m["rel_effect"] < 1e-2 else ">1e-2"))):
        c = Counter(fn(m) for m in pert)
        tot_ = sum(c.values())
        print(f"  {dim:>17}: " + "  ".join(
            f"{k}={v} ({100*v/tot_:.0f}%)" for k, v in sorted(c.items())))
    print("\n  per (anchor, family) minimum count: " + str(min(
        Counter((m["anchor"], m["family"]) for m in pert).values())))


if __name__ == "__main__":
    main()
