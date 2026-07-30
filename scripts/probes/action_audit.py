"""Do the weak action families contain a learnable signal at all?

The pilot got wire-change ranking to rho +0.42..+0.86 but decap to +0.05 and
load-frequency to +0.09. Four explanations were on the table:

    (a) not enough action diversity
    (b) targets with almost no distinguishable relative variation
    (c) inadequate frequency coverage
    (d) the model is simply failing

This measures the ground truth ONLY -- the simulator and the circuit, never
the model -- so (a)-(c) can be separated from (d) before any more training
compute is spent. If the target itself carries no rankable information, no
architecture will find it, and generating 20k more pairs of it is waste.

Part A: per-family target statistics
------------------------------------
    |d| abs     mean per-load absolute change
    |d| log     mean per-load change in log space
    disp        coefficient of variation of the RELATIVE change across
                loads. This is the crux: if every load moves by the same
                relative amount, log-space ranking is ranking noise, and a
                low Spearman is the correct answer rather than a failure.
    ties        share of load PAIRS whose log-space changes differ by less
                than 1e-6 -- the float32 precision the simulator stores
                ``peak_droop_loads`` in, which is the real resolution floor
    stab        mean Spearman between the |log-space change| rankings induced
                by two different magnitudes of the SAME action on the SAME
                circuit. Unstable ranking means the target is noise, not
                signal. Taken on the MAGNITUDE of the change, not its signed
                value: magnitudes straddle 1.0, so the effect sign flips
                between "more decap" and "less decap" and a signed
                correlation would average a consistent +1 and -1 to zero and
                report perfectly good physics as noise.
    sign        majority-sign baseline (linear change)
    rho y0      Spearman of the change against the BASE droop, in linear and
                in log space. High linear + low log is the signature of a
                target that only looks learnable: it means "rank by which
                load already droops most" already solves the linear version,
                which the model knows from the forward task alone.

Part B: frequency coverage
--------------------------
The impedance factors are built on a FIXED omega grid derived from the
constant ``FIXED_FREQ``, not from each sample's load frequency. So a
``load_freq`` action leaves every impedance feature bit-identical and moves
exactly one scalar node feature. Part B verifies that directly, then asks
whether the information is recoverable: it compares the ranking induced by
the driving-point impedance at the MODIFIED frequency (the oracle feature)
against the one at the sampled grid frequency (what the model actually has).

    python scripts/probes/action_audit.py --anchors 7,13 13,13 --n-bases 12
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.stats import spearmanr

from tools.dataset_runner import SimConfig, run_many
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (admittance, branch_system,
                                     impedance_factors, knob_tensors)
from tools.pdn_actions import ActionSpace
from tools.sampler import (FIXED_CONSTANTS, FIXED_DUTY, FIXED_FREQ,
                           FIXED_I_PEAK, FIXED_PHASE, FIXED_R_VIA,
                           FIXED_RSHEET_BOT, FIXED_RSHEET_TOP, GLOBAL_RANGES)

DT = torch.float64
TIE_TOL = 1e-6          # float32 storage precision of peak_droop_loads
MAGS = (0.25, 0.5, 2.0, 4.0)
FRACS = (0.25, 0.5, 0.9)


def base_circuit(nt, nb, rng, decap_fill=0.6):
    g0 = build_regular_pdn(n_top=nt, n_bot=nb)
    ip = float(np.exp(rng.uniform(np.log(FIXED_I_PEAK * 0.5),
                                  np.log(FIXED_I_PEAK * 2.0))))
    loads = np.tile([[ip, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]], (g0.n_loads, 1))
    ww = GLOBAL_RANGES.by_name("wire_width")
    cd = GLOBAL_RANGES.by_name("C_decap")
    wt = rng.uniform(ww.lo, ww.hi, g0.top_edges.shape[0])
    wb = rng.uniform(ww.lo, ww.hi, g0.bot_edges.shape[0])
    c0 = float(np.exp(rng.uniform(np.log(cd.lo), np.log(cd.hi))))
    cs = np.full(g0.n_decaps, c0)
    cs[rng.choice(g0.n_decaps, int(g0.n_decaps * (1 - decap_fill)),
                  replace=False)] = 0.0
    g = build_regular_pdn(n_top=nt, n_bot=nb, Rsheet_top=FIXED_RSHEET_TOP,
                          Rsheet_bot=FIXED_RSHEET_BOT, wire_width=0.5,
                          R_via=FIXED_R_VIA, C_decap=c0, freq=FIXED_FREQ,
                          loads=loads, ww_top_edges=wt, ww_bot_edges=wb,
                          C_decap_sites=cs)
    sys_ = branch_system(g)
    acts = ActionSpace(g, sys_, torch.tensor(wt, dtype=DT),
                       torch.tensor(wb, dtype=DT), torch.tensor(cs, dtype=DT),
                       torch.tensor(loads, dtype=DT), FIXED_RSHEET_TOP,
                       FIXED_RSHEET_BOT, FIXED_R_VIA)
    return g, sys_, acts


def spec_of(nt, nb, cand):
    p = dict(FIXED_CONSTANTS)
    p.update(n_top=int(nt), n_bot=int(nb), wire_width=0.5,
             C_decap=float(max(cand.C_sites.max(), 1e-15)),
             loads=cand.loads.numpy().astype(float),
             ww_top_edges=cand.wt.numpy().astype(float),
             ww_bot_edges=cand.wb.numpy().astype(float),
             C_decap_sites=cand.C_sites.numpy().astype(float))
    return p


def build_actions(acts, rng):
    """The capacitor and load-frequency families the audit compares."""
    full = np.nonzero(acts.C_sites.numpy() > 0)[0]
    empty = np.nonzero(acts.C_sites.numpy() == 0)[0]
    if full.size < 4 or empty.size < 2:
        return []
    half = full.size // 2
    out = []
    for m in MAGS:
        out.append(("decap_global", m, acts.decap_global(m)))
        out.append(("decap_resize_one", m,
                    acts.decap_resize(int(rng.choice(full)), m)))
    for f in FRACS:
        out.append(("decap_redistribute", f,
                    acts.decap_redistribute(full[:half], full[half:], f)))
    # placement changes. add/move get several sizes so they have a magnitude
    # axis and therefore a stability number; remove has only one meaning, so
    # its stability stays nan by construction rather than by oversight.
    v = float(acts.C_sites[full].max())
    site_add, site_rm = int(rng.choice(empty)), int(rng.choice(full))
    site_mv = int(rng.choice(empty))
    src_mv = int(rng.choice(full))
    for m in (0.5, 1.0, 2.0):
        out.append(("decap_add_one", m,
                    acts.decap_add(site_add, torch.tensor(v * m, dtype=DT))))
    out.append(("decap_remove_one", 1.0, acts.decap_remove(site_rm)))
    out.append(("decap_move_one", 1.0, acts.decap_move(src_mv, site_mv)))
    for m in (0.5, 0.75, 1.5, 2.0):
        out.append(("load_freq", m, acts.load_freq(m)))
    return out


def stats_for(y0, y1):
    """Every target statistic for one (base, perturbed) pair."""
    y0 = np.asarray(y0, float); y1 = np.asarray(y1, float)
    d = y1 - y0
    dlog = np.log10(np.maximum(y1, 1e-300) / np.maximum(y0, 1e-300))
    rel = y1 / np.maximum(y0, 1e-300) - 1.0
    mrel = np.abs(rel).mean()
    disp = float(rel.std() / mrel) if mrel > 0 else 0.0
    pairs = list(combinations(range(len(y0)), 2))
    ties = float(np.mean([abs(dlog[i] - dlog[j]) < TIE_TOL for i, j in pairs])) \
        if pairs else 1.0
    sgn = np.sign(d)
    maj = float(max((sgn == v).mean() for v in (-1, 0, 1)))
    def sp(a, b):
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return 0.0
        r = spearmanr(a, b).statistic
        return 0.0 if not np.isfinite(r) else float(r)
    return dict(abs=float(np.abs(d).mean()), log=float(np.abs(dlog).mean()),
                disp=disp, ties=ties, sign=maj,
                rho_y0_lin=abs(sp(d, y0)), rho_y0_log=abs(sp(dlog, y0)),
                dlog=dlog)


def part_a(anchors, n_bases, seed, workers):
    print("\nPART A -- simulator ground truth, per action family")
    rows = defaultdict(list)
    stab = defaultdict(list)
    for nt, nb in anchors:
        rng = np.random.default_rng(seed + 1000 * nt + nb)
        specs, tags = [], []
        for b in range(n_bases):
            g, sys_, acts = base_circuit(nt, nb, rng)
            base_cand = acts.decap_global(1.0)          # identity
            specs.append(spec_of(nt, nb, base_cand))
            tags.append(("base", b, None, None))
            for fam, mag, cand in build_actions(acts, rng):
                specs.append(spec_of(nt, nb, cand))
                tags.append((fam, b, mag, None))
        print(f"  ({nt},{nb}): {len(specs)} simulations...", flush=True)
        res = run_many(specs, cfg=SimConfig(), n_workers=workers, chunksize=8)
        ys = [np.asarray(r["peak_droop_loads"], float) for r in res]
        base_y = {b: y for (f, b, m, _), y in zip(tags, ys) if f == "base"}
        per_fam_base = defaultdict(dict)
        for (fam, b, mag, _), y in zip(tags, ys):
            if fam == "base":
                continue
            st = stats_for(base_y[b], y)
            rows[(nt, nb, fam)].append(st)
            per_fam_base[(fam, b)][mag] = st["dlog"]
        for (fam, b), by_mag in per_fam_base.items():
            ms = sorted(by_mag)
            for m1, m2 in combinations(ms, 2):
                a, c = np.abs(by_mag[m1]), np.abs(by_mag[m2])
                if np.ptp(a) and np.ptp(c):
                    r = spearmanr(a, c).statistic
                    if np.isfinite(r):
                        stab[(nt, nb, fam)].append(float(r))

    print(f"\n{'anchor':>8} {'family':>20} {'n':>4} {'|d| abs':>9} {'|d| log':>9} "
          f"{'disp':>7} {'ties':>7} {'stab':>7} {'sign':>6} "
          f"{'rho y0 lin':>11} {'rho y0 log':>11}")
    out = []
    for (nt, nb, fam), sts in sorted(rows.items()):
        g = lambda k: float(np.mean([s[k] for s in sts]))
        sv = stab[(nt, nb, fam)]
        s_ = float(np.mean(sv)) if sv else float("nan")
        print(f"{f'({nt},{nb})':>8} {fam:>20} {len(sts):>4} {g('abs'):>9.2e} "
              f"{g('log'):>9.2e} {g('disp'):>7.3f} {g('ties'):>7.3f} "
              f"{s_:>7.3f} {g('sign'):>6.3f} {g('rho_y0_lin'):>11.3f} "
              f"{g('rho_y0_log'):>11.3f}")
        out.append(dict(anchor=[nt, nb], family=fam, n=len(sts),
                        abs=g("abs"), log=g("log"), disp=g("disp"),
                        ties=g("ties"), stability=s_, sign_majority=g("sign"),
                        rho_base_linear=g("rho_y0_lin"),
                        rho_base_log=g("rho_y0_log")))
    return out


# ------------------------------------------------------------------ part B

def driving_impedance(sys_, R, C, omega):
    """|Z| seen between each load's two terminals at one frequency."""
    Y = admittance(sys_, R, C, torch.tensor(float(omega), dtype=DT))
    a = sys_.free_of[sys_.load_terminals[:, 0]]
    b = sys_.free_of[sys_.load_terminals[:, 1]]
    n = sys_.n_free
    rhs = torch.zeros((n, a.shape[0]), dtype=Y.dtype)
    k = torch.arange(a.shape[0])
    m = a >= 0
    rhs[a[m], k[m]] += 1.0
    m = b >= 0
    rhs[b[m], k[m]] -= 1.0
    V = torch.linalg.solve(Y, rhs)
    z = (V[a.clamp_min(0), k] * (a >= 0) - V[b.clamp_min(0), k] * (b >= 0))
    return z.abs().numpy()


def part_b(anchors, n_bases, seed, workers):
    print("\nPART B -- frequency coverage for the load_freq family")
    w0 = 2 * np.pi * FIXED_FREQ
    grid = torch.tensor([0.0, w0, 5 * w0], dtype=DT)

    nt, nb = anchors[0]
    rng = np.random.default_rng(seed)
    g, sys_, acts = base_circuit(nt, nb, rng)

    # 1. the factors literally do not move when the load frequency changes
    cand = acts.load_freq(2.0)
    p0, s0 = impedance_factors(sys_, acts.R, acts.C, grid, m=16, n_power=2)
    p1, s1 = impedance_factors(sys_, cand.R, cand.C, grid, m=16, n_power=2)
    same = float((p0 - p1).abs().max() + (s0 - s1).abs().max())
    print(f"  factor difference between a base and its load_freq x2 "
          f"perturbation: {same:.3e}")
    print(f"  -> the omega grid is fixed at FIXED_FREQ, so a load-frequency")
    print(f"     action moves ONE scalar node feature and nothing else.")

    # 2. does the response actually depend on frequency over the acted range?
    print(f"\n  driving-point |Z| at the loads, relative to its value at w0:")
    z0 = driving_impedance(sys_, acts.R, acts.C, w0)
    print(f"  {'w / w0':>8} {'mean |Z|/|Z(w0)|':>17} {'spread across loads':>21}")
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0, 5.0):
        z = driving_impedance(sys_, acts.R, acts.C, mult * w0)
        r = z / z0
        print(f"  {mult:>8.2f} {r.mean():>17.4f} {r.std() / r.mean():>21.4f}")

    # 3. oracle vs available feature: can the SAMPLED grid rank the response?
    print(f"\n  ranking the simulated load_freq change (n={n_bases} bases x "
          f"4 magnitudes):")
    specs, tags = [], []
    circuits = []
    for b in range(n_bases):
        gg, ss, aa = base_circuit(nt, nb, rng)
        circuits.append((gg, ss, aa))
        specs.append(spec_of(nt, nb, aa.decap_global(1.0)))
        tags.append((b, None))
        for m in (0.5, 0.75, 1.5, 2.0):
            specs.append(spec_of(nt, nb, aa.load_freq(m)))
            tags.append((b, m))
    res = run_many(specs, cfg=SimConfig(), n_workers=workers, chunksize=8)
    ys = [np.asarray(r["peak_droop_loads"], float) for r in res]
    ybase = {b: y for (b, m), y in zip(tags, ys) if m is None}
    rho_or, rho_fx, rho_y0 = [], [], []
    for (b, m), y in zip(tags, ys):
        if m is None:
            continue
        gg, ss, aa = circuits[b]
        y0 = ybase[b]
        dlog = np.log10(np.maximum(y, 1e-300) / np.maximum(y0, 1e-300))
        if np.ptp(dlog) == 0:
            continue
        z_or = driving_impedance(ss, aa.R, aa.C, m * w0)      # true frequency
        z_fx = driving_impedance(ss, aa.R, aa.C, w0)          # sampled grid
        f = lambda v: abs(float(spearmanr(v, dlog).statistic))
        rho_or.append(f(z_or)); rho_fx.append(f(z_fx)); rho_y0.append(f(y0))
    print(f"    |Z| at the MODIFIED frequency  (oracle) : rho {np.mean(rho_or):.3f}")
    print(f"    |Z| at the SAMPLED grid w0     (actual) : rho {np.mean(rho_fx):.3f}")
    print(f"    base droop y0                (baseline) : rho {np.mean(rho_y0):.3f}")
    return dict(factor_diff_under_load_freq=same,
                rho_oracle_freq=float(np.mean(rho_or)),
                rho_sampled_grid=float(np.mean(rho_fx)),
                rho_base_droop=float(np.mean(rho_y0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", nargs="*", default=["7,13", "13,13"])
    ap.add_argument("--n-bases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/analysis/action_audit.json"))
    args = ap.parse_args()
    anchors = [tuple(int(v) for v in a.split(",")) for a in args.anchors]

    a = part_a(anchors, args.n_bases, args.seed, args.n_workers)
    b = part_b(anchors, args.n_bases, args.seed, args.n_workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"part_a": a, "part_b": b}, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
