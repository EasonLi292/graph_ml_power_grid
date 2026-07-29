"""Sensitivity gate: is the surrogate's ∂droop/∂knob trustworthy enough
for gradient-based repair?

The repair loop consumes derivatives, not absolute droop. This gate
measures, against re-simulation, the three properties backprop-repair
needs, per anchor (never aggregated — v7 lesson):

  1. SIGN     — sign(model Δ) vs sign(sim Δ) for single-edge width
                changes and global decap changes;
  2. RANKING  — within a design, Spearman across perturbed edges of
                (model Δ, sim Δ): does the model order repair sites
                correctly?
  3. MAGNITUDE — median |model Δ / sim Δ| (step-size calibration; least
                critical, a verifier/line-search absorbs it).

Plus a model-internal linearity check: autograd grad·ΔR vs the model's
own finite difference (is one backward pass a valid proposer?).

Pass bar (per anchor): sign ≥ 95 %, mean site-ranking Spearman ≥ 0.8.

STATISTICAL POWER (measured 2026-07-27, docs/SOBOLEV_RESULTS_REVIEW.md):
designs are the unit of variance, not perturbations — edges within one
design are strongly correlated, so per-edge binomial/Fisher tests
overstate significance badly. Re-sampling designs for a *fixed*
checkpoint gives these spreads over 6 seeds:

    n_designs=3   (7,13): sign spread 0.49, ρ spread 0.58   <- unusable
    n_designs=12  (7,13): sign spread 0.20, ρ spread 0.40

Hence the default is 12, and every number is reported with a CI
(Wilson for sign, design-bootstrap for ρ). Treat any difference smaller
than the reported interval as noise, and compare checkpoints only at
equal --n-designs/--seed.

    python scripts/sensitivity_gate.py --ckpt checkpoints/droop_v7_edgeconv.pt \\
        --conv-type edgeconv --out docs/analysis/sensitivity_gate_edgeconv.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eason import EncoderConfig, PDNDroopRegressor
from tools.dataset_runner import SimConfig, run_one
from tools.grid_construction import EDGE_ATTR_COLS, EDGE_ATTR_DIM, build_regular_pdn, to_hetero_data
from tools.sampler import (
    FIXED_CONSTANTS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    FIXED_R_VIA,
    FIXED_RSHEET_BOT,
    FIXED_RSHEET_TOP,
    GLOBAL_RANGES,
    sample_edge_widths,
)

R_COL = EDGE_ATTR_COLS.index("R")
C_COL = EDGE_ATTR_COLS.index("C")
I_COL = EDGE_ATTR_COLS.index("I_peak")
F_COL = EDGE_ATTR_COLS.index("freq")
D_COL = EDGE_ATTR_COLS.index("duty")
P_COL = EDGE_ATTR_COLS.index("phase")

# |sim Δ|/droop below this: excluded. NOT solver noise — the transient sim
# is deterministic and reruns of an identical input agree to exactly 0.0e+00
# (measured at (4,7), (7,13), (11,31)). The old 1e-4 value was mislabelled
# and was discarding real physics: single-edge response shrinks with grid
# size (median 1.6e-4 at (4,7), 4.9e-6 at (7,13), 2.4e-7 at (11,31)), so on
# the large anchors 1e-4 excluded essentially everything — 0/12 live at
# (11,31) vs 9/12 at 1e-7. The floor exists only to drop edges with NO
# effect on the worst load (exact zeros do occur: an edge far from the
# peak-drooping load), so it belongs just above zero, not at a magnitude
# where real responses live.
SIM_DELTA_FLOOR = 1e-6


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion (better than normal at k≈n)."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def _boot_ci(vals, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap CI over DESIGNS (the true unit of variance)."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def build_batch(g, ww_top, ww_bot, C_decap):
    """HeteroData batch matching pyg_dataset conventions exactly."""
    data = to_hetero_data(g)
    R_top = FIXED_RSHEET_TOP * g.pitch_top / ww_top
    R_bot = FIXED_RSHEET_BOT * g.pitch_bot / ww_bot

    def strap(Rdir):
        a = torch.zeros(Rdir.shape[0], EDGE_ATTR_DIM)
        a[:, R_COL] = Rdir
        return a

    data["mesh_top", "strap", "mesh_top"].edge_attr = strap(torch.cat([R_top, R_top]))
    data["mesh_bot", "strap", "mesh_bot"].edge_attr = strap(torch.cat([R_bot, R_bot]))
    nvia = data["mesh_top", "via", "mesh_bot"].edge_index.size(1)
    rv = torch.full((nvia,), float(FIXED_R_VIA))
    data["mesh_top", "via", "mesh_bot"].edge_attr = strap(rv)
    data["mesh_bot", "via", "mesh_top"].edge_attr = strap(rv.clone())
    dec = torch.zeros(data["mesh_bot", "decap", "mesh_bot"].edge_index.size(1), EDGE_ATTR_DIM)
    dec[:, C_COL] = C_decap
    data["mesh_bot", "decap", "mesh_bot"].edge_attr = dec
    nload = data["mesh_bot", "load", "mesh_bot"].edge_index.size(1)
    ld = torch.zeros(nload, EDGE_ATTR_DIM)
    ld[:, I_COL] = FIXED_I_PEAK; ld[:, F_COL] = FIXED_FREQ
    ld[:, D_COL] = FIXED_DUTY;   ld[:, P_COL] = FIXED_PHASE
    data["mesh_bot", "load", "mesh_bot"].edge_attr = ld
    data["y"] = torch.zeros(g.n_loads, dtype=torch.float32)
    return Batch.from_data_list([data])


class ImpedanceAdapter:
    """Exposes the impedance-attention model through the gate's interface.

    Rebuilds the impedance factors under autograd on every call — that is
    the whole point of the gate, so factor caching is deliberately NOT used
    here even though training uses it.
    """

    def __init__(self, model, omegas, m, n_power, want_fdc: bool = False):
        from tools.impedance_factors import branch_system, node_features
        self.model, self.omegas, self.m, self.n_power = model, omegas, m, n_power
        self.want_fdc = want_fdc
        self._bs, self._nf = branch_system, node_features
        self._cache = {}

    def __call__(self, g, ww_top, ww_bot, C_decap):
        from tools.impedance_factors import (dc_symmetric_factor,
                                             impedance_factors, knob_tensors)
        key = (g.n_top, g.n_bot)
        if key not in self._cache:
            s = self._bs(g)
            self._cache[key] = (s, self._nf(s, torch.tensor(g.loads,
                                                            dtype=torch.float64)))
        s, x = self._cache[key]
        cd = C_decap if torch.is_tensor(C_decap) else torch.tensor(
            float(C_decap), dtype=torch.float64)
        R, C = knob_tensors(g, ww_top.double(), ww_bot.double(), cd.double(),
                            FIXED_RSHEET_TOP, FIXED_RSHEET_BOT, FIXED_R_VIA)
        p, sf = impedance_factors(s, R, C, self.omegas, m=self.m,
                                  n_power=self.n_power)
        # The kernel score needs the symmetric DC factor, rebuilt here under
        # autograd like everything else. Note what this implies: the kernel
        # term is a function of DC effective resistance, so its gradient to
        # C_decap is structurally zero (a capacitor is an open circuit at
        # DC). Decap sensitivity therefore still rides entirely on the
        # bilinear term's AC channels — the kernel can only reweight the
        # resistive picture.
        fdc = (dc_symmetric_factor(s, R, C, m=self.m, n_power=self.n_power)
               if self.want_fdc else None)
        return (10.0 ** self.model(x, p, sf, s.n_elec, fdc=fdc)).max()


def strap_groups(edges: np.ndarray, n: int) -> list[np.ndarray]:
    """Edge indices grouped into column straps (vertical edges only).

    Node index is ``r*n + c``, so an edge (i, j) is vertical iff
    ``j - i == n``, and its column is ``i % n``. One group = one full
    power strap running the height of the die.

    This is the unit a designer actually turns. A single edge is one
    segment of one strap, and on a large die its effect on global worst
    droop is ~1e-7 relative — real, but far below what any surrogate can
    resolve and far below what repair cares about. Widening a whole strap
    is both a legal repair move and a perturbation whose magnitude does
    not vanish as the die grows.
    """
    src, dst = edges[:, 0], edges[:, 1]
    vert = (dst - src) == n
    col = src % n
    groups: dict[int, list[int]] = {}
    for idx in np.nonzero(vert)[0]:
        groups.setdefault(int(col[idx]), []).append(int(idx))
    return [np.asarray(v, dtype=int) for _, v in sorted(groups.items())]


def model_worst(model, g, ww_top, ww_bot, C_decap):
    if isinstance(model, ImpedanceAdapter):
        return model(g, ww_top, ww_bot, C_decap)
    return (10.0 ** model(build_batch(g, ww_top, ww_bot, C_decap))).max()


def sim_worst(n_top, n_bot, ww_top, ww_bot, C_decap, n_loads):
    loads = np.tile(np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                    (n_loads, 1))
    p = dict(FIXED_CONSTANTS)
    p.update(n_top=n_top, n_bot=n_bot, wire_width=0.5,  # placeholder; per-edge wins
             C_decap=C_decap, loads=loads,
             ww_top_edges=np.asarray(ww_top, dtype=np.float64),
             ww_bot_edges=np.asarray(ww_bot, dtype=np.float64))
    return float(run_one(p, cfg=SimConfig())["peak_droop_loads"].max())


def gate_anchor(model, n_top, n_bot, n_designs, k_bot, k_top, delta_ww,
                rng, verbose=True, k_frac=None, unit="edge"):
    from scipy.stats import spearmanr

    g = build_regular_pdn(n_top=n_top, n_bot=n_bot)
    n_tp, n_bp = g.top_edges.shape[0], g.bot_edges.shape[0]
    if k_frac is not None:
        # Test a fixed FRACTION of each layer's edges, so the *coverage* of
        # the ranking estimate is comparable across anchors spanning 79 to
        # 2232 nodes (k_bot=12 is 17 % of (4,7)'s 70 bot edges but 0.8 % of
        # (11,31)'s 1550).
        #
        # This does NOT make individual perturbations stronger — each site
        # is still one edge — so it is not the fix for low live counts;
        # lowering SIM_DELTA_FLOOR is. And it is expensive: cost is
        # O(k x n_designs) simulations AND autograd factor rebuilds, so
        # k_frac=0.15 at (11,31) means 250 sites per design. Use small
        # values or absolute k on the large anchors.
        k_bot = max(1, int(round(k_frac * n_bp)))
        k_top = max(1, int(round(k_frac * n_tp)))
    cd_p = GLOBAL_RANGES.by_name("C_decap")

    rows = []
    for d_i in range(n_designs):
        wt, wb = sample_edge_widths(n_top, rng, n_bot=n_bot)
        cd = float(np.exp(rng.uniform(np.log(cd_p.lo), np.log(cd_p.hi))))
        wt_t, wb_t = torch.from_numpy(wt).float(), torch.from_numpy(wb).float()

        base_sim = sim_worst(n_top, n_bot, wt, wb, cd, g.n_loads)
        # autograd at baseline (for the linearity check)
        wt_g = wt_t.clone().requires_grad_(True)
        wb_g = wb_t.clone().requires_grad_(True)
        w0 = model_worst(model, g, wt_g, wb_g, cd)
        w0.backward()
        grad_top, grad_bot = wt_g.grad.numpy(), wb_g.grad.numpy()
        base_model = float(w0.detach())

        if unit == "strap":
            g_bot, g_top = strap_groups(g.bot_edges, n_bot), strap_groups(g.top_edges, n_top)
            sites = ([("bot", g_bot[i]) for i in
                      rng.choice(len(g_bot), min(k_bot, len(g_bot)), replace=False)]
                     + [("top", g_top[i]) for i in
                        rng.choice(len(g_top), min(k_top, len(g_top)), replace=False)])
        else:
            sites = ([("bot", np.array([int(e)])) for e in
                      rng.choice(n_bp, min(k_bot, n_bp), replace=False)]
                     + [("top", np.array([int(e)])) for e in
                        rng.choice(n_tp, min(k_top, n_tp), replace=False)])
        for tier, idx in sites:
            wt2, wb2 = wt.copy(), wb.copy()
            (wt2 if tier == "top" else wb2)[idx] *= (1.0 + delta_ww)
            with torch.no_grad():
                dm = float(model_worst(model, g, torch.from_numpy(wt2).float(),
                                       torch.from_numpy(wb2).float(), cd)) - base_model
            ds = sim_worst(n_top, n_bot, wt2, wb2, cd, g.n_loads) - base_sim
            gr = (grad_top if tier == "top" else grad_bot)[idx]
            dw = ((wt2 if tier == "top" else wb2)[idx]
                  - (wt if tier == "top" else wb)[idx])
            glin = float(np.dot(gr, dw))
            rows.append(dict(design=d_i, kind=f"ww_{tier}", edge=int(idx[0]),
                             n_edges=int(idx.size),
                             d_model=dm, d_sim=ds, d_lin=glin,
                             base_sim=base_sim))
        for fac in (2.0, 0.5):
            with torch.no_grad():
                dm = float(model_worst(model, g, wt_t, wb_t, cd * fac)) - base_model
            ds = sim_worst(n_top, n_bot, wt, wb, cd * fac, g.n_loads) - base_sim
            rows.append(dict(design=d_i, kind=f"decap_x{fac}", edge=-1,
                             d_model=dm, d_sim=ds, d_lin=float("nan"),
                             base_sim=base_sim))

    # ---- metrics ----
    ww = [r for r in rows if r["kind"].startswith("ww")]
    dec = [r for r in rows if r["kind"].startswith("decap")]
    live = [r for r in ww if abs(r["d_sim"]) / r["base_sim"] > SIM_DELTA_FLOOR]
    sign_ok = [np.sign(r["d_model"]) == np.sign(r["d_sim"]) for r in live]
    dec_live = [r for r in dec if abs(r["d_sim"]) / r["base_sim"] > SIM_DELTA_FLOOR]
    dec_ok = [np.sign(r["d_model"]) == np.sign(r["d_sim"]) for r in dec_live]
    rhos, lin_errs = [], []
    for d_i in range(n_designs):
        sub = [r for r in ww if r["design"] == d_i]
        rhos.append(float(spearmanr([r["d_model"] for r in sub],
                                    [r["d_sim"] for r in sub]).statistic))
        fd = np.array([r["d_model"] for r in sub])
        ln = np.array([r["d_lin"] for r in sub])
        lin_errs.append(float(np.median(np.abs(ln - fd) / (np.abs(fd) + 1e-12))))
    ratios = [abs(r["d_model"]) / abs(r["d_sim"]) for r in live]
    # Null baseline the gate never reported: what a CONSTANT sign predictor
    # scores. Widening is not reliably good -- measured base rate 0.44-0.54
    # across anchors -- so the majority class is only ~0.54 accurate. A
    # learned derivative must beat that to have earned anything.
    helps = [r["d_sim"] < 0 for r in live]
    base_rate = float(np.mean(helps)) if helps else float("nan")
    majority = max(base_rate, 1.0 - base_rate) if helps else float("nan")
    sign_lo, sign_hi = _wilson(int(np.sum(sign_ok)), len(sign_ok)) if sign_ok else (None, None)
    rho_lo, rho_hi = _boot_ci(rhos)
    out = {
        "anchor": [n_top, n_bot],
        "n_designs": n_designs,
        "n_ww_perturb": len(ww), "n_ww_live": len(live),
        "perturb_unit": unit,
        "ww_sign_acc": float(np.mean(sign_ok)) if sign_ok else None,
        "ww_sign_ci95": [sign_lo, sign_hi],
        "ww_sign_base_rate_helps": base_rate,
        "ww_sign_majority_baseline": majority,
        "ww_sign_lift_over_majority": (float(np.mean(sign_ok)) - majority
                                       if sign_ok else None),
        "site_rank_spearman_mean": float(np.mean(rhos)),
        "site_rank_spearman_ci95": [rho_lo, rho_hi],
        "site_rank_spearman_per_design": rhos,
        "magnitude_ratio_median": float(np.median(ratios)) if ratios else None,
        "decap_sign_acc": float(np.mean(dec_ok)) if dec_ok else None,
        "n_decap_live": len(dec_live),
        "autograd_linearity_medrelerr": float(np.median(lin_errs)),
    }
    if verbose:
        print(f"  ({n_top},{n_bot}): sign {out['ww_sign_acc']:.3f} "
              f"[{sign_lo:.2f},{sign_hi:.2f}], "
              f"rank-rho {out['site_rank_spearman_mean']:+.3f} "
              f"[{rho_lo:+.2f},{rho_hi:+.2f}], "
              f"mag-ratio {out['magnitude_ratio_median']:.2f}, "
              f"decap-sign {out['decap_sign_acc']:.2f}, "
              f"lin-err {out['autograd_linearity_medrelerr']:.1%}, "
              f"live {len(live)}/{len(ww)}")
        print(f"      unit={unit}  majority-class baseline {majority:.3f} "
              f"(base rate helps {base_rate:.3f}) -> lift "
              f"{out['ww_sign_lift_over_majority']:+.3f}")
        if len(live) < 30:
            print(f"      ! only {len(live)} live perturbations — most {unit} "
                  f"moves fall under the {SIM_DELTA_FLOOR:.0e} relative floor here; "
                  f"raise --n-designs or --delta-ww before trusting this row")
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/droop_v7_edgeconv.pt"))
    ap.add_argument("--arch", default="local", choices=["local", "impedance"],
                    help="'local' = the depth-based GNN baselines; "
                         "'impedance' = one-shot impedance attention")
    ap.add_argument("--conv-type", default="edgeconv",
                    choices=["admittance", "edgeconv", "edge_aware"])
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--anchors", nargs="*", default=["3,7", "4,7", "7,13"],
                    help="train (3,7); OOD interp (4,7); OOD transfer (7,13)")
    ap.add_argument("--n-designs", type=int, default=12,
                    help="designs per anchor — the unit of variance; 3 is "
                         "unusable (rho spread 0.58 at (7,13)), 12 is the "
                         "cheap default (~25 s for 3 anchors)")
    ap.add_argument("--k-bot", type=int, default=12, help="bot strap edges perturbed")
    ap.add_argument("--k-top", type=int, default=6, help="top strap edges perturbed")
    ap.add_argument("--perturb-unit", default="edge", choices=["edge", "strap"],
                    help="'edge' = one strap segment (historical unit); "
                         "'strap' = a full column strap, the unit a designer "
                         "actually turns. Single-edge effect on global worst "
                         "droop vanishes as the die grows (median 1.6e-4 at "
                         "(4,7) but 2.4e-7 at (11,31)), so on large grids the "
                         "edge unit measures physics no surrogate can resolve.")
    ap.add_argument("--k-frac", type=float, default=None,
                    help="test this FRACTION of each layer's edges instead "
                         "of the absolute --k-bot/--k-top, so ranking "
                         "COVERAGE is comparable across anchors. It does "
                         "not strengthen individual perturbations and is "
                         "not the fix for low live counts (that is "
                         "SIM_DELTA_FLOOR). Cost is O(k x n_designs) sims "
                         "and factor rebuilds -- 0.15 at (11,31) is 250 "
                         "sites per design.")
    ap.add_argument("--delta-ww", type=float, default=0.25, help="width step (+25%%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.arch == "impedance":
        from eason.impedance_attention_model import (
            ImpAttnConfig, ImpedanceAttentionRegressor)
        from scripts.train_impedance_attention import default_omegas
        cfgd = dict(state.get("cfg", {}))
        # checkpoints trained before the invariant-channel back-port have no
        # such key and were built on the raw channels; defaulting them to
        # True would silently mismatch the encoder/phi/psi widths
        cfgd.setdefault("invariant", False)
        cfg = ImpAttnConfig(**cfgd)
        inner = ImpedanceAttentionRegressor(cfg).to(torch.float64)
        inner.load_state_dict(state["model"]); inner.eval()
        for p in inner.parameters():
            p.requires_grad = False
        sa = state.get("args", {})
        model = ImpedanceAdapter(
            inner, default_omegas(cfg.n_freq).double(),
            cfg.m_factor, int(sa.get("n_power", 2)),
            want_fdc=getattr(cfg, "score", "bilinear") == "kernel")
    else:
        model = PDNDroopRegressor(
            EncoderConfig(hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                          conv_type=args.conv_type, drop_edge_p=0.0),
            target_space="log")
        model.load_state_dict(state["model"]); model.eval()
        for p in model.parameters():
            p.requires_grad = False

    rng = np.random.default_rng(args.seed)
    print(f"gate: {args.ckpt.name} ({args.conv_type}) | "
          f"{args.n_designs} designs x ({args.k_bot}+{args.k_top}) edges x "
          f"+{args.delta_ww:.0%} width, decap x2/x0.5")
    t0 = time.time()
    results = []
    for a in args.anchors:
        nt, nb = (int(v) for v in a.split(","))
        res, _ = gate_anchor(model, nt, nb, args.n_designs, args.k_bot,
                             args.k_top, args.delta_ww, rng,
                             k_frac=args.k_frac, unit=args.perturb_unit)
        results.append(res)

    print(f"\n{'anchor':>8} | {'sign (95% CI)':>20} | {'rank-rho (95% CI)':>24} | "
          f"{'mag':>5} | verdict")
    for r in results:
        ok = ((r["ww_sign_ci95"][0] or 0) >= 0.95
              and r["site_rank_spearman_ci95"][0] >= 0.8)
        sl, sh = r["ww_sign_ci95"]
        rl, rh = r["site_rank_spearman_ci95"]
        print(f"  {tuple(r['anchor'])!s:>7} | {r['ww_sign_acc']:.2f} "
              f"[{sl:.2f},{sh:.2f}] | {r['site_rank_spearman_mean']:+.3f} "
              f"[{rl:+.2f},{rh:+.2f}] | {r['magnitude_ratio_median']:>5.2f} | "
              f"{'PASS' if ok else 'FAIL'}")
    print("  (PASS requires the CI LOWER bound to clear the bar)")
    print(f"({time.time()-t0:.0f}s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"ckpt": str(args.ckpt), "conv_type": args.conv_type,
             "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
             "delta_ww": args.delta_ww, "n_designs": args.n_designs,
             "k_bot": args.k_bot, "k_top": args.k_top,
            "k_frac": args.k_frac, "seed": args.seed,
             "results": results}, indent=2))
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
