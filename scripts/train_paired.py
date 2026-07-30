"""Train on controlled-variation PAIRS, with an explicit change loss.

    L = L_fwd(yhat, y) + L_fwd(yhat', y') + lambda * L_change(dyhat, dy)

Targets are PER LOAD, not just worst droop: a modification can change which
load is worst, and per-load supervision is a smoother, more informative
signal.

The change is matched in log space, ``d = log10(y') - log10(y)``. That is a
relative change, naturally scaled across four orders of magnitude of droop,
directly available as ``pred' - pred`` from a log-space model, and sign-
preserving (log is monotone) so change-sign accuracy means what it says.

The third term is implied by the first two only if they were exact. It is
weighted explicitly because the absolute terms are dominated by the droop
LEVEL, while the change is a small residual on top — and the change is the
thing a proposer consumes.

Why this dataset and not the bulk one: in the bulk data no two samples share
a base circuit (minimum 260 of 260 differing edges at (13,13)) and there are
0.6-2.3 samples per wire dimension, so per-edge sensitivity is
unidentifiable. See scripts/build_paired_dataset.py.

Factor cost note: the dense factor path makes a full cache ~5 h for this
dataset, but the four cheapest anchors are 3.9 min. Pilot there first —
identifiability is answerable on small grids.

    python scripts/train_paired.py --anchors 3,7 7,7 5,13 13,13 \\
        --score dynamic_kernel --local-rc --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import spearmanr

from eason.impedance_attention_model import ImpAttnConfig, ImpedanceAttentionRegressor
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (branch_system, dc_symmetric_factor,
                                     impedance_factors, knob_tensors,
                                     local_rc_features, node_features)
from tools.pyg_dataset import LOG_FLOOR
from tools.sampler import (FIXED_R_VIA, FIXED_RSHEET_BOT, FIXED_RSHEET_TOP)

DT, FDT = torch.float32, torch.float64


def default_omegas(n_freq):
    from tools.sampler import FIXED_FREQ
    b = 2 * np.pi * FIXED_FREQ
    return torch.tensor([0.0, b, 5 * b, 25 * b, 125 * b][:n_freq], dtype=FDT)


def load_pairs(h5_path, split, anchors=None):
    """-> list of dicts, one per row, plus base->perturbation index map."""
    with h5py.File(h5_path, "r") as f:
        g = f[f"pairs/{split}"]
        fams = json.loads(f.attrs["families"])
        d = {k: g[k][:] for k in g.keys()}
    n = d["n_top"].shape[0]
    sel = np.ones(n, dtype=bool)
    if anchors:
        sel &= np.array([(int(a), int(b)) in anchors
                         for a, b in zip(d["n_top"], d["n_bot"])])
    rows = [{k: d[k][i] for k in d} for i in np.nonzero(sel)[0]]
    by_base = defaultdict(lambda: {"base": None, "perts": []})
    for r in rows:
        slot = by_base[int(r["base_of"])]
        if int(r["is_base"]):
            slot["base"] = r
        else:
            slot["perts"].append(r)
    groups = [v for v in by_base.values()
              if v["base"] is not None and v["perts"]]
    if not groups:
        raise SystemExit(
            f"no complete base+perturbation groups in split '{split}'"
            + (f" for anchors {sorted(anchors)}" if anchors else "")
            + f" (rows matched: {len(rows)})")
    return groups, fams


class Anchors:
    """Lazy per-anchor topology, and per-row factors + features."""

    def __init__(self, omegas, m, n_power, local_rc):
        self.om, self.m, self.q = omegas, m, n_power
        self.local_rc = local_rc
        self.sys, self.g = {}, {}

    def get(self, a):
        if a not in self.sys:
            gg = build_regular_pdn(n_top=a[0], n_bot=a[1])
            self.g[a] = gg
            self.sys[a] = branch_system(gg)
        return self.g[a], self.sys[a]

    def row(self, r):
        a = (int(r["n_top"]), int(r["n_bot"]))
        g, s_ = self.get(a)
        te, be = g.top_edges.shape[0], g.bot_edges.shape[0]
        nl = int(s_.n_loads)
        wt = torch.tensor(r["ww_top_edges"][:te], dtype=FDT)
        wb = torch.tensor(r["ww_bot_edges"][:be], dtype=FDT)
        cd = torch.tensor(float(r["C_decap"]), dtype=FDT)
        R, C = knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                            FIXED_R_VIA)
        loads = torch.tensor(r["loads"][:nl], dtype=FDT)
        x = node_features(s_, loads)
        if self.local_rc:
            x = torch.cat([x, local_rc_features(s_, R, C)], -1)
        p, s = impedance_factors(s_, R, C, self.om, m=self.m, n_power=self.q)
        fdc = dc_symmetric_factor(s_, R, C, m=self.m, n_power=self.q)
        y = np.maximum(r["peak_droop_loads"][:nl], LOG_FLOOR)
        return dict(a=a, n_elec=s_.n_elec, x=x.to(DT), p=p.to(DT), s=s.to(DT),
                    fdc=fdc.to(DT), logy=torch.tensor(np.log10(y), dtype=DT))


def build_cache(groups, ac, tag, cache_dir):
    key = f"{tag}_m{ac.m}_q{ac.q}_f{ac.om.numel()}_rc{int(ac.local_rc)}"
    path = cache_dir / f"{key}.pt" if cache_dir else None
    if path is not None and path.exists():
        print(f"  cache <- {path}")
        return torch.load(path, weights_only=False)
    out, t0, n = [], time.time(), 0
    for gi, grp in enumerate(groups):
        b = ac.row(grp["base"])
        ps = [ac.row(r) for r in grp["perts"]]
        out.append((b, ps, [int(r["family"]) for r in grp["perts"]]))
        n += 1 + len(ps)
        if (gi + 1) % 25 == 0:
            el = time.time() - t0
            print(f"    {gi+1}/{len(groups)} groups, {n} rows "
                  f"({el:.0f}s, {el/n*1e3:.0f} ms/row)")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
        print(f"  cache -> {path}")
    return out


def fwd(model, r, dev):
    return model(r["x"].to(dev), r["p"].to(dev), r["s"].to(dev), r["n_elec"],
                 fdc=r["fdc"].to(dev))


def evaluate(model, cache, fams, dev):
    """Per-family change metrics plus absolute per-load accuracy."""
    model.eval()
    per_fam = defaultdict(lambda: {"sign": [], "rho": [], "mae": [],
                                   "base_sign": []})
    abs_err, abs_tot = [], []
    with torch.no_grad():
        for b, ps, fcs in cache:
            pb = fwd(model, b, dev).cpu()
            abs_err.append((10 ** pb - 10 ** b["logy"]).abs().mean())
            abs_tot.append((10 ** b["logy"]).mean())
            for r, fc in zip(ps, fcs):
                pr = fwd(model, r, dev).cpu()
                dp = (pr - pb).numpy()
                dt = (r["logy"] - b["logy"]).numpy()
                live = np.abs(dt) > 1e-6
                if live.sum() < 3:
                    continue
                nm = fams[fc]
                per_fam[nm]["sign"].append(
                    float((np.sign(dp[live]) == np.sign(dt[live])).mean()))
                # majority-class baseline for THIS pair
                frac = float((dt[live] > 0).mean())
                per_fam[nm]["base_sign"].append(max(frac, 1 - frac))
                per_fam[nm]["rho"].append(
                    float(spearmanr(dp[live], dt[live]).statistic))
                per_fam[nm]["mae"].append(float(np.abs(dp - dt).mean()))
                abs_err.append((10 ** pr - 10 ** r["logy"]).abs().mean())
                abs_tot.append((10 ** r["logy"]).mean())
    rel = float(torch.stack(abs_err).sum() / torch.stack(abs_tot).sum())
    out = {"abs_relerr": rel}
    for nm, d in per_fam.items():
        out[nm] = {k: (float(np.nanmean(v)) if v else float("nan"))
                   for k, v in d.items()}
        out[nm]["n"] = len(d["sign"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("datasets/paired_v1/pairs.h5"))
    ap.add_argument("--anchors", nargs="*", default=["3,7", "7,7", "5,13", "13,13"])
    ap.add_argument("--val-anchors", nargs="*", default=None)
    ap.add_argument("--score", default="dynamic_kernel",
                    choices=["bilinear", "kernel", "dynamic_kernel", "simple"])
    ap.add_argument("--local-rc", action="store_true")
    ap.add_argument("--lambda-delta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--m-factor", type=int, default=16)
    ap.add_argument("--n-freq", type=int, default=3)
    ap.add_argument("--n-power", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/paired.pt"))
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    anc = {tuple(int(v) for v in a.split(",")) for a in args.anchors}
    vanc = ({tuple(int(v) for v in a.split(",")) for a in args.val_anchors}
            if args.val_anchors else None)
    om = default_omegas(args.n_freq)
    cache_dir = args.cache_dir or args.data.parent / "_pfactors"

    tr_g, fams = load_pairs(args.data, "train", anc)
    va_g, _ = load_pairs(args.data, "val", vanc)
    print(f"train groups {len(tr_g)}  val groups {len(va_g)}  families {fams}")
    ac = Anchors(om, args.m_factor, args.n_power, args.local_rc)
    tag = "tr_" + "-".join(f"{a}x{b}" for a, b in sorted(anc))
    print("factors (train)")
    tr = build_cache(tr_g, ac, tag, cache_dir)
    print("factors (val)")
    va = build_cache(va_g, ac, "va_all" if vanc is None else
                     "va_" + "-".join(f"{a}x{b}" for a, b in sorted(vanc)),
                     cache_dir)

    cfg = ImpAttnConfig(score=args.score, n_freq=args.n_freq,
                        m_factor=args.m_factor, local_rc=args.local_rc)
    init = float(np.mean([r[0]["logy"].mean() for r in tr]))
    model = ImpedanceAttentionRegressor(cfg, init_bias=init).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"score={args.score} params {sum(p.numel() for p in model.parameters()):,} "
          f"init_bias {init:.3f}  lambda_delta {args.lambda_delta}")

    hist, best = [], -np.inf
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        lf, lc = [], []
        for gi in rng.permutation(len(tr)):
            b, ps, _ = tr[gi]
            pb = fwd(model, b, args.device)
            loss = F.mse_loss(pb, b["logy"].to(args.device))
            nfw = 1
            for r in ps:
                pr = fwd(model, r, args.device)
                loss = loss + F.mse_loss(pr, r["logy"].to(args.device))
                dch = F.mse_loss(pr - pb,
                                 (r["logy"] - b["logy"]).to(args.device))
                loss = loss + args.lambda_delta * dch
                lc.append(float(dch.detach())); nfw += 1
            loss = loss / nfw
            opt.zero_grad(); loss.backward(); opt.step()
            lf.append(float(loss.detach()))
        sched.step()
        rep = evaluate(model, va, fams, args.device)
        wire = [k for k in rep if k.startswith("ww")]
        ws = float(np.nanmean([rep[k]["sign"] for k in wire])) if wire else float("nan")
        wb_ = float(np.nanmean([rep[k]["base_sign"] for k in wire])) if wire else float("nan")
        wr = float(np.nanmean([rep[k]["rho"] for k in wire])) if wire else float("nan")
        print(f"ep{ep:>3} loss {np.mean(lf):.4f} dloss {np.mean(lc):.5f} | "
              f"val relerr {rep['abs_relerr']:.4f} | WIRE sign {ws:.3f} "
              f"(base {wb_:.3f}, lift {ws-wb_:+.3f}) rho {wr:+.3f} "
              f"({time.time()-t0:.0f}s)")
        hist.append({"epoch": ep, "loss": float(np.mean(lf)), "val": rep,
                     "wire_sign": ws, "wire_base": wb_, "wire_rho": wr})
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        blob = {"model": model.state_dict(), "epoch": ep, "cfg": vars(cfg),
                "args": {**vars(args), "ckpt": str(args.ckpt),
                         "data": str(args.data)}}
        torch.save(blob, args.ckpt.with_suffix(".last.pt"))
        score = ws - wb_ if np.isfinite(ws - wb_) else -np.inf
        if score > best:
            best = score; torch.save(blob, args.ckpt)

    print(f"\nselected by WIRE sign lift over the majority baseline: {best:+.3f}")
    print("\nper-family on val, final epoch:")
    rep = hist[-1]["val"]
    print(f"{'family':>14} {'n':>5} {'sign':>7} {'baseline':>9} {'lift':>7} "
          f"{'rho':>7} {'|d| mae':>8}")
    for k in sorted(rep):
        if k == "abs_relerr":
            continue
        v = rep[k]
        print(f"{k:>14} {v['n']:>5} {v['sign']:>7.3f} {v['base_sign']:>9.3f} "
              f"{v['sign']-v['base_sign']:>+7.3f} {v['rho']:>+7.3f} "
              f"{v['mae']:>8.4f}")
    hp = args.ckpt.with_suffix(".history.json")
    hp.write_text(json.dumps(hist, indent=2))
    print(f"history -> {hp}")


if __name__ == "__main__":
    main()
