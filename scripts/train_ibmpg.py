"""Fine-tune / train the per-node droop model on the IBM transient grids.

Leave-one-grid-out: train on all benchmarks except ``--holdout``, then
evaluate per-node metrics on the held-out grid (every grid node gets a
prediction via the seed-tiling patches).

Usage:
    python3.12 scripts/train_ibmpg.py --holdout ibmpg2t \\
        --init checkpoints/droop_v6_peredge_edgeconv.pt \\
        --ckpt checkpoints/ibmpg_node_transfer.pt
    python3.12 scripts/train_ibmpg.py --holdout ibmpg2t \\
        --ckpt checkpoints/ibmpg_node_scratch.pt          # from scratch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_geometric.loader import DataLoader

from tools.ibmpg_patches import (
    DROOP_FLOOR,
    IBMPGPatches,
    compute_stats,
    load_graph,
)
from eason.node_model import IBMNodeRegressor, transfer_from_synthetic

ALL_BENCHES = ["ibmpg1t", "ibmpg2t", "ibmpg3t", "ibmpg4t", "ibmpg5t", "ibmpg6t"]


class _Concat:
    def __init__(self, parts):
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(p) for p in parts])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        j = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.parts[j][i - self.offsets[j]]


def _masked_metrics(pred_log: np.ndarray, true_log: np.ndarray) -> dict:
    """Per-node metrics in linear volts + ranking quality."""
    from scipy.stats import spearmanr

    p = 10.0 ** pred_log
    t = 10.0 ** true_log
    err = p - t
    ss_res = float((err ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    k = max(1, int(0.01 * t.size))
    hot_true = set(np.argpartition(-t, k)[:k].tolist())
    hot_pred = set(np.argpartition(-p, k)[:k].tolist())
    return {
        "n_nodes": int(t.size),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-30),
        "mae_mV": float(np.abs(err).mean() * 1e3),
        "p99_abs_err_mV": float(np.percentile(np.abs(err), 99) * 1e3),
        "rel_mae": float((np.abs(err) / t).mean()),
        "spearman": float(spearmanr(p, t).statistic),
        "hotspot_recall_1pct": len(hot_true & hot_pred) / k,
    }


@torch.no_grad()
def eval_patches(model, ds, device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Predict every seed node once; returns (pred_log, true_log)."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size)
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        m = batch["node"].loss_mask
        p = out[m]
        t = batch["node"].y[m]
        if hasattr(batch["node"], "sd"):        # residual mode: back to
            sd = batch["node"].sd[m]            # absolute log droop
            p = p + sd
            t = t + sd
        preds.append(p.cpu().numpy())
        trues.append(t.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def val_score(model, val_parts, device, batch_size: int) -> dict:
    """Within-grid mean Spearman (shape) + pooled log-space metrics.

    Model selection must reward *within-grid* ranking — pooled linear R²
    is dominated by cross-grid scale and rewards predicting each grid's
    mean, which is exactly the collapse mode to avoid.
    """
    from scipy.stats import spearmanr

    rhos, log_maes = [], []
    for part in val_parts:
        p, t = eval_patches(model, part, device, batch_size)
        rhos.append(float(spearmanr(p, t).statistic))
        log_maes.append(float(np.abs(p - t).mean()))
    return {
        "spearman_within_grid": float(np.mean(rhos)),
        "spearman_per_grid": rhos,
        "log_mae": float(np.mean(log_maes)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ibmpg2t")
    ap.add_argument("--init", type=Path, default=None,
                    help="synthetic edgeconv ckpt to transfer the conv stack from")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/ibmpg_node.pt"))
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patches-per-graph", type=int, default=3000,
                    help="random patch subset per graph per epoch")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 3e-4 with --init, 1e-3 from scratch")
    ap.add_argument("--residual", action="store_true",
                    help="predict log(droop) - log(static drop) instead of "
                         "log(droop); metrics still in absolute droop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    lr = args.lr or (3e-4 if args.init else 1e-3)
    train_benches = [b for b in ALL_BENCHES if b != args.holdout]

    print(f"loading train graphs: {train_benches}")
    t0 = time.time()
    graphs = {b: load_graph(b) for b in train_benches}
    stats = compute_stats(list(graphs.values()))
    print(f"  loaded + stats in {time.time()-t0:.0f}s | stats: "
          + ", ".join(f"{k}: {stats.mu[k]:.2f}±{stats.sigma[k]:.2f}" for k in stats.mu))

    t0 = time.time()
    train_parts, val_parts = [], []
    rng = np.random.default_rng(args.seed)
    for b, g in graphs.items():
        ds = IBMPGPatches(g, stats, shuffle_seed=args.seed, residual=args.residual)
        # fixed 5% of patches per graph become validation
        n_val = max(1, len(ds) // 20)
        val_idx = set(rng.choice(len(ds), n_val, replace=False).tolist())
        tr = [ds.patches[i] for i in range(len(ds)) if i not in val_idx]
        va = [ds.patches[i] for i in range(len(ds)) if i in val_idx]
        ds_tr = IBMPGPatches.__new__(IBMPGPatches)
        ds_tr.__dict__ = {**ds.__dict__, "patches": tr}
        ds_va = IBMPGPatches.__new__(IBMPGPatches)
        ds_va.__dict__ = {**ds.__dict__, "patches": va}
        train_parts.append(ds_tr)
        val_parts.append(ds_va)
        print(f"  {b}: {len(tr)} train / {len(va)} val patches")
    train_ds, val_ds = _Concat(train_parts), _Concat(val_parts)
    print(f"patch tiling in {time.time()-t0:.0f}s | total {len(train_ds)} train patches")

    model = IBMNodeRegressor(args.hidden_dim, args.n_layers).to(args.device)
    if args.init:
        n = transfer_from_synthetic(model, str(args.init))
        print(f"transferred {n} conv tensors from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | lr {lr}")

    best_rho, history = -np.inf, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = np.concatenate([
            train_ds.offsets[i] + rng.choice(
                len(p), min(args.patches_per_graph, len(p)), replace=False)
            for i, p in enumerate(train_ds.parts)
        ])
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            sampler=idx.tolist())
        t0, total, n_obs = time.time(), 0.0, 0
        for batch in loader:
            batch = batch.to(args.device)
            pred = model(batch)
            m = batch["node"].loss_mask
            loss = F.mse_loss(pred[m], batch["node"].y[m])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * int(m.sum()); n_obs += int(m.sum())
        met = val_score(model, val_parts, args.device, args.batch_size)
        history.append({"epoch": epoch, "train_loss": total / max(n_obs, 1), **met})
        print(f"ep{epoch:>3} loss {total/max(n_obs,1):.4f} | "
              f"val within-grid spearman {met['spearman_within_grid']:.4f} "
              f"log-mae {met['log_mae']:.4f} ({time.time()-t0:.0f}s)")
        if met["spearman_within_grid"] > best_rho:
            best_rho = met["spearman_within_grid"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "stats": {"mu": stats.mu, "sigma": stats.sigma},
                        "args": vars(args) | {"init": str(args.init)}}, args.ckpt)

    # ----- held-out grid -----
    print(f"\nevaluating best ckpt on held-out {args.holdout} ...")
    model.load_state_dict(
        torch.load(args.ckpt, map_location=args.device, weights_only=False)["model"])
    g_test = load_graph(args.holdout)
    ds_test = IBMPGPatches(g_test, stats, shuffle_seed=123, residual=args.residual)
    p, t = eval_patches(model, ds_test, args.device, args.batch_size)
    # masked predictions come out in local (= sorted-global-id) order per
    # patch, so sort each patch's seed ids to align.
    seeds_all = np.concatenate([np.sort(s) for s, _ in ds_test.patches])
    vdd = g_test.x_raw["net_vdd"][seeds_all]

    # physics baseline on the same nodes: predict droop = static IR drop
    from tools.ibmpg_patches import _static_drop
    sd = np.log10(np.maximum(_static_drop(g_test), DROOP_FLOOR))[seeds_all]

    report = {}
    for tag, m in (("all", np.ones_like(vdd)), ("vdd", vdd), ("gnd", ~vdd)):
        m = m.astype(bool)
        report[tag] = _masked_metrics(p[m], t[m])
        report[f"baseline_{tag}"] = _masked_metrics(sd[m], t[m])

    print(f"\n== held-out {args.holdout} (per-node) ==")
    print(json.dumps(report, indent=2))
    hist_path = args.ckpt.with_suffix(".history.json")
    hist_path.write_text(json.dumps({"history": history, "test": report}, indent=2))
    print(f"history → {hist_path}")


if __name__ == "__main__":
    main()
