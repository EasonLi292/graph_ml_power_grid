"""Stage-0 attention diagnostic on IBM grids — FULL-GRAPH training (GPU).

Tests the load-bearing hypothesis of the attention architecture: does
exact DC impedance geometry (sketch) + superposition attention close the
within-grid ranking gap that local patch models can't?

Full-graph steps (one grid per optimizer step, no patches). Residual
target (log droop − log static drop) by default, so the static-IR
baseline's ranking is the floor.

    python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t \\
        --device cuda --epochs 60 --ckpt checkpoints/ibmpg_attn.pt

Nothing here launches automatically — see docs/GPU_HANDOFF.md.
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

from torch_geometric.data import HeteroData

from tools.ibmpg_patches import (
    DROOP_FLOOR,
    LOG_EPS,
    _node_features,
    _static_drop,
    compute_stats,
    load_graph,
)
from tools.impedance_sketch import load_or_compute_sketch
from eason.attention_model import IBMAttnRegressor
from scripts.train_ibmpg import ALL_BENCHES, _masked_metrics


def build_full_data(bench, g, stats, m_sketch, residual, device):
    r, floating = load_or_compute_sketch(bench, g, m=m_sketch)
    data = HeteroData()
    x = _node_features(g, stats)
    data["node"].x = torch.from_numpy(x)
    data["node"].sketch = torch.from_numpy(r)
    sd_log = np.log10(np.maximum(_static_drop(g), DROOP_FLOOR)).astype(np.float32)
    y_abs = np.log10(np.maximum(g.droop, DROOP_FLOOR)).astype(np.float32)
    data["node"].y = torch.from_numpy(y_abs - sd_log if residual else y_abs)
    data["node"].sd = torch.from_numpy(sd_log)
    data["node"].load_mask = torch.from_numpy(g.x_raw["load_I"] > 0)
    data["node"].grid_mask = torch.from_numpy(np.asarray(g.is_grid))
    data["node"].net_vdd = torch.from_numpy(np.asarray(g.x_raw["net_vdd"]))
    for rel, stat in (("R", "logR"), ("C", "logC"), ("L", "logL")):
        ei, val = g.edges[rel]
        z = stats.z(stat, np.log10(val.astype(np.float64) + LOG_EPS)).astype(np.float32)
        src = np.concatenate([ei[0], ei[1]])
        dst = np.concatenate([ei[1], ei[0]])
        zz = np.concatenate([z, z])[:, None]
        et = ("node", rel, "node")
        data[et].edge_index = torch.from_numpy(np.stack([src, dst]))
        data[et].edge_attr = torch.from_numpy(zz)
    return data.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ibmpg2t")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/ibmpg_attn.pt"))
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--m-sketch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--train-benches", nargs="*", default=None,
                    help="subset of train grids (default: all except holdout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    residual = not args.no_residual

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_benches = args.train_benches or [b for b in ALL_BENCHES if b != args.holdout]

    print(f"loading {train_benches} (holdout {args.holdout}) ...")
    t0 = time.time()
    graphs = {b: load_graph(b) for b in train_benches}
    stats = compute_stats(list(graphs.values()))
    datas = {}
    for b, g in graphs.items():
        t1 = time.time()
        datas[b] = build_full_data(b, g, stats, args.m_sketch, residual, args.device)
        print(f"  {b}: {g.n_nodes} nodes, sketch+tensors in {time.time()-t1:.0f}s")
    print(f"total setup {time.time()-t0:.0f}s")

    # fixed 5% val node mask per grid (grid nodes only)
    val_masks = {}
    for b, g in graphs.items():
        gm = np.flatnonzero(g.is_grid)
        vm = np.zeros(g.n_nodes, dtype=bool)
        vm[rng.choice(gm, max(1, gm.size // 20), replace=False)] = True
        val_masks[b] = torch.from_numpy(vm).to(args.device)

    model = IBMAttnRegressor(args.hidden_dim, args.n_conv, args.heads).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | "
          f"residual={residual} | device={args.device}")

    from scipy.stats import spearmanr
    best_rho, history = -np.inf, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for b in rng.permutation(train_benches):
            data = datas[b]
            m = data["node"].grid_mask & ~val_masks[b]
            pred = model(data)
            loss = F.mse_loss(pred[m], data["node"].y[m])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        # val: within-grid spearman on ABSOLUTE droop
        model.eval()
        rhos = []
        with torch.no_grad():
            for b in train_benches:
                data = datas[b]
                m = val_masks[b]
                p = model(data)[m]
                t = data["node"].y[m]
                if residual:
                    p = p + data["node"].sd[m]
                    t = t + data["node"].sd[m]
                rhos.append(float(spearmanr(p.cpu().numpy(), t.cpu().numpy()).statistic))
        rho = float(np.mean(rhos))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_spearman_within_grid": rho})
        print(f"ep{epoch:>3} loss {np.mean(losses):.4f} | val within-grid "
              f"spearman {rho:.4f} ({time.time()-t0:.0f}s)")
        if rho > best_rho:
            best_rho = rho
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "stats": {"mu": stats.mu, "sigma": stats.sigma},
                        "args": {**vars(args), "ckpt": str(args.ckpt),
                                 "train_benches": train_benches}}, args.ckpt)

    # ----- held-out grid -----
    print(f"\nevaluating best ckpt on {args.holdout} ...")
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device,
                                     weights_only=False)["model"])
    model.eval()
    g_test = load_graph(args.holdout)
    data = build_full_data(args.holdout, g_test, stats, args.m_sketch,
                           residual, args.device)
    with torch.no_grad():
        p = model(data)
        if residual:
            p = p + data["node"].sd
        t = torch.from_numpy(
            np.log10(np.maximum(g_test.droop, DROOP_FLOOR))).to(args.device)
    gm = data["node"].grid_mask.cpu().numpy()
    vdd = data["node"].net_vdd.cpu().numpy()
    p = p.cpu().numpy(); t = t.cpu().numpy()
    sd = data["node"].sd.cpu().numpy()

    report = {}
    for tag, m in (("all", gm), ("vdd", gm & vdd), ("gnd", gm & ~vdd)):
        report[tag] = _masked_metrics(p[m], t[m])
        report[f"baseline_{tag}"] = _masked_metrics(sd[m], t[m])
    print(json.dumps(report, indent=2))
    hist_path = args.ckpt.with_suffix(".history.json")
    hist_path.write_text(json.dumps({"history": history, "test": report}, indent=2))
    print(f"history → {hist_path}")


if __name__ == "__main__":
    main()
