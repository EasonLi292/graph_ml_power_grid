"""Stage-1 attention training on IBM grids — FULL-GRAPH (GPU).

Stage-0 postmortem (docs/STAGE1_PLAN.md): the DC sketch carried zero
ranking signal beyond the static baseline, and the real driver of
within-net droop — per-load pulse *timing* — was never an input. Stage 1
encodes it three ways:

- residual base = ``tqs_peak`` (exact quasi-static timing peak, backfilled
  by scripts/patch_ibmpg_timing.py) — predicting 0 now floors at
  0.846/0.793 within-net on pg2t instead of 0.437/0.397;
- per-node binned load waveforms as conv features (near-field timing);
- time-structured attention values (sketched v_i(t) profile per head).

Floor-preserving training: the head is zero-initialized (start exactly
at the floor) and departures pay an L2 penalty (--res-penalty).
Model selection on a fully held-out validation grid (--val-bench),
not on training-grid node masks.

    python3.12 scripts/train_ibmpg_attn.py --holdout ibmpg2t \\
        --val-bench ibmpg4t --device cuda --epochs 60 \\
        --ckpt checkpoints/ibmpg_attn_s1.pt

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

    sd_log = np.log10(np.maximum(_static_drop(g), DROOP_FLOOR)).astype(np.float32)
    if "tqs_peak" in g.x_raw:
        base = np.log10(np.maximum(g.x_raw["tqs_peak"], DROOP_FLOOR)).astype(np.float32)
    else:
        print(f"  WARNING {bench}: no tqs_peak in npz "
              f"(run scripts/patch_ibmpg_timing.py) — falling back to static drop")
        base = sd_log

    # timing features: per-grid z of the timing-QS base + shape-normalized
    # binned load waveform (magnitude already lives in the zI feature)
    bz = (base - base.mean()) / (base.std() + 1e-6)
    cols = [x, np.clip(bz, -5, 5)[:, None].astype(np.float32)]
    if "wave_node" in g.x_raw:
        wb = g.x_raw["wave_bins"]
        wave = np.zeros((g.n_nodes, wb.shape[1]), dtype=np.float32)
        amp = np.abs(wb).max(axis=1, keepdims=True)
        wave[g.x_raw["wave_node"]] = wb / np.maximum(amp, 1e-12)
        cols.append(wave)
        data["node"].wave = torch.from_numpy(wave)
    data["node"].x = torch.from_numpy(np.concatenate(cols, axis=1))
    data["node"].sketch = torch.from_numpy(r)

    y_abs = np.log10(np.maximum(g.droop, DROOP_FLOOR)).astype(np.float32)
    data["node"].y = torch.from_numpy(y_abs - base if residual else y_abs)
    data["node"].base = torch.from_numpy(base)
    data["node"].sd_static = torch.from_numpy(sd_log)
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


def within_net_spearman(spearmanr, p, t, gm, vdd):
    """Mean of vdd-net and gnd-net Spearman over grid nodes."""
    rhos = []
    for m in (gm & vdd, gm & ~vdd):
        if m.sum() > 1:
            rhos.append(float(spearmanr(p[m], t[m]).statistic))
    return float(np.mean(rhos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ibmpg2t")
    ap.add_argument("--val-bench", default="ibmpg4t",
                    help="fully held-out grid for checkpoint selection "
                         "('none': legacy training-grid node masks)")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/ibmpg_attn_s1.pt"))
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--m-sketch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--res-penalty", type=float, default=1e-3,
                    help="L2 penalty on the residual prediction (0: off)")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-time-values", action="store_true",
                    help="ablation: scalar attention values (stage-0 style)")
    ap.add_argument("--train-benches", nargs="*", default=None,
                    help="subset of train grids (default: all except holdout+val)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    residual = not args.no_residual
    val_bench = None if args.val_bench in ("none", "") else args.val_bench

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_benches = args.train_benches or [
        b for b in ALL_BENCHES if b not in (args.holdout, val_bench)
    ]
    assert args.holdout not in train_benches

    print(f"loading train {train_benches} | val {val_bench} | holdout {args.holdout}")
    t0 = time.time()
    graphs = {b: load_graph(b) for b in train_benches}
    stats = compute_stats(list(graphs.values()))
    datas = {}
    for b, g in graphs.items():
        t1 = time.time()
        datas[b] = build_full_data(b, g, stats, args.m_sketch, residual, args.device)
        print(f"  {b}: {g.n_nodes} nodes, sketch+tensors in {time.time()-t1:.0f}s")
    if val_bench:
        g_val = load_graph(val_bench)
        data_val = build_full_data(val_bench, g_val, stats, args.m_sketch,
                                   residual, args.device)
        print(f"  {val_bench} (val): {g_val.n_nodes} nodes")
    print(f"total setup {time.time()-t0:.0f}s")

    # 5% training-grid node masks — kept as a train-fit diagnostic only
    diag_masks = {}
    for b, g in graphs.items():
        gm = np.flatnonzero(g.is_grid)
        vm = np.zeros(g.n_nodes, dtype=bool)
        vm[rng.choice(gm, max(1, gm.size // 20), replace=False)] = True
        diag_masks[b] = torch.from_numpy(vm).to(args.device)

    in_dim = datas[train_benches[0]]["node"].x.shape[1]
    model = IBMAttnRegressor(args.hidden_dim, args.n_conv, args.heads,
                             in_dim=in_dim,
                             time_values=not args.no_time_values).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | in_dim {in_dim} | "
          f"residual={residual} penalty={args.res_penalty} "
          f"time_values={not args.no_time_values} | device={args.device}")

    from scipy.stats import spearmanr
    best_rho, history = -np.inf, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for b in rng.permutation(train_benches):
            data = datas[b]
            m = data["node"].grid_mask & ~diag_masks[b]
            pred = model(data)
            loss = F.mse_loss(pred[m], data["node"].y[m])
            if residual and args.res_penalty > 0:
                loss = loss + args.res_penalty * pred[m].pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        with torch.no_grad():
            # train-fit diagnostic (within-grid spearman on held-out nodes)
            diag = []
            for b in train_benches:
                data = datas[b]
                m = diag_masks[b]
                p, t = model(data)[m], data["node"].y[m]
                if residual:
                    p, t = p + data["node"].base[m], t + data["node"].base[m]
                diag.append(float(spearmanr(p.cpu().numpy(),
                                            t.cpu().numpy()).statistic))
            diag_rho = float(np.mean(diag))
            # selection metric: within-net spearman on the val GRID
            if val_bench:
                p = model(data_val)
                t = data_val["node"].y
                if residual:
                    p, t = p + data_val["node"].base, t + data_val["node"].base
                rho = within_net_spearman(
                    spearmanr, p.cpu().numpy(), t.cpu().numpy(),
                    data_val["node"].grid_mask.cpu().numpy(),
                    data_val["node"].net_vdd.cpu().numpy())
            else:
                rho = diag_rho
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_within_net_spearman": rho,
                        "train_grids_spearman": diag_rho})
        print(f"ep{epoch:>3} loss {np.mean(losses):.4f} | val within-net "
              f"spearman {rho:.4f} | train-diag {diag_rho:.4f} "
              f"({time.time()-t0:.0f}s)")
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
            p = p + data["node"].base
        t = torch.from_numpy(
            np.log10(np.maximum(g_test.droop, DROOP_FLOOR))).to(args.device)
    gm = data["node"].grid_mask.cpu().numpy()
    vdd = data["node"].net_vdd.cpu().numpy()
    p = p.cpu().numpy(); t = t.cpu().numpy()
    base = data["node"].base.cpu().numpy()
    sd_static = data["node"].sd_static.cpu().numpy()

    report = {}
    for tag, m in (("all", gm), ("vdd", gm & vdd), ("gnd", gm & ~vdd)):
        report[tag] = _masked_metrics(p[m], t[m])
        report[f"baseline_tqs_{tag}"] = _masked_metrics(base[m], t[m])
        report[f"baseline_static_{tag}"] = _masked_metrics(sd_static[m], t[m])
    print(json.dumps(report, indent=2))
    hist_path = args.ckpt.with_suffix(".history.json")
    hist_path.write_text(json.dumps({"history": history, "test": report}, indent=2))
    print(f"history → {hist_path}")


if __name__ == "__main__":
    main()
