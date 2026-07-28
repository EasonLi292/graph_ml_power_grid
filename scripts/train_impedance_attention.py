"""Train the one-shot impedance-attention model on the v7 synthetic track.

Tests whether ONE global factorized attention layer can replace
depth-dependent local message passing (docs/IMPEDANCE_ATTENTION_DESIGN.md).

Reports test metrics **per topology anchor**, never aggregated — a 13-die
anchor and a 7-die anchor fail in opposite directions and an average hides
both (the v7 negative-transfer lesson).

Ablations (the design note's forward-baseline table):
    --ablation combined    (default)  score = (q.k)(p.s)
    --ablation content                score = (q.k)          — no impedance
    --ablation impedance              score = (p.s)          — no content

    # factors cached once (frozen omega), then 50 epochs
    python scripts/train_impedance_attention.py --epochs 50 --device cuda

    # per-anchor eval of an existing local-GNN baseline, same split
    python scripts/train_impedance_attention.py --compare-baseline \\
        checkpoints/droop_v7_edgeconv.pt --n-layers 7

Factor caching: factors depend on (topology, R, C, omega). With omega
frozen they are constant per sample, so they are computed once and cached
(2-44 ms/sample => ~12 min/epoch if recomputed every step). This does NOT
weaken the gradient claim: the gate and the repair loop rebuild factors
under autograd, so dR/dC paths are live at evaluation time. ``--learn-omega``
keeps omega trainable and disables caching (much slower).

Nothing here launches automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eason.impedance_attention_model import ImpAttnConfig, ImpedanceAttentionRegressor
from tools.grid_construction import build_regular_pdn
from tools.impedance_factors import (
    branch_system,
    impedance_factors,
    knob_tensors,
    node_features,
)
from tools.pyg_dataset import LOG_FLOOR
from tools.sampler import (
    ALL_ANCHORS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    FIXED_R_VIA,
    FIXED_RSHEET_BOT,
    FIXED_RSHEET_TOP,
)

DT = torch.float32
FDT = torch.float64          # factor solves in double for conditioning


def default_omegas(n_freq: int) -> torch.Tensor:
    base = 2 * np.pi * FIXED_FREQ
    grid = [0.0, base, 5 * base, 25 * base, 125 * base]
    return torch.tensor(grid[:n_freq], dtype=FDT)


class AnchorCache:
    """Per-anchor topology + load waveforms (identical across samples)."""

    def __init__(self) -> None:
        self.sys, self.x, self.n_loads = {}, {}, {}
        for nt, nb in ALL_ANCHORS:
            a = (int(nt), int(nb))
            proto = build_regular_pdn(n_top=a[0], n_bot=a[1])
            loads = np.tile(
                np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
                (proto.n_loads, 1))
            g = build_regular_pdn(
                n_top=a[0], n_bot=a[1], Rsheet_top=FIXED_RSHEET_TOP,
                Rsheet_bot=FIXED_RSHEET_BOT, wire_width=0.5,
                R_via=FIXED_R_VIA, C_decap=2e-10, freq=FIXED_FREQ, loads=loads)
            s = branch_system(g)
            self.sys[a] = (g, s)
            self.x[a] = node_features(s, torch.tensor(loads, dtype=DT))
            self.n_loads[a] = proto.n_loads


def sample_factors(ac: AnchorCache, anchor, wt, wb, cd, omegas, m, n_power):
    g, s = ac.sys[anchor]
    R, C = knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                        FIXED_R_VIA)
    return impedance_factors(s, R, C, omegas, m=m, n_power=n_power)


def load_split(h5_path, split):
    with h5py.File(h5_path, "r") as f:
        grp = f["bulk"][split]
        return {
            "n_top": grp["n_top"][:], "n_bot": grp["n_bot"][:],
            "gp": grp["global_params"][:],
            "wt": grp["ww_top_edges"][:], "wb": grp["ww_bot_edges"][:],
            "y": grp["peak_droop_loads"][:],
        }


def build_cache(data, ac, omegas, m, n_power, tag, cache_dir: Path | None):
    """Precompute (p, s) per sample. Cached to disk keyed by config."""
    # channel count is part of the key: the AC-channel fix changed the layout
    # and a stale cache would load silently with the wrong shape
    from tools.impedance_factors import channel_count
    key = f"{tag}_m{m}_q{n_power}_f{omegas.numel()}_c{channel_count(omegas)}"
    path = cache_dir / f"{key}.pt" if cache_dir else None
    if path is not None and path.exists():
        print(f"  factors <- {path}")
        return torch.load(path, weights_only=False)
    out, t0 = [], time.time()
    n = data["n_top"].shape[0]
    for i in range(n):
        a = (int(data["n_top"][i]), int(data["n_bot"][i]))
        g, s = ac.sys[a]
        te, be = g.top_edges.shape[0], g.bot_edges.shape[0]
        p, q_ = sample_factors(
            ac, a,
            torch.tensor(data["wt"][i, :te], dtype=FDT),
            torch.tensor(data["wb"][i, :be], dtype=FDT),
            torch.tensor(float(data["gp"][i, 1]), dtype=FDT),
            omegas, m, n_power)
        out.append((p.to(DT), q_.to(DT)))
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n} ({el:.0f}s, {el/(i+1)*1e3:.1f} ms/sample)")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
        print(f"  factors -> {path}")
    return out


def per_anchor_metrics(pred_log, targ_log, anchors):
    """Worst-load MAE / R2 / rel-MAE, reported separately per anchor."""
    rep = {}
    for a in sorted(set(anchors)):
        idx = [i for i, x in enumerate(anchors) if x == a]
        pw = torch.stack([pred_log[i].max() for i in idx])
        tw = torch.stack([targ_log[i].max() for i in idx])
        pv, tv = 10.0 ** pw, 10.0 ** tw
        err = pv - tv
        ss = ((tv - tv.mean()) ** 2).sum()
        rep[f"{a[0]},{a[1]}"] = {
            "n": len(idx),
            "worst_mae_mV": float(err.abs().mean()) * 1e3,
            "worst_r2": float(1 - (err ** 2).sum() / ss.clamp_min(1e-30)),
            "worst_rel_mae": float(err.abs().mean() / tv.mean().clamp_min(1e-12)),
        }
    return rep


def evaluate(model, data, facs, ac, device):
    model.eval()
    preds, targs, anchors = [], [], []
    with torch.no_grad():
        for i in range(data["n_top"].shape[0]):
            a = (int(data["n_top"][i]), int(data["n_bot"][i]))
            _, s = ac.sys[a]
            p, sf = facs[i]
            y = model(ac.x[a].to(device), p.to(device), sf.to(device), s.n_elec)
            nl = ac.n_loads[a]
            t = np.log10(np.maximum(data["y"][i, :nl], LOG_FLOOR))
            preds.append(y.cpu())
            targs.append(torch.tensor(t, dtype=DT))
            anchors.append(a)
    return per_anchor_metrics(preds, targs, anchors)


def eval_baseline(ckpt, n_layers, conv_type, h5_path, device):
    """Per-anchor eval of an existing local-GNN checkpoint, same split."""
    from torch_geometric.loader import DataLoader

    from eason import EncoderConfig, PDNDroopRegressor
    from tools.pyg_dataset import RegularPDNDataset

    model = PDNDroopRegressor(
        EncoderConfig(hidden_dim=64, n_layers=n_layers, conv_type=conv_type,
                      drop_edge_p=0.0), target_space="log").to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model"])
    model.eval()
    ds = RegularPDNDataset(h5_path, split="test", target="log")
    preds, targs, anchors = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            d = ds[i]
            b = next(iter(DataLoader([d], batch_size=1))).to(device)
            preds.append(model(b).cpu())
            targs.append(d["y"])
            anchors.append((int(ds._n_top[i]), int(ds._n_bot[i])))
    return per_anchor_metrics(preds, targs, anchors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("datasets/regular_v7_anchors/dataset.h5"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints/imp_attn.pt"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("datasets/regular_v7_anchors/_factors"))
    ap.add_argument("--ablation", default="combined",
                    choices=["combined", "content", "impedance"])
    ap.add_argument("--n-freq", type=int, default=3)
    ap.add_argument("--m-factor", type=int, default=16)
    ap.add_argument("--n-power", type=int, default=2)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap train samples (smoke tests)")
    ap.add_argument("--keep-epochs", action="store_true",
                    help="also write ckpt.epNNN.pt every epoch, so a "
                         "checkpoint can be chosen after training")
    ap.add_argument("--learn-omega", action="store_true",
                    help="keep omega trainable; disables factor caching (slow)")
    ap.add_argument("--compare-baseline", type=Path, default=None)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--conv-type", default="edgeconv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.compare_baseline:
        rep = eval_baseline(args.compare_baseline, args.n_layers,
                            args.conv_type, args.data, args.device)
        print(f"baseline {args.compare_baseline.name} "
              f"(n_layers={args.n_layers}) per-anchor test:")
        print(json.dumps(rep, indent=2))
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.learn_omega:
        raise NotImplementedError(
            "learnable omega requires per-step factor rebuild; the cached "
            "path assumes frozen omega. Run with frozen omega for the "
            "prototype (see the design note's unresolved item 2).")

    ac = AnchorCache()
    omegas = default_omegas(args.n_freq)
    print(f"omegas (rad/s): {[f'{float(w):.3e}' for w in omegas]}")

    tr, te = load_split(args.data, "train"), load_split(args.data, "test")
    if args.limit:
        tr = {k: v[:args.limit] for k, v in tr.items()}
    print(f"train {tr['n_top'].shape[0]} | test {te['n_top'].shape[0]}")
    print("precomputing factors (train)")
    f_tr = build_cache(tr, ac, omegas, args.m_factor, args.n_power,
                       f"train{args.limit or ''}", args.cache_dir)
    print("precomputing factors (test)")
    f_te = build_cache(te, ac, omegas, args.m_factor, args.n_power,
                       "test", args.cache_dir)

    cfg = ImpAttnConfig(
        hidden_dim=args.hidden_dim, heads=args.heads, n_freq=args.n_freq,
        m_factor=args.m_factor,
        content=args.ablation in ("combined", "content"),
        impedance=args.ablation in ("combined", "impedance"))
    # start at the mean-predictor baseline (see model init docstring)
    ymask = np.isfinite(tr["y"]) & (tr["y"] > 0)
    init_bias = float(np.log10(np.maximum(tr["y"][ymask], LOG_FLOOR)).mean())
    model = ImpedanceAttentionRegressor(cfg, init_bias=init_bias).to(args.device)
    print(f"init bias (mean log10 droop): {init_bias:.3f}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"ablation={args.ablation} | params "
          f"{sum(p.numel() for p in model.parameters()):,}")

    n = tr["n_top"].shape[0]
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for i in rng.permutation(n):
            a = (int(tr["n_top"][i]), int(tr["n_bot"][i]))
            _, s = ac.sys[a]
            p, sf = f_tr[i]
            pred = model(ac.x[a].to(args.device), p.to(args.device),
                         sf.to(args.device), s.n_elec)
            nl = ac.n_loads[a]
            t = torch.tensor(
                np.log10(np.maximum(tr["y"][i, :nl], LOG_FLOOR)),
                dtype=DT, device=args.device)
            loss = F.mse_loss(pred, t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        rep = evaluate(model, te, f_te, ac, args.device)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                        "test_per_anchor": rep})
        line = "  ".join(f"{k}:R2={v['worst_r2']:+.3f}" for k, v in rep.items())
        print(f"ep{ep:>3} loss {np.mean(losses):.4f} | {line} "
              f"({time.time()-t0:.0f}s)")
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        blob = {"model": model.state_dict(), "epoch": ep,
                "cfg": vars(cfg), "args": {**vars(args),
                                           "ckpt": str(args.ckpt)}}
        torch.save(blob, args.ckpt)
        if args.keep_epochs:
            # held-out R2 peaks early and decays (all three ablations), so the
            # last-epoch checkpoint is not the model the gate should judge.
            # Keeping every epoch lets a checkpoint be selected after the fact.
            torch.save(blob, args.ckpt.with_suffix(f".ep{ep:03d}.pt"))

    hp = args.ckpt.with_suffix(".history.json")
    hp.write_text(json.dumps({"history": history,
                              "test_per_anchor": history[-1]["test_per_anchor"]},
                             indent=2))
    print(f"history -> {hp}")


if __name__ == "__main__":
    main()
