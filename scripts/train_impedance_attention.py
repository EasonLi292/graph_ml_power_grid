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

**Checkpoint selection is broken by default, and it is the dataset's
fault, not the model's.** The val split holds out *samples* of the four
training topologies; the test split holds out *topologies*. Measured over
50 epochs (seed 0, three architectures):

    arch        corr(val, OOD test)   best OOD epoch   val-selected OOD
    bilinear          -0.290                1              +0.110
    kernel n_rff=128  -0.138                2                 ...
    kernel n_rff=512  -0.490                1                 ...

Val climbs +0.675 -> +0.969 while held-out-topology test *falls*
+0.755 -> +0.117. Selecting on val costs 0.645 test R2 versus stopping at
the OOD optimum. Every architecture shows it, so it is a property of the
4-topology training set: 16k samples over 4 topologies lets the model
memorise topology-specific structure long before it runs out of capacity.

``--holdout-anchor`` buys an honest selection signal by excluding one
training topology and selecting on it. It costs 25 % of the topological
diversity, which on a 4-topology set is expensive. The real fix is more
topologies with fewer samples each (dataset regeneration — the anchor
lists live in ``tools/sampler.py``).

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
    dc_symmetric_factor,
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
    """Per-anchor topology + load waveforms (identical across samples).

    Built lazily. With the 14-anchor set the largest topology is (31,31)
    at 1922 electrical nodes, so eagerly constructing every anchor costs
    real time and memory for anchors a given run may never touch (e.g.
    under ``--holdout-anchor``, or in probes that use one anchor).
    """

    def __init__(self) -> None:
        self.sys, self.x, self.n_loads = {}, {}, {}

    def _build(self, a: tuple[int, int]) -> None:
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

    def ensure(self, a: tuple[int, int]) -> tuple[int, int]:
        a = (int(a[0]), int(a[1]))
        if a not in self.sys:
            self._build(a)
        return a


def sample_factors(ac: AnchorCache, anchor, wt, wb, cd, omegas, m, n_power,
                   want_fdc=False):
    g, s = ac.sys[ac.ensure(anchor)]
    R, C = knob_tensors(g, wt, wb, cd, FIXED_RSHEET_TOP, FIXED_RSHEET_BOT,
                        FIXED_R_VIA)
    p, sf = impedance_factors(s, R, C, omegas, m=m, n_power=n_power)
    fdc = (dc_symmetric_factor(s, R, C, m=m, n_power=n_power)
           if want_fdc else None)
    return p, sf, fdc


def load_split(h5_path, split):
    with h5py.File(h5_path, "r") as f:
        grp = f["bulk"][split]
        return {
            "n_top": grp["n_top"][:], "n_bot": grp["n_bot"][:],
            "gp": grp["global_params"][:],
            "wt": grp["ww_top_edges"][:], "wb": grp["ww_bot_edges"][:],
            "y": grp["peak_droop_loads"][:],
        }


def filter_anchors(data, anchors, keep: bool):
    """Subset ``data`` to (keep=True) or away from (keep=False) ``anchors``."""
    sel = np.zeros(data["n_top"].shape[0], dtype=bool)
    for a in anchors:
        sel |= (data["n_top"] == a[0]) & (data["n_bot"] == a[1])
    if not keep:
        sel = ~sel
    return {k: v[sel] for k, v in data.items()}


def build_cache(data, ac, omegas, m, n_power, tag, cache_dir: Path | None,
                want_fdc: bool = False):
    """Precompute (p, s) per sample. Cached to disk keyed by config."""
    # channel count is part of the key: the AC-channel fix changed the layout
    # and a stale cache would load silently with the wrong shape
    from tools.impedance_factors import channel_count
    key = (f"{tag}_m{m}_q{n_power}_f{omegas.numel()}"
           f"_c{channel_count(omegas)}{'_fdc' if want_fdc else ''}")
    path = cache_dir / f"{key}.pt" if cache_dir else None
    if path is not None and path.exists():
        print(f"  factors <- {path}")
        return torch.load(path, weights_only=False)
    out, t0 = [], time.time()
    n = data["n_top"].shape[0]
    for i in range(n):
        a = ac.ensure((data["n_top"][i], data["n_bot"][i]))
        g, s = ac.sys[a]
        te, be = g.top_edges.shape[0], g.bot_edges.shape[0]
        p, q_, fdc = sample_factors(
            ac, a,
            torch.tensor(data["wt"][i, :te], dtype=FDT),
            torch.tensor(data["wb"][i, :be], dtype=FDT),
            torch.tensor(float(data["gp"][i, 1]), dtype=FDT),
            omegas, m, n_power, want_fdc=want_fdc)
        out.append((p.to(DT), q_.to(DT),
                    fdc.to(DT) if fdc is not None else None))
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
            a = ac.ensure((data["n_top"][i], data["n_bot"][i]))
            _, s = ac.sys[a]
            p, sf, fdc = facs[i]
            y = model(ac.x[a].to(device), p.to(device), sf.to(device), s.n_elec,
                      fdc=None if fdc is None else fdc.to(device))
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
    ap.add_argument("--score", default="bilinear",
                    choices=["bilinear", "kernel"])
    ap.add_argument("--n-scales", type=int, default=3)
    ap.add_argument("--n-rff", type=int, default=128)
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
    ap.add_argument("--holdout-anchor", nargs="*", default=None,
                    metavar="NT,NB",
                    help="Exclude these topologies from training and select "
                         "on them. Without it, val is held-out SAMPLES of "
                         "trained topologies, which is ANTI-correlated with "
                         "held-out TOPOLOGY performance (r=-0.29 bilinear, "
                         "-0.49 kernel512) — see the module docstring. "
                         "Pass a whole die size (e.g. --holdout-anchor 9,25 "
                         "13,25 25,25) to make the selection signal match "
                         "the size-extrapolation test axis; a single anchor "
                         "only holds out a density within a still-trained "
                         "die size.")
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

    want_fdc = args.score == "kernel"
    ac = AnchorCache()
    omegas = default_omegas(args.n_freq)
    print(f"omegas (rad/s): {[f'{float(w):.3e}' for w in omegas]}")

    tr, te = load_split(args.data, "train"), load_split(args.data, "test")
    va = load_split(args.data, "val")
    if args.holdout_anchor:
        hos = [tuple(int(v) for v in tok.split(","))
               for tok in args.holdout_anchor]
        n_before = tr["n_top"].shape[0]
        tr = filter_anchors(tr, hos, keep=False)     # never trained on
        va = filter_anchors(va, hos, keep=True)      # selection signal only
        print(f"topology holdout {hos}: train {n_before} -> "
              f"{tr['n_top'].shape[0]}, val = {va['n_top'].shape[0]} samples "
              f"of the held-out topolog{'y' if len(hos) == 1 else 'ies'}")
        if va["n_top"].shape[0] == 0:
            raise SystemExit(f"no val samples for anchors {hos}")
        if tr["n_top"].shape[0] == 0:
            raise SystemExit("holdout removed every training sample")
    if args.limit:
        tr = {k: v[:args.limit] for k, v in tr.items()}
        va = {k: v[:max(64, args.limit // 4)] for k, v in va.items()}
    print(f"train {tr['n_top'].shape[0]} | val {va['n_top'].shape[0]} "
          f"| test {te['n_top'].shape[0]}")
    print("precomputing factors (train)")
    f_tr = build_cache(tr, ac, omegas, args.m_factor, args.n_power,
                       f"train{args.limit or ''}", args.cache_dir, want_fdc)
    print("precomputing factors (val)")
    f_va = build_cache(va, ac, omegas, args.m_factor, args.n_power,
                       f"val{args.limit or ''}", args.cache_dir, want_fdc)
    print("precomputing factors (test)")
    f_te = build_cache(te, ac, omegas, args.m_factor, args.n_power,
                       "test", args.cache_dir, want_fdc)

    cfg = ImpAttnConfig(
        hidden_dim=args.hidden_dim, heads=args.heads, n_freq=args.n_freq,
        m_factor=args.m_factor,
        content=args.ablation in ("combined", "content"),
        impedance=args.ablation in ("combined", "impedance"),
        score=args.score, n_scales=args.n_scales, n_rff=args.n_rff)
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
    best_val, best_ep = -np.inf, -1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for i in rng.permutation(n):
            a = ac.ensure((tr["n_top"][i], tr["n_bot"][i]))
            _, s = ac.sys[a]
            p, sf, fdc = f_tr[i]
            pred = model(ac.x[a].to(args.device), p.to(args.device),
                         sf.to(args.device), s.n_elec,
                         fdc=None if fdc is None else fdc.to(args.device))
            nl = ac.n_loads[a]
            t = torch.tensor(
                np.log10(np.maximum(tr["y"][i, :nl], LOG_FLOOR)),
                dtype=DT, device=args.device)
            loss = F.mse_loss(pred, t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        rep = evaluate(model, te, f_te, ac, args.device)
        # Selection runs on the VAL split (train anchors only — no test
        # leakage). Held-out R2 oscillates by up to 0.5 at (7,13) between
        # late epochs, so the last epoch is a noisy draw, not the model to
        # ship or to gate.
        vrep = evaluate(model, va, f_va, ac, args.device)
        # Selection score: mean over anchors of R2 CLAMPED BELOW AT -1.
        # per_anchor_metrics clamps SS_tot at 1e-30, so an anchor whose
        # worst-load droop happens to have near-zero variance can emit an
        # R2 of -1e23 and single-handedly decide selection (seen at -2e23
        # on a 20-sample smoke). Anything below -1 is "worse than
        # predicting the mean", and for *choosing a checkpoint* all such
        # anchors are equally bad — so floor them rather than let one
        # degenerate anchor outvote the other nine. Reporting stays
        # unclamped; this affects selection only.
        vscore = float(np.mean([max(v["worst_r2"], -1.0)
                                for v in vrep.values()]))
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                        "test_per_anchor": rep, "val_per_anchor": vrep,
                        "val_score": vscore})
        line = "  ".join(f"{k}:R2={v['worst_r2']:+.3f}" for k, v in rep.items())
        print(f"ep{ep:>3} loss {np.mean(losses):.4f} | {line} "
              f"| val {vscore:+.3f} ({time.time()-t0:.0f}s)")
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        blob = {"model": model.state_dict(), "epoch": ep,
                "cfg": vars(cfg), "args": {**vars(args),
                                           "ckpt": str(args.ckpt)}}
        torch.save(blob, args.ckpt.with_suffix(".last.pt"))
        if vscore > best_val:
            best_val, best_ep = vscore, ep
            torch.save(blob, args.ckpt)          # <-- the selected checkpoint
        if args.keep_epochs:
            torch.save(blob, args.ckpt.with_suffix(f".ep{ep:03d}.pt"))

    sel = next(h for h in history if h["epoch"] == best_ep)
    print(f"\nselected epoch {best_ep} (val {best_val:+.3f}) -> {args.ckpt}")
    print(f"  its test per-anchor: " + "  ".join(
        f"{k}:R2={v['worst_r2']:+.3f}" for k, v in sel["test_per_anchor"].items()))
    print(f"  last epoch for reference: " + "  ".join(
        f"{k}:R2={v['worst_r2']:+.3f}" for k, v in history[-1]["test_per_anchor"].items()))
    hp = args.ckpt.with_suffix(".history.json")
    hp.write_text(json.dumps({"history": history,
                              "selected_epoch": best_ep,
                              "selected_val_score": best_val,
                              "test_per_anchor": sel["test_per_anchor"],
                              "test_per_anchor_last": history[-1]["test_per_anchor"]},
                             indent=2))
    print(f"history -> {hp}")


if __name__ == "__main__":
    main()
