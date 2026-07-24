"""Training and evaluation helpers for ``PDNDroopRegressor``.

Metrics are always reported in linear-volt space (mV) and on `worst_node_droop`
regardless of whether the model trained against the linear or log target —
inverse transform is applied during evaluation when ``target_space="log"``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from .pyg_dataset import LOG_FLOOR


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_epochs: int = 50
    batch_size: int = 32
    grad_clip: float = 1.0


def _to_linear_volts(t: torch.Tensor, target_space: str) -> torch.Tensor:
    if target_space == "log":
        return torch.pow(10.0, t)
    return t.clamp_min(0.0)


def _fold_bidir(grad_rows: torch.Tensor, vals: torch.Tensor, segs: torch.Tensor):
    """Fold per-directed-edge grads back to physical edges.

    Rows are packed per graph as ``[e_0..e_{S-1}, e_0..e_{S-1}]`` (bidir
    tiling), graphs concatenated. ``segs`` holds each graph's physical
    edge count S. Returns (grad_phys, val_phys) concatenated over graphs.
    """
    gs, vs = [], []
    off = 0
    for s in segs.tolist():
        block_g = grad_rows[off:off + 2 * s]
        block_v = vals[off:off + 2 * s]
        gs.append(block_g[:s] + block_g[s:])
        vs.append(block_v[:s])          # both halves carry the same value
        off += 2 * s
    return torch.cat(gs), torch.cat(vs)


def sobolev_loss(batch, pred) -> torch.Tensor:
    """Gradient-matching term: model's ∂(worst droop)/∂(ln knob) vs the
    exact adjoint labels, in per-graph relative-sensitivity units.

    ∂/∂ln ww_e = −R_e · ∂/∂R_e (R = Rsheet·pitch/ww);
    ∂/∂ln C_site = C · ∂/∂C_site. Uses ``torch.autograd.grad`` with
    ``create_graph=True`` (double backward) so the term trains the model.
    ``pred`` must come from a forward pass made AFTER enabling
    ``requires_grad`` on the strap/decap edge_attr tensors, and assumes
    log-space targets (the project default).
    """
    from .grid_construction import EDGE_ATTR_COLS
    R_COL = EDGE_ATTR_COLS.index("R")
    C_COL = EDGE_ATTR_COLS.index("C")
    et_top = ("mesh_top", "strap", "mesh_top")
    et_bot = ("mesh_bot", "strap", "mesh_bot")
    et_dec = ("mesh_bot", "decap", "mesh_bot")
    attrs = [batch[et].edge_attr for et in (et_top, et_bot, et_dec)]

    pred_v = torch.pow(10.0, pred)
    gid = batch["mesh_bot"].batch[batch["mesh_bot", "load", "mesh_bot"].edge_index[0]]
    n_graphs = int(batch.num_graphs)
    neg_inf = torch.full((n_graphs,), -torch.inf, device=pred_v.device)
    worst = neg_inf.scatter_reduce(0, gid, pred_v, reduce="amax", include_self=True)
    grads = torch.autograd.grad(worst.sum(), attrs, create_graph=True)

    segs = batch["jac_seg"].view(-1, 3)
    droop_true = neg_inf.scatter_reduce(
        0, gid, torch.pow(10.0, batch["y"]), reduce="amax", include_self=True)
    terms = []
    for i, (grad, attr, col, sign, lbl_key) in enumerate((
        (grads[0], attrs[0], R_COL, -1.0, "jac_top"),
        (grads[1], attrs[1], R_COL, -1.0, "jac_bot"),
        (grads[2], attrs[2], C_COL, +1.0, "jac_dec"),
    )):
        g_phys, v_phys = _fold_bidir(grad[:, col], attr[:, col].detach(), segs[:, i])
        model_jac = sign * v_phys * g_phys          # ∂(worst)/∂(ln knob)
        scale = torch.repeat_interleave(droop_true, segs[:, i])
        terms.append(F.mse_loss(model_jac / scale, batch[lbl_key] / scale))
    return sum(terms) / len(terms)


def train_one_epoch(model, loader, opt, device, grad_clip: float = 1.0,
                    sobolev_lambda: float = 0.0) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        use_sob = sobolev_lambda > 0 and bool(batch["has_jac"].all())
        if use_sob:
            for et in (("mesh_top", "strap", "mesh_top"),
                       ("mesh_bot", "strap", "mesh_bot"),
                       ("mesh_bot", "decap", "mesh_bot")):
                batch[et].edge_attr.requires_grad_(True)
        pred = model(batch)
        target = batch["y"]
        loss = F.mse_loss(pred, target)
        if use_sob:
            loss = loss + sobolev_lambda * sobolev_loss(batch, pred)
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        bs = target.numel()
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, target_space: str = "log") -> dict:
    """Compute aggregate + per-sample-worst-load metrics.

    Load counts vary per graph (14 on the 7-die, 52 on the 13-die), so
    the per-sample worst is computed by scatter-max over a per-load
    graph-id vector (the batch assignment of each load edge's source
    node) rather than a rectangular reshape.
    """
    model.eval()
    preds, targets, gids = [], [], []
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu())
        targets.append(batch["y"].cpu())
        ei = batch["mesh_bot", "load", "mesh_bot"].edge_index
        gids.append(batch["mesh_bot"].batch[ei[0]].cpu() + n_graphs)
        n_graphs += batch.num_graphs
    pred = torch.cat(preds)
    target = torch.cat(targets)
    gid = torch.cat(gids)

    train_space_loss = F.mse_loss(pred, target).item()

    pred_v = _to_linear_volts(pred, target_space)
    target_v = _to_linear_volts(target, target_space)

    err = pred_v - target_v
    mae_v = err.abs().mean().item()
    rmse_v = err.pow(2).mean().sqrt().item()
    mean_target_v = target_v.mean().item()

    ss_res = err.pow(2).sum().item()
    ss_tot = (target_v - target_v.mean()).pow(2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    neg_inf = torch.full((n_graphs,), -torch.inf)
    pred_worst = neg_inf.clone().scatter_reduce(
        0, gid, pred_v, reduce="amax", include_self=True
    )
    target_worst = neg_inf.clone().scatter_reduce(
        0, gid, target_v, reduce="amax", include_self=True
    )
    worst_err = pred_worst - target_worst
    worst_mae_v = worst_err.abs().mean().item()
    worst_rel = worst_mae_v / target_worst.mean().clamp_min(LOG_FLOOR).item()

    return {
        "loss_train_space": train_space_loss,
        "mae_mV": mae_v * 1e3,
        "rmse_mV": rmse_v * 1e3,
        "rel_mae": mae_v / max(mean_target_v, LOG_FLOOR),
        "r2": r2,
        "worst_mae_mV": worst_mae_v * 1e3,
        "worst_rel_mae": worst_rel,
    }


def make_loaders(h5_path, target_space: str, batch_size: int, num_workers: int = 0,
                 jac_path=None):
    from .pyg_dataset import RegularPDNDataset

    train = RegularPDNDataset(h5_path, split="train", target=target_space,
                              jac_path=jac_path)
    val = RegularPDNDataset(h5_path, split="val", target=target_space)
    test = RegularPDNDataset(h5_path, split="test", target=target_space)
    common = {"batch_size": batch_size, "num_workers": num_workers}
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


def train(model, train_loader, val_loader, cfg: TrainConfig, device, target_space: str,
          ckpt_path=None, sobolev_lambda: float = 0.0):
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.n_epochs)

    best = {"epoch": -1, "val_loss": float("inf"), "metrics": None}
    history = []

    for epoch in range(1, cfg.n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device, cfg.grad_clip,
                                     sobolev_lambda=sobolev_lambda)
        val_metrics = evaluate(model, val_loader, device, target_space)
        sched.step()

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        if val_metrics["loss_train_space"] < best["val_loss"]:
            best = {
                "epoch": epoch,
                "val_loss": val_metrics["loss_train_space"],
                "metrics": val_metrics,
            }
            if ckpt_path is not None:
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics},
                    ckpt_path,
                )

        print(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss_train_space']:.4f}  "
            f"val_mae={val_metrics['mae_mV']:.3f} mV  "
            f"val_R²={val_metrics['r2']:.4f}  "
            f"worst_mae={val_metrics['worst_mae_mV']:.3f} mV"
        )

    return best, history
