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


def train_one_epoch(model, loader, opt, device, grad_clip: float = 1.0) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch["mesh_bot"].y
        loss = F.mse_loss(pred, target)
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
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu())
        targets.append(batch["mesh_bot"].y.cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)

    # Loss in the training space
    train_space_loss = F.mse_loss(pred, target).item()

    # Convert to linear volts for reportable metrics
    pred_v = _to_linear_volts(pred, target_space)
    target_v = _to_linear_volts(target, target_space)

    err = pred_v - target_v
    mae_v = err.abs().mean().item()
    rmse_v = err.pow(2).mean().sqrt().item()
    mean_target_v = target_v.mean().item()

    ss_res = err.pow(2).sum().item()
    ss_tot = (target_v - target_v.mean()).pow(2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    # Per-sample worst-node droop
    pred_per = pred_v.view(-1, 49)
    target_per = target_v.view(-1, 49)
    pred_worst = pred_per.max(dim=1).values
    target_worst = target_per.max(dim=1).values
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


def make_loaders(h5_path, target_space: str, batch_size: int, num_workers: int = 0):
    from .pyg_dataset import RegularPDNDataset

    train = RegularPDNDataset(h5_path, split="train", target=target_space)
    val = RegularPDNDataset(h5_path, split="val", target=target_space)
    test = RegularPDNDataset(h5_path, split="test", target=target_space)
    common = {"batch_size": batch_size, "num_workers": num_workers}
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


def train(model, train_loader, val_loader, cfg: TrainConfig, device, target_space: str, ckpt_path=None):
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.n_epochs)

    best = {"epoch": -1, "val_loss": float("inf"), "metrics": None}
    history = []

    for epoch in range(1, cfg.n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device, cfg.grad_clip)
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
