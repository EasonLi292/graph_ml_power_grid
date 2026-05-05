"""Train ``PDNDroopRegressor`` on the regular-PDN dataset.

Usage:
    python scripts/train_droop.py \
        --data datasets/regular_v1/dataset.h5 \
        --target log --epochs 50 --hidden-dim 64 --batch-size 32 \
        --ckpt checkpoints/droop_regressor.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.encoder import EncoderConfig, PDNDroopRegressor
from tools.training import TrainConfig, evaluate, make_loaders, train


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("datasets/regular_v1/dataset.h5"))
    ap.add_argument("--target", choices=["log", "linear"], default="log")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu | cuda | mps | auto")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/droop_regressor.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"device: {device}, target_space: {args.target}")

    train_loader, val_loader, test_loader = make_loaders(
        args.data, args.target, args.batch_size, args.num_workers
    )

    enc_cfg = EncoderConfig(
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout
    )
    model = PDNDroopRegressor(enc_cfg, target_space=args.target).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    best, history = train(
        model, train_loader, val_loader, cfg, device, args.target, ckpt_path=args.ckpt
    )

    # Reload best weights for the test report
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device, args.target)
    print("\n== best (val) ==")
    print(json.dumps(best["metrics"], indent=2))
    print("\n== test (best ckpt) ==")
    print(json.dumps(test_metrics, indent=2))

    log_path = args.ckpt.with_suffix(".history.json")
    log_path.write_text(json.dumps({"history": history, "test": test_metrics}, indent=2))
    print(f"history written to {log_path}")


if __name__ == "__main__":
    main()
