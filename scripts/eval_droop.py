"""Load a trained checkpoint and report metrics on a chosen split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.encoder import EncoderConfig, PDNDroopRegressor
from tools.pyg_dataset import RegularPDNDataset
from tools.training import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("datasets/regular_v4/dataset.h5"))
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/droop_regressor.pt"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--target", choices=["log", "linear"], default="log")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = RegularPDNDataset(args.data, split=args.split, target=args.target)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = PDNDroopRegressor(
        EncoderConfig(hidden_dim=args.hidden_dim, n_layers=args.n_layers),
        target_space=args.target,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])

    metrics = evaluate(model, loader, device, args.target)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
