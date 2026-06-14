"""Print an exact parameter-count breakdown of PDNDroopRegressor.

Source of truth for docs/MODEL_SIZE.md. Run:  python3.12 scripts/model_params.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eason import EncoderConfig, PDNDroopRegressor


def build(hidden_dim: int, n_layers: int) -> PDNDroopRegressor:
    return PDNDroopRegressor(
        EncoderConfig(hidden_dim=hidden_dim, n_layers=n_layers,
                      conv_type="admittance"),
        target_space="log",
    )


def count(model, prefix: str) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))


def main() -> None:
    H, L = 64, 7
    m = build(H, L)
    total = sum(p.numel() for p in m.parameters())
    print(f"default config: hidden_dim={H}, n_layers={L}")
    print(f"TOTAL trainable parameters: {total:,}")
    print(f"float32 size: {total * 4 / 1e6:.3f} MB\n")

    print("top-level breakdown:")
    for grp in ["encoder.node_proj", "encoder.edge_proj", "encoder.convs",
                "encoder.norms", "head"]:
        c = count(m, grp)
        print(f"  {grp:22s} {c:>10,}  ({100*c/total:4.1f}%)")

    # per-relation within one conv layer
    rel = collections.OrderedDict()
    for n, p in m.named_parameters():
        if n.startswith("encoder.convs.0.convs."):
            key = n.split("convs.0.convs.")[1].split(">")[0].strip("<")
            rel[key] = rel.get(key, 0) + p.numel()
    print(f"\none conv layer = {sum(rel.values()):,} params  (×{L} layers):")
    for k, v in rel.items():
        print(f"  {k:30s} {v:>8,}")

    print("\nexact closed form:  N(h, L) = (34L+2)·h² + (30L+56)·h + (4L+1)")
    print("scaling table (rows hidden_dim, cols n_layers):")
    print(f"{'h\\L':>6} " + " ".join(f"{x:>11}" for x in [3, 5, 7]))
    for h in [32, 64, 128]:
        row = " ".join(f"{sum(p.numel() for p in build(h, x).parameters()):>11,}"
                        for x in [3, 5, 7])
        print(f"{h:>6} {row}")


if __name__ == "__main__":
    main()
