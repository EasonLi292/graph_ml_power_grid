"""Gradient-based inverse design via the trained surrogate.

The classic ML-for-design loop:

    1. Initialise a *latent* ``z`` (unconstrained ℝ²).
    2. Decode z → physical parameters in the valid box::

           wire_width = sigmoid(z₀) · (ww_hi − ww_lo) + ww_lo
           C_decap    = 10^( sigmoid(z₁) · (log10 cd_hi − log10 cd_lo) + log10 cd_lo )

       (so any z gives a valid design — no projection step inside the loop.)
    3. Build the PDN graph from these continuous parameters with **autograd
       enabled on the edge attributes**, push it through the frozen
       surrogate, and read the predicted worst-load droop.
    4. Loss is *hinge spec + cost*::

           L = ReLU(droop − target)²  +  λ · cost(wire_width, C_decap)

       — when the design is infeasible, the spec term dominates and drives
       droop down; once feasible, the cost term pulls toward the cheapest
       point on the spec boundary.
    5. Backprop through (surrogate → graph builder → decoder) and Adam-step ``z``.
    6. After N steps, report the recovered physical design and validate
       against the simulator.

The "decode" step is the bridge to a VAE-style generator: replace the
hand-coded sigmoid with a learned ``decoder_mlp(z)`` and you get the
same loop with arbitrarily high-dimensional latent + learned design
distribution.

Usage:
    python scripts/design_grad.py \\
        --ckpt checkpoints/droop_v5_conductance.pt \\
        --target-mV 0.10 \\
        --lambda-cost 1e-3
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eason import EncoderConfig, PDNDroopRegressor
from tools.dataset_runner import SimConfig, run_one
from tools.grid_construction import (
    EDGE_ATTR_COLS,
    EDGE_ATTR_DIM,
    build_regular_pdn,
    to_hetero_data,
)
from tools.pyg_dataset import LOG_FLOOR
from tools.sampler import (
    ALL_N_TOP,
    FIXED_CONSTANTS,
    FIXED_DUTY,
    FIXED_FREQ,
    FIXED_I_PEAK,
    FIXED_PHASE,
    GLOBAL_RANGES,
)


R_COL = EDGE_ATTR_COLS.index("R")
C_COL = EDGE_ATTR_COLS.index("C")
I_COL = EDGE_ATTR_COLS.index("I_peak")
F_COL = EDGE_ATTR_COLS.index("freq")
D_COL = EDGE_ATTR_COLS.index("duty")
P_COL = EDGE_ATTR_COLS.index("phase")


# -----------------------------------------------------------------------------
# Decoder: ℝ² → physical (wire_width, C_decap)
# -----------------------------------------------------------------------------

def decode(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid-mapped decoder. ``z`` is shape [2]."""
    ww_p = GLOBAL_RANGES.by_name("wire_width")
    cd_p = GLOBAL_RANGES.by_name("C_decap")
    sig = torch.sigmoid(z)
    wire_width = sig[0] * (ww_p.hi - ww_p.lo) + ww_p.lo
    log10_cd = sig[1] * (math.log10(cd_p.hi) - math.log10(cd_p.lo)) + math.log10(cd_p.lo)
    C_decap = 10.0 ** log10_cd
    return wire_width, C_decap


# -----------------------------------------------------------------------------
# Differentiable graph builder: takes (wire_width, C_decap) tensors and produces
# a HeteroData whose edge_attr tensors carry autograd through to the model.
# -----------------------------------------------------------------------------

def build_diff_data(
    n_top: int,
    wire_width: torch.Tensor,
    C_decap: torch.Tensor,
) -> Batch:
    """Build a single-graph batch with differentiable edge attributes.

    Topology (edge_index lists, node features) is computed once from
    float placeholders, then the edge_attr tensors are rebuilt as
    differentiable functions of ``wire_width`` and ``C_decap``.
    """
    # Topology placeholder (any non-zero values; edge_attr is overridden below).
    g_proto = build_regular_pdn(n_top=n_top, wire_width=0.5, C_decap=1e-10)
    n_loads = g_proto.n_loads
    loads = np.tile(
        np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
        (n_loads, 1),
    )
    g = build_regular_pdn(
        n_top=n_top,
        n_bot=FIXED_CONSTANTS["n_bot"],
        Rsheet_top=FIXED_CONSTANTS["Rsheet_top"],
        Rsheet_bot=FIXED_CONSTANTS["Rsheet_bot"],
        wire_width=0.5,                        # placeholder — overridden by tensors
        R_via=FIXED_CONSTANTS["R_via"],
        C_decap=1e-10,                          # placeholder — overridden
        freq=FIXED_CONSTANTS["freq"],
        loads=loads,
    )
    data = to_hetero_data(g)

    def _strap_attr(n_edges: int, R: torch.Tensor) -> torch.Tensor:
        a = torch.zeros(n_edges, EDGE_ATTR_DIM)
        a[:, R_COL] = R
        return a

    R_top = FIXED_CONSTANTS["Rsheet_top"] * g.pitch_top / wire_width
    R_bot = FIXED_CONSTANTS["Rsheet_bot"] * g.pitch_bot / wire_width
    R_via_t = torch.tensor(float(FIXED_CONSTANTS["R_via"]))

    n_top_strap = data["mesh_top", "strap", "mesh_top"].edge_index.size(1)
    n_bot_strap = data["mesh_bot", "strap", "mesh_bot"].edge_index.size(1)
    n_via = data["mesh_top", "via", "mesh_bot"].edge_index.size(1)
    n_decap = data["mesh_bot", "decap", "mesh_bot"].edge_index.size(1)
    n_load = data["mesh_bot", "load", "mesh_bot"].edge_index.size(1)

    data["mesh_top", "strap", "mesh_top"].edge_attr = _strap_attr(n_top_strap, R_top)
    data["mesh_bot", "strap", "mesh_bot"].edge_attr = _strap_attr(n_bot_strap, R_bot)
    data["mesh_top", "via", "mesh_bot"].edge_attr = _strap_attr(n_via, R_via_t)
    data["mesh_bot", "via", "mesh_top"].edge_attr = _strap_attr(n_via, R_via_t)

    decap_attr = torch.zeros(n_decap, EDGE_ATTR_DIM)
    decap_attr[:, C_COL] = C_decap
    data["mesh_bot", "decap", "mesh_bot"].edge_attr = decap_attr

    load_attr = torch.zeros(n_load, EDGE_ATTR_DIM)
    load_attr[:, I_COL] = FIXED_I_PEAK
    load_attr[:, F_COL] = FIXED_FREQ
    load_attr[:, D_COL] = FIXED_DUTY
    load_attr[:, P_COL] = FIXED_PHASE
    data["mesh_bot", "load", "mesh_bot"].edge_attr = load_attr

    # Match the trained loader's y attribute so PyG batching is clean.
    data["y"] = torch.zeros(n_loads, dtype=torch.float32)
    return Batch.from_data_list([data])


# -----------------------------------------------------------------------------
# Loss: hinge-spec + cost
# -----------------------------------------------------------------------------

def cost(wire_width: torch.Tensor, C_decap: torch.Tensor) -> torch.Tensor:
    """Normalised cost ∈ [~0, ~1]. Wider wire and bigger cap both cost more."""
    ww_p = GLOBAL_RANGES.by_name("wire_width")
    cd_p = GLOBAL_RANGES.by_name("C_decap")
    ww_frac = (wire_width - ww_p.lo) / (ww_p.hi - ww_p.lo)
    cd_frac = (torch.log10(C_decap) - math.log10(cd_p.lo)) / (
        math.log10(cd_p.hi) - math.log10(cd_p.lo)
    )
    return ww_frac + cd_frac


def design_loss(
    model: PDNDroopRegressor,
    z: torch.Tensor,
    n_top: int,
    target_v: float,
    lambda_cost: float,
) -> tuple[torch.Tensor, dict]:
    wire_width, C_decap = decode(z)
    batch = build_diff_data(n_top, wire_width, C_decap)
    pred_log = model(batch)                            # [n_loads_in_batch]
    pred_v = 10.0 ** pred_log
    worst = pred_v.max()
    # Relative-fraction-over-spec hinge: 0 when feasible, grows quadratically
    # past the spec boundary. Scale-invariant in droop magnitude, so
    # ``lambda_cost`` can be a number like ``0.01`` without dimensional
    # gymnastics.
    rel_over = torch.relu(worst / target_v - 1.0)
    spec_violation = rel_over ** 2
    c = cost(wire_width, C_decap)
    loss = spec_violation + lambda_cost * c
    return loss, {
        "wire_width": float(wire_width.detach()),
        "C_decap":    float(C_decap.detach()),
        "worst_pred": float(worst.detach()),
        "spec_violation": float(spec_violation.detach()),
        "cost":      float(c.detach()),
        "loss":      float(loss.detach()),
    }


def optimize(
    model: PDNDroopRegressor,
    n_top: int,
    target_v: float,
    lambda_cost: float,
    n_steps: int,
    lr: float,
    seed: int,
) -> dict:
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(2, requires_grad=True)
    z.data = torch.randn(2, generator=g) * 0.5  # start near box centre
    opt = torch.optim.Adam([z], lr=lr)
    trace = []
    for step in range(n_steps):
        loss, info = design_loss(model, z, n_top, target_v, lambda_cost)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            trace.append({"step": step, **info})
    return {"z": z.detach().tolist(), "trace": trace}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", type=Path,
        default=Path("checkpoints/droop_v5_conductance.pt"),
    )
    ap.add_argument("--target-mV", type=float, default=0.10)
    ap.add_argument("--lambda-cost", type=float, default=1e-3)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=7)
    args = ap.parse_args()

    target_v = args.target_mV * 1e-3
    model = PDNDroopRegressor(
        EncoderConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            conv_type="admittance",
            drop_edge_p=0.0,
        ),
        target_space="log",
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Target: worst-load droop ≤ {args.target_mV:.3f} mV")
    print(f"λ_cost = {args.lambda_cost}, steps = {args.n_steps}, lr = {args.lr}")
    print()
    print(f"{'n_top':>5} | {'opt ww':>10} | {'opt C_decap':>12} | "
          f"{'pred drp':>10} | {'sim drp':>10} | spec  ")
    print("-" * 75)
    for n_top in ALL_N_TOP:
        out = optimize(
            model, int(n_top), target_v,
            lambda_cost=args.lambda_cost,
            n_steps=args.n_steps,
            lr=args.lr,
            seed=args.seed,
        )
        last = out["trace"][-1]
        ww_opt = last["wire_width"]
        cd_opt = last["C_decap"]
        pred_mV = last["worst_pred"] * 1e3
        loads_arr = np.tile(
            np.array([[FIXED_I_PEAK, FIXED_FREQ, FIXED_DUTY, FIXED_PHASE]]),
            (build_regular_pdn(n_top=int(n_top)).n_loads, 1),
        )
        p = dict(FIXED_CONSTANTS)
        p.update(wire_width=ww_opt, C_decap=cd_opt, n_top=int(n_top), loads=loads_arr)
        truth = float(run_one(p, cfg=SimConfig())["peak_droop_loads"].max()) * 1e3
        margin = "✓" if truth <= args.target_mV else "✗"
        print(
            f"{n_top:>5} | {ww_opt:>10.4f} | {cd_opt:>12.2e} | "
            f"{pred_mV:>9.4f} mV | {truth:>9.4f} mV | {margin}"
        )

    print()
    print("Notes:")
    print("  * 'opt' = gradient-optimised design point in the latent space.")
    print("  * 'pred drp' = surrogate prediction at the recovered design;")
    print("    'sim drp' = simulator ground truth.")
    print("  * '✓' means the simulator confirms the spec; '✗' means the")
    print("    surrogate was over-optimistic and the recovered design")
    print("    actually violates the spec.")


if __name__ == "__main__":
    main()
