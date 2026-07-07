"""Stage-0 diagnostic: superposition attention over impedance sketches.

Physics anchor: DC droop obeys ``droop_i = Σ_j Z_ij I_j`` and the sketch
satisfies ``r_i · r_j ≈ Z_ij`` — so the far-field term is *literally*
linear attention with the sketch as feature map. The learnable part
modulates that kernel with node features (frequency response, net
structure, observer sensitivity), it does not replace it.

Attention (per head c):

    score_ij = φ_c(h_i) · ψ_c(h_j) · (r_i · r_j)   [geometry-modulated]
             + α_c(h_i) · β_c(h_j)                  [geometry-free bias]
    out_i    = Σ_{j ∈ loads} score_ij · v_j         [NO softmax: superposition]

Both terms factorize, so the whole thing is two cached tensors per head:

    S_c = Σ_j ψ_c(h_j) · (r_j ⊗ v_j)   ∈ R^{m×d}    (the "KV cache")
    t_c = Σ_j β_c(h_j) · v_j           ∈ R^{d}
    out_i = φ_c(h_i) · r_iᵀ S_c + α_c(h_i) · t_c    → O(N·m·d) total

Basis invariance: the sketch enters ONLY through r_i·r_j and ‖r_i‖²
(orthogonal-invariant), never through a learned map of r — the random
projection basis differs per grid, so anything else cannot transfer.

Local sharpness comes from the same gated EdgeConv stack as before,
sandwiching the attention block.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.ibmpg_patches import NODE_FEATURE_DIM

from .convs import EdgeConvGated

RELS = ("R", "C", "L")
_KIND = {"R": "resistor", "C": "capacitor", "L": "resistor"}


class ConvBlock(nn.Module):
    def __init__(self, h: int) -> None:
        super().__init__()
        self.convs = nn.ModuleDict({r: EdgeConvGated(h, kind=_KIND[r]) for r in RELS})
        self.norm = nn.LayerNorm(h)

    def forward(self, x, data):
        out = torch.zeros_like(x)
        for r in RELS:
            et = ("node", r, "node")
            ei = data[et].edge_index
            if ei.numel():
                out = out + self.convs[r](x, ei, data[et].edge_attr)
        return self.norm(F.relu(out) + x)


class SuperpositionAttention(nn.Module):
    """Unnormalized bipartite attention from load nodes, kernel = r_i·r_j."""

    def __init__(self, h: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.phi = nn.Linear(h, heads)     # observer modulation (geometry term)
        self.psi = nn.Linear(h, heads)     # source modulation   (geometry term)
        self.alpha = nn.Linear(h, heads)   # observer bias term
        self.beta = nn.Linear(h, heads)    # source bias term
        self.v = nn.Linear(h, h)
        self.out = nn.Linear(heads * h, h)
        self.norm = nn.LayerNorm(h)
        # physics init: head 0 starts as the raw DC superposition kernel
        # (φ=ψ=1), other heads start silent; bias terms start silent.
        nn.init.zeros_(self.phi.weight); nn.init.zeros_(self.psi.weight)
        nn.init.zeros_(self.alpha.weight); nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.alpha.bias); nn.init.zeros_(self.beta.bias)
        with torch.no_grad():
            self.phi.bias.copy_(torch.tensor([1.0] + [0.0] * (heads - 1)))
            self.psi.bias.copy_(torch.tensor([1.0] + [0.0] * (heads - 1)))

    def forward(self, x, r, load_mask):
        # x: [N, h]; r: [N, m]; load_mask: [N] bool
        xl, rl = x[load_mask], r[load_mask]              # [J, ·]
        v = self.v(xl)                                   # [J, h]
        phi, psi = self.phi(x), self.psi(xl)             # [N, H], [J, H]
        alpha, beta = self.alpha(x), self.beta(xl)
        outs = []
        for c in range(self.heads):
            S = (rl * psi[:, c:c + 1]).T @ v             # [m, h] KV cache
            t = (v * beta[:, c:c + 1]).sum(0)            # [h]
            outs.append(phi[:, c:c + 1] * (r @ S) + alpha[:, c:c + 1] * t)
        return self.norm(self.out(torch.cat(outs, dim=-1)) + x)


class IBMAttnRegressor(nn.Module):
    """conv ×k → superposition attention → conv ×k → per-node head."""

    def __init__(self, hidden_dim: int = 64, n_conv: int = 2, heads: int = 4) -> None:
        super().__init__()
        h = hidden_dim
        # +1: log ‖r_i‖² (self-impedance, basis-invariant)
        self.proj = nn.Linear(NODE_FEATURE_DIM + 1, h)
        self.pre = nn.ModuleList([ConvBlock(h) for _ in range(n_conv)])
        self.attn = SuperpositionAttention(h, heads)
        self.post = nn.ModuleList([ConvBlock(h) for _ in range(n_conv)])
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, data) -> torch.Tensor:
        r = data["node"].sketch                                   # [N, m]
        self_z = torch.log10((r * r).sum(-1, keepdim=True) + 1e-9)
        x = self.proj(torch.cat([data["node"].x, self_z], dim=-1))
        for blk in self.pre:
            x = blk(x, data)
        x = self.attn(x, r, data["node"].load_mask)
        for blk in self.post:
            x = blk(x, data)
        return self.head(x).squeeze(-1)
