"""One-shot global impedance-factorized attention (no local message passing).

Replaces depth-dependent propagation with a single global attention layer
whose score is the product of a learned content term and a physical
impedance term:

    a_(i<-j) = (q_i . k_j) * (p_i . s_j)          [no softmax: sources superpose]
    out_i    = sum_j a_(i<-j) v_j

Because the score is a product of two inner products it is one inner
product in the Kronecker space, so the sum is evaluated through a cache
that is independent of the node count:

    q~_i = q_i (x) p_i ,  k~_j = k_j (x) s_j
    cache = sum_j k~_j v_j^T                      [H, d_qk*C*m, d_v]
    out_i = q~_i^T cache

No ``[N, N]`` tensor is built on the normal path (the naive path exists
only for the equivalence test, behind ``naive=True``).

Design decisions that are load-bearing (docs/IMPEDANCE_ATTENTION_DESIGN.md):

* **One shared encoder and one shared Q/K/V/phi/psi set for all node
  types.** Node type is an input feature; nothing is keyed on it.
* **Loads are nodes** and are the prediction site. Their impedance factor
  is the oriented terminal difference, which makes the exact DC
  superposition a special case of this layer.
* **Basis invariance:** factors enter only through the invariant scalars
  ``p_i^c . s_j^c``. Directionality comes from independent ``Q/K`` and
  per-channel gains ``phi != psi`` learned from features — never from a
  learned map on raw factor coordinates, which would destroy transfer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

N_NODE_FEATURES = 10          # tools.impedance_factors.node_features


@dataclass
class ImpAttnConfig:
    hidden_dim: int = 64
    heads: int = 4
    d_qk: int = 4
    d_v: int = 32
    n_freq: int = 3           # -> C = 2 * n_freq real channels
    m_factor: int = 16
    content: bool = True      # ablation: content-only  (q.k)
    impedance: bool = True    # ablation: impedance-only (p.s)


class ImpedanceAttention(nn.Module):
    """Single unnormalized factorized global attention layer."""

    def __init__(self, cfg: ImpAttnConfig, n_ch: int) -> None:
        super().__init__()
        self.cfg = cfg
        h, H = cfg.hidden_dim, cfg.heads
        self.n_ch = n_ch
        self.q = nn.Linear(h, H * cfg.d_qk)
        self.k = nn.Linear(h, H * cfg.d_qk)
        self.v = nn.Linear(h, H * cfg.d_v)
        self.phi = nn.Linear(h, H * n_ch)      # observer channel gains
        self.psi = nn.Linear(h, H * n_ch)      # source channel gains
        self.out = nn.Linear(H * cfg.d_v, h)
        self.norm = nn.LayerNorm(h)

        # Physics-flavoured init: head 0 reads the DC-real channel with unit
        # gain (recovering plain superposition), other heads start quiet.
        nn.init.zeros_(self.phi.weight); nn.init.zeros_(self.psi.weight)
        with torch.no_grad():
            pb = torch.zeros(H, n_ch); pb[0, 0] = 1.0
            self.phi.bias.copy_(pb.reshape(-1)); self.psi.bias.copy_(pb.reshape(-1))

    def _factor_terms(self, hstate, p, s):
        """-> P, S of shape [N, H, n_ch*m] (or None if impedance is off)."""
        if not self.cfg.impedance:
            return None, None
        N, H = hstate.shape[0], self.cfg.heads
        phi = self.phi(hstate).view(N, H, self.n_ch, 1)
        psi = self.psi(hstate).view(N, H, self.n_ch, 1)
        P = (phi * p.unsqueeze(1)).reshape(N, H, -1)
        S = (psi * s.unsqueeze(1)).reshape(N, H, -1)
        return P, S

    def forward(self, hstate, p, s, naive: bool = False):
        """hstate [N, h]; p, s [N, n_ch, m] -> [N, h]."""
        N, H = hstate.shape[0], self.cfg.heads
        q = self.q(hstate).view(N, H, self.cfg.d_qk)
        k = self.k(hstate).view(N, H, self.cfg.d_qk)
        v = self.v(hstate).view(N, H, self.cfg.d_v)
        if not self.cfg.content:                      # impedance-only ablation
            q = torch.ones_like(q) / self.cfg.d_qk ** 0.5
            k = torch.ones_like(k) / self.cfg.d_qk ** 0.5
        P, S = self._factor_terms(hstate, p, s)

        if naive:
            # Explicit [N, N] scores — TEST PATH ONLY, never used in training.
            score = torch.einsum("ihd,jhd->hij", q, k)
            if P is not None:
                score = score * torch.einsum("ihd,jhd->hij", P, S)
            out = torch.einsum("hij,jhe->ihe", score, v)
        else:
            if P is None:
                cache = torch.einsum("jhd,jhe->hde", k, v)
                out = torch.einsum("ihd,hde->ihe", q, cache)
            else:
                # q~ = q (x) P without ever forming [N, N]
                qt = torch.einsum("ihd,ihc->ihdc", q, P).reshape(N, H, -1)
                kt = torch.einsum("jhd,jhc->jhdc", k, S).reshape(N, H, -1)
                cache = torch.einsum("jhD,jhe->hDe", kt, v)
                out = torch.einsum("ihD,hDe->ihe", qt, cache)

        return self.norm(hstate + self.out(out.reshape(N, H * self.cfg.d_v)))


class ImpedanceAttentionRegressor(nn.Module):
    """input MLP -> one global attention layer -> per-load decoder.

    No pre- or post-attention graph convolutions: whether one global layer
    can replace depth is exactly the hypothesis under test.
    """

    def __init__(self, cfg: ImpAttnConfig | None = None,
                 target_space: str = "log", init_bias: float = 0.0) -> None:
        super().__init__()
        cfg = cfg or ImpAttnConfig()
        self.cfg = cfg
        n_ch = 2 * cfg.n_freq
        h = cfg.hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(N_NODE_FEATURES + 1, h), nn.ReLU(), nn.Linear(h, h)
        )
        self.attn = ImpedanceAttention(cfg, n_ch)
        self.decoder = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.target_space = target_space
        # Start at the mean-predictor baseline, not at random and not at
        # zero (stage-1 lesson: a head that starts below its own physics
        # floor spends training climbing back). With a log10 target,
        # ``init_bias`` should be the mean log10(droop) of the training set —
        # zero would mean "predict 1 volt".
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.constant_(self.decoder[-1].bias, init_bias)

    @staticmethod
    def normalize_factors(p, s, n_elec: int, eps: float = 1e-30):
        """Scale factors to O(1) using a per-graph *invariant* scalar.

        Uses the mean self-impedance of the load nodes (an inner product, so
        basis-invariant) and splits it evenly between p and s. Differentiable.
        """
        if p.shape[0] <= n_elec:
            return p, s
        diag = (p[n_elec:] * s[n_elec:]).sum(-1).abs().mean().clamp_min(eps)
        scale = diag.sqrt()
        return p / scale, s / scale

    def forward(self, x, p, s, n_elec: int, naive: bool = False):
        """x [N, F]; p, s [N, n_ch, m]; loads are rows [n_elec:].

        Returns one droop prediction per load node, shape [L].
        """
        p, s = self.normalize_factors(p, s, n_elec)
        # log self-impedance is an invariant scalar and a strong scale cue
        selfz = (p * s).sum(-1).abs().sum(-1, keepdim=True).clamp_min(1e-30).log10()
        hstate = self.encoder(torch.cat([x, selfz], -1))
        hstate = self.attn(hstate, p, s, naive=naive)
        return self.decoder(hstate[n_elec:]).squeeze(-1)
