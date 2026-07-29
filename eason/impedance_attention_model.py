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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_NODE_FEATURES = 10          # tools.impedance_factors.node_features


@dataclass
class ImpAttnConfig:
    hidden_dim: int = 64
    heads: int = 4
    d_qk: int = 4
    d_v: int = 32
    n_freq: int = 3           # -> C = 1 + 4*(n_freq-1) real channels
    m_factor: int = 16
    content: bool = True      # ablation: content-only  (q.k)
    impedance: bool = True    # ablation: impedance-only (p.s)
    # --- nonlinear kernel score -------------------------------------
    score: str = "bilinear"   # "bilinear" | "kernel"
    n_scales: int = 3         # learnable gammas per head
    kernel_feature: str = "rff"   # "rff" | "taylor"
    n_rff: int = 128          # random Fourier features per (head, scale)
    taylor_k: int = 2         # taylor only; unusable past gamma ~0.1 (see below)


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


class KernelImpedanceAttention(ImpedanceAttention):
    """Bilinear score plus a learned multi-scale kernel of the impedance.

    The bilinear term ``(q.k)(p.s)`` is the exact-physics anchor and is
    kept. Added on top is a Gaussian kernel of the *effective-resistance*
    distance, at several learnable scales per head:

        K_ij = sum_t w_ht * exp(-gamma_ht * ||F_i - F_j||^2)

    where ``F`` is the symmetric DC factor, so ``||F_i - F_j||^2 = R_eff``
    exactly (``tools.impedance_factors.dc_symmetric_factor``).

    Two things this buys that a bilinear score cannot:

    * **Reweighting.** Large ``gamma`` concentrates a node's droop on
      electrically nearby sources — a soft near/far split with no
      neighbour list, so the layer stays purely global and one-shot.
    * **Rank.** ``exp`` of a low-rank quantity is high-rank, so the
      operator expresses interaction patterns the rank-m factors cannot.
      (Note this raises the rank of the *operator*; it cannot change the
      per-pair ordering induced by a single scalar score — that requires
      the multivariate route, i.e. phi/psi seeing per-node self-impedance.)

    O(N) is preserved by random Fourier features (Rahimi-Recht): with
    ``w ~ N(0, 2*gamma*I)`` and ``z(u) = sqrt(2/D) cos(w.u + b)``,
    ``<z(u), z(v)> ~= exp(-gamma||u-v||^2)``, so the score stays an inner
    product and the existing Kronecker cache applies unchanged. ``gamma``
    remains learnable because the frequency is ``sqrt(2*gamma) * w0`` with
    ``w0`` frozen.

    **Why RFF and not a Taylor expansion of the exponential.** The Taylor
    route is an exact inner product, but it diverges once
    ``2*gamma*z_ij > 1``: measured on this track it gives 8 % error at
    gamma=0.1 (where the kernel barely discriminates, spread 3.7x) and
    59 % at gamma=0.3 (spread 50x), with k=3 no better than k=2. It cannot
    reach the sharp-locality regime that motivates the kernel at all. RFF
    error is instead flat in gamma (~0.07 at D=512 for gamma=0.1, ~0.16 at
    gamma=100) and is controlled by D alone. ``kernel_feature="taylor"``
    is retained only for the exactness test.

    Basis invariance: the kernel is a function of ``R_eff``, an invariant.
    The RFF directions are random-but-frozen per model, exactly like the
    factor probes, and enter only through an inner product that
    approximates the invariant kernel.
    """

    def __init__(self, cfg: ImpAttnConfig, n_ch: int) -> None:
        super().__init__(cfg, n_ch)
        h, H, m = cfg.hidden_dim, cfg.heads, cfg.m_factor
        self.T, self.D = cfg.n_scales, cfg.n_rff
        self.feature = cfg.kernel_feature
        # scales spread over decades so heads start at different localities
        self.log_gamma = nn.Parameter(
            torch.linspace(-1.0, 1.5, self.T).repeat(H, 1))
        self.kw = nn.Parameter(torch.zeros(H, self.T))   # starts silent
        self.kphi = nn.Linear(h, H)
        self.kpsi = nn.Linear(h, H)
        nn.init.zeros_(self.kphi.weight); nn.init.zeros_(self.kpsi.weight)
        nn.init.ones_(self.kphi.bias); nn.init.ones_(self.kpsi.bias)
        # fixed random Fourier basis; gamma stays learnable because the
        # frequency is sqrt(2*gamma)*w0 with w0 ~ N(0, I) frozen
        g = torch.Generator().manual_seed(0)
        self.register_buffer("w0", torch.randn(self.D, m, generator=g))
        self.register_buffer("rb", 2 * np.pi * torch.rand(self.D, generator=g))
        self.register_buffer("tk", torch.tensor(float(cfg.taylor_k)))

    def _rff(self, f: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """z(u) with <z(u), z(v)> ~= exp(-gamma ||u - v||^2).

        Bochner/Rahimi-Recht: w ~ N(0, 2*gamma*I), z = sqrt(2/D) cos(w.u + b).
        Unlike the Taylor route this is accurate at *any* gamma, which is
        what the sharp-locality regime needs (Taylor diverges once
        2*gamma*z_ij > 1 — measured 59 % error at gamma = 0.3).
        """
        proj = f @ (torch.sqrt(2 * gamma) * self.w0).T + self.rb
        return np.sqrt(2.0 / self.D) * torch.cos(proj)

    def _taylor(self, f: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Exact inner-product expansion of exp(2*beta*<p,s>), truncated.

        Kept for the exactness test only — it is an inner product by
        construction, but useless past gamma ~ 0.1 on this track.
        """
        N = f.shape[0]
        terms = [f.new_ones(N, 1), torch.sqrt(2 * beta) * f]
        if int(self.tk) >= 2:
            outer = (f.unsqueeze(-1) * f.unsqueeze(-2)).reshape(N, -1)
            terms.append(beta * torch.sqrt(f.new_tensor(2.0)) * outer)
        return torch.cat(terms, dim=-1)

    def _kernel_maps(self, hstate, fdc):
        """-> (Phi, Psi) [N, H, T*D] from the symmetric DC factor ``fdc``."""
        N, H = hstate.shape[0], self.cfg.heads
        kphi = self.kphi(hstate).view(N, H, 1)
        kpsi = self.kpsi(hstate).view(N, H, 1)
        gam = F.softplus(self.log_gamma)                        # [H, T]
        Phi, Psi = [], []
        for hh in range(H):
            for t in range(self.T):
                gm = gam[hh, t]
                if self.feature == "rff":
                    z = self._rff(fdc, gm)
                    gp = gs = z
                else:
                    zii = (fdc * fdc).sum(-1)
                    dec = torch.exp(-gm * zii).unsqueeze(-1)
                    gp = gs = self._taylor(fdc, gm) * dec
                Phi.append(gp * (self.kw[hh, t] * kphi[:, hh]))
                Psi.append(gs * kpsi[:, hh])
        D = Phi[0].shape[-1]
        Phi = torch.stack(Phi, 1).view(N, H, self.T * D)
        Psi = torch.stack(Psi, 1).view(N, H, self.T * D)
        return Phi, Psi

    def forward(self, hstate, p, s, naive: bool = False, fdc=None):
        N, H = hstate.shape[0], self.cfg.heads
        q = self.q(hstate).view(N, H, self.cfg.d_qk)
        k = self.k(hstate).view(N, H, self.cfg.d_qk)
        v = self.v(hstate).view(N, H, self.cfg.d_v)
        if not self.cfg.content:
            q = torch.ones_like(q) / self.cfg.d_qk ** 0.5
            k = torch.ones_like(k) / self.cfg.d_qk ** 0.5
        P, S = self._factor_terms(hstate, p, s)
        if fdc is None:
            raise ValueError("kernel score needs the symmetric DC factor "
                             "(tools.impedance_factors.dc_symmetric_factor)")
        Phi, Psi = self._kernel_maps(hstate, fdc)

        if naive:
            score = torch.einsum("ihd,jhd->hij", q, k)
            if P is not None:
                score = score * torch.einsum("ihd,jhd->hij", P, S)
            score = score + torch.einsum("ihd,jhd->hij", Phi, Psi)
            out = torch.einsum("hij,jhe->ihe", score, v)
        else:
            qt = (torch.einsum("ihd,ihc->ihdc", q, P).reshape(N, H, -1)
                  if P is not None else q)
            kt = (torch.einsum("jhd,jhc->jhdc", k, S).reshape(N, H, -1)
                  if S is not None else k)
            qt = torch.cat([qt, Phi], dim=-1)
            kt = torch.cat([kt, Psi], dim=-1)
            cache = torch.einsum("jhD,jhe->hDe", kt, v)
            out = torch.einsum("ihD,hDe->ihe", qt, cache)

        return self.norm(hstate + self.out(out.reshape(N, H * self.cfg.d_v)))


class ImpedanceAttentionRegressor(nn.Module):
    """input MLP -> one global attention layer -> per-load decoder.

    No pre- or post-attention graph convolutions: whether one global layer
    can replace depth is exactly the hypothesis under test.
    """

    def __init__(self, cfg: ImpAttnConfig | None = None,
                 target_space: str = "log", init_bias: float = 0.0,
                 n_ch: int | None = None) -> None:
        super().__init__()
        cfg = cfg or ImpAttnConfig()
        self.cfg = cfg
        # 1 channel at DC + 4 per non-zero frequency (see
        # tools.impedance_factors.channel_count). Pass n_ch explicitly if the
        # frequency grid does not start at DC.
        n_ch = n_ch if n_ch is not None else 1 + 4 * (cfg.n_freq - 1)
        self.n_ch = n_ch
        h = cfg.hidden_dim
        # +n_ch: PER-CHANNEL log self-impedance. This is the multivariate
        # enabler — with only the pair score z_ij available, any scalar
        # monotone function of it leaves the per-observer ordering over
        # sources unchanged (measured). Letting phi/psi see z_ii and z_jj
        # is what allows the score to reorder at all.
        self.encoder = nn.Sequential(
            nn.Linear(N_NODE_FEATURES + n_ch, h), nn.ReLU(), nn.Linear(h, h)
        )
        self.attn = (KernelImpedanceAttention(cfg, n_ch)
                     if cfg.score == "kernel" else ImpedanceAttention(cfg, n_ch))
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

    def embed(self, x, p, s, n_elec: int):
        """Normalise factors and run the input MLP -> (hstate, p, s).

        Exposed so probes can inspect the pre-attention state without
        re-implementing the feature construction (which has drifted once).
        """
        p, s = self.normalize_factors(p, s, n_elec)
        selfz = (p * s).sum(-1).abs().clamp_min(1e-30).log10()     # [N, n_ch]
        return self.encoder(torch.cat([x, selfz], -1)), p, s

    def forward(self, x, p, s, n_elec: int, naive: bool = False, fdc=None):
        """x [N, F]; p, s [N, n_ch, m]; loads are rows [n_elec:].

        Returns one droop prediction per load node, shape [L].
        """
        hstate, p, s = self.embed(x, p, s, n_elec)
        if isinstance(self.attn, KernelImpedanceAttention):
            if fdc is not None:
                fdc = fdc / fdc.norm(dim=-1, keepdim=True).mean().clamp_min(1e-30)
            hstate = self.attn(hstate, p, s, naive=naive, fdc=fdc)
        else:
            hstate = self.attn(hstate, p, s, naive=naive)
        return self.decoder(hstate[n_elec:]).squeeze(-1)
