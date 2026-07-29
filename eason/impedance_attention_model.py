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

from tools.impedance_factors import (invariant_channel_count,
                                     invariant_channels,
                                     invariant_factor_tensors)

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
    score: str = "bilinear"   # "bilinear"|"kernel"|"dynamic_kernel"|"simple"
    n_scales: int = 3         # learnable gammas per head
    kernel_feature: str = "rff"   # "rff" | "taylor"
    n_rff: int = 128          # random Fourier features per (head, scale)
    taylor_k: int = 2         # taylor only; unusable past gamma ~0.1 (see below)
    # --- unified dynamic-impedance kernel ---------------------------
    max_degree: int = 2       # polynomial degree in each invariant channel
    # Consume BASIS-INVARIANT impedance channels (Z at DC, Re Z / Im Z per
    # frequency) instead of the four raw real channels per frequency. The
    # raw AC channels are not individually basis-invariant, and a model
    # built on them has ~100 % probe-seed dependence in its wire and decap
    # sensitivities even at exact rank (docs/FACTOR_STABILITY_AUDIT.md).
    # False reproduces every pre-back-port checkpoint.
    invariant: bool = True
    # Divide each frequency's factors by its own invariant impedance scale
    # s_w = sqrt(mean_loads |Z_ii(w)|^2), so no frequency dominates the score
    # for purely numerical reasons. Per-node, O(N), differentiable, and
    # factorization-preserving (see _normalize_per_frequency).
    freq_norm: bool = True
    # "frob" = RMS entry of Z over all pairs via two [d,d] Grams (O(N d^2),
    # never [N,N]); "diag" = mean squared self-impedance over load nodes.
    # Measured: "diag" fails to align the off-diagonal ranges it targets,
    # because diagonal impedance falls faster with omega than off-diagonal
    # does. See docs/analysis/freq_norm_audit.json.
    freq_norm_mode: str = "frob"
    # Feed per-node incident conductance and capacitance (log10). Until this
    # existed the model could not see R or C except through the impedance
    # factors. False reproduces pre-existing checkpoints.
    local_rc: bool = False


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
        """-> (Phi, Psi) [N, H, T*D] from the symmetric DC factor ``fdc``.

        The RFF path is evaluated for all (head, scale) pairs in one matmul.
        Every pair shares the frozen basis ``w0`` and differs only by the
        scalar ``sqrt(2*gamma)``, so the H*T projections are one
        ``[N, m] @ [m, H*T*D]`` product rather than H*T small ones.

        Measured end-to-end (fwd+bwd, 4 heads, 3 scales) vs the reference
        loop: 1.66x at N=72/D=128, 1.32x at N=390/D=128, and a wash at
        D=512 (1.06x at N=390, 0.92x at N=270) where the per-pair matmuls
        are already large enough to saturate. It is kept because the small
        anchors dominate the sample count, but it is not a large win.
        ``_kernel_maps_loop`` is the readable reference; check 8 asserts the
        two agree (forward bit-identical, gradients to float32 rounding).
        """
        N, H, T, D = hstate.shape[0], self.cfg.heads, self.T, self.D
        gam = F.softplus(self.log_gamma)                        # [H, T]
        if self.feature != "rff":
            return self._kernel_maps_loop(hstate, fdc, gam)
        kphi = self.kphi(hstate).view(N, H, 1, 1)
        kpsi = self.kpsi(hstate).view(N, H, 1, 1)
        # w[h,t] = sqrt(2*gamma_ht) * w0  ->  fold the scalar into the input
        # projection so all H*T live in a single [m, H*T*D] matrix
        wht = (torch.sqrt(2 * gam).view(H, T, 1, 1) * self.w0)   # [H,T,D,m]
        proj = torch.einsum("nm,htdm->nhtd", fdc, wht) + self.rb
        z = np.sqrt(2.0 / D) * torch.cos(proj)                   # [N,H,T,D]
        Phi = z * (self.kw.view(1, H, T, 1) * kphi)
        Psi = z * kpsi
        return Phi.reshape(N, H, T * D), Psi.reshape(N, H, T * D)

    def _kernel_maps_loop(self, hstate, fdc, gam=None):
        """Reference implementation of :meth:`_kernel_maps` (also the Taylor
        path, which does not share a basis across scales)."""
        N, H = hstate.shape[0], self.cfg.heads
        kphi = self.kphi(hstate).view(N, H, 1)
        kpsi = self.kpsi(hstate).view(N, H, 1)
        if gam is None:
            gam = F.softplus(self.log_gamma)                    # [H, T]
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


class SimpleDistanceAttention(nn.Module):
    """The cheapest defensible score: one pairwise physical scalar.

        A_h(i<-j) = phi_h(i) . psi_h(j)  over  [ 1 , RFF(u_i) ]

    where ``u`` is the symmetric DC factor, so the RFF block approximates a
    Gaussian kernel of **DC effective resistance** — one scalar per pair,
    basis-invariant, and reciprocal and well-conditioned at every rank
    (measured 6e-16 reciprocity and probe-seed gradient rank ~1.0, versus
    0.15-0.24 and ~0.7 for the AC channels).

    Everything frequency- and time-dependent is left to be **learned** from
    local component values (``local_rc_features``) and load waveform
    features, rather than handed over as a solved frequency sweep. The
    constant block is the one content term.

    This exists to falsify the multi-frequency stack: if it is not
    materially worse on forward accuracy, the extra machinery is not earning
    its complexity.

    Known limitation, stated rather than hidden: ``d(R_eff)/d(C) = 0``
    exactly, so any capacitance sensitivity must come through the local-C
    node feature and the learned dynamics. Whether that suffices is the
    experiment.
    """

    def __init__(self, cfg: ImpAttnConfig, m: int) -> None:
        super().__init__()
        self.cfg = cfg
        h, H, T, D = cfg.hidden_dim, cfg.heads, cfg.n_scales, cfg.n_rff
        self.T, self.D = T, D
        self.n_blocks = 1 + T                      # content + one per scale
        self.register_buffer("block_of", torch.cat(
            [torch.zeros(1, dtype=torch.long),
             torch.repeat_interleave(torch.arange(1, T + 1),
                                     torch.full((T,), D))]))
        self.F = 1 + T * D
        self.log_gamma = nn.Parameter(torch.linspace(-1.0, 1.5, T).repeat(H, 1))
        self.phi = nn.Linear(h, H * self.n_blocks)
        self.psi = nn.Linear(h, H * self.n_blocks)
        self.v = nn.Linear(h, H * cfg.d_v)
        self.out = nn.Linear(H * cfg.d_v, h)
        self.norm = nn.LayerNorm(h)
        g = torch.Generator().manual_seed(0)
        self.register_buffer("w0", torch.randn(D, m, generator=g))
        self.register_buffer("rb", 2 * np.pi * torch.rand(D, generator=g))
        nn.init.zeros_(self.phi.weight); nn.init.ones_(self.phi.bias)
        nn.init.zeros_(self.psi.weight); nn.init.ones_(self.psi.bias)

    def _maps(self, hstate, fdc):
        N, H, T, D = hstate.shape[0], self.cfg.heads, self.T, self.D
        gam = F.softplus(self.log_gamma)                       # [H, T]
        wht = torch.sqrt(2 * gam).view(H, T, 1, 1) * self.w0   # [H,T,D,m]
        z = np.sqrt(2.0 / D) * torch.cos(
            torch.einsum("nm,htdm->nhtd", fdc, wht) + self.rb)
        U = torch.cat([z.new_ones(N, H, 1), z.reshape(N, H, T * D)], -1)
        gphi = self.phi(hstate).view(N, H, self.n_blocks)
        gpsi = self.psi(hstate).view(N, H, self.n_blocks)
        return U * gphi[..., self.block_of], U * gpsi[..., self.block_of]

    def forward(self, hstate, fdc, naive: bool = False):
        N, H = hstate.shape[0], self.cfg.heads
        v = self.v(hstate).view(N, H, self.cfg.d_v)
        Phi, Psi = self._maps(hstate, fdc)
        if naive:
            score = torch.einsum("ihd,jhd->hij", Phi, Psi)     # TEST PATH ONLY
            out = torch.einsum("hij,jhe->ihe", score, v)
        else:
            cache = torch.einsum("jhD,jhe->hDe", Psi, v)
            out = torch.einsum("ihD,hDe->ihe", Phi, cache)
        return self.norm(hstate + self.out(out.reshape(N, H * self.cfg.d_v)))


def _sym_square(a: torch.Tensor) -> torch.Tensor:
    """Symmetric tensor square with <sym(a), sym(b)> == <a, b>**2 exactly.

    Upper triangle only, off-diagonals weighted sqrt(2):
        sum_{i<j} 2 a_i a_j b_i b_j + sum_i a_i^2 b_i^2 = (sum_i a_i b_i)^2
    Width d(d+1)/2 instead of d^2, and it is still an inner product, so the
    O(N) contraction is unaffected.
    """
    d = a.shape[-1]
    iu, ju = torch.triu_indices(d, d, device=a.device)
    w = torch.where(iu == ju, 1.0, np.sqrt(2.0)).to(a.dtype)
    return a[..., iu] * a[..., ju] * w


class DynamicKernelAttention(nn.Module):
    """One unified multi-head dynamic-impedance kernel.

        A_h(i<-j) = Phi_h(i) . Psi_h(j)
                  = sum_{c,k} alpha_h[c,k] phi_h[c,k](i) psi_h[c,k](j) (z^c_ij)^k

    ``c`` runs over the **basis-invariant** impedance channels — Z at DC and
    Re Z / Im Z at each non-zero frequency — and ``k`` over polynomial
    degrees up to ``cfg.max_degree``. There is no separate "DC kernel" and
    no separate bilinear term: degree 1 at the DC channel *is* the old
    bilinear score, degree 0 is a pure content score, and everything else is
    new. Which combination a head uses is learned, not assigned.

    Why invariant channels and not the raw ones. ``impedance_factors`` emits
    (re,re), (im,im), (re,im), (im,re) per frequency, and those four are
    **not** individually basis-invariant — measured at exact rank, rebuilding
    from a different probe seed moves them by 0.19-0.76 relative while DC
    moves by 1.4e-15. Only ``Re Z = rr - ii`` and ``Im Z = ri + ir`` are
    stable. Any per-raw-channel gain therefore learns a quantity that is an
    artifact of the probe draw. Regrouping first (``invariant_channels``)
    costs width 2m per AC channel and fixes it.

    Directionality: ``phi`` and ``psi`` are separate maps of the node
    embedding, so A_h(i<-j) != A_h(j<-i) even though every underlying
    impedance channel is reciprocal. No asymmetry is injected into the
    physics.

    Capacitance path: the AC channels depend on C, so d(score)/dC is
    structurally non-zero — unlike the DC-only Gaussian kernel it replaces,
    whose derivative to decap is exactly zero.

    Factorization: every term is an inner product of per-node features, so
    the Kronecker cache applies unchanged and no [N, N] tensor is built.
    """

    def __init__(self, cfg: ImpAttnConfig, dims: list[int]) -> None:
        super().__init__()
        self.cfg = cfg
        h, H, K = cfg.hidden_dim, cfg.heads, cfg.max_degree
        self.n_ch, self.K = len(dims), K
        # feature width of each (channel, degree) block
        widths, self._block_index = [], []
        for ci, d in enumerate(dims):
            for k in range(K + 1):
                widths.append(1 if k == 0 else (d if k == 1 else d * (d + 1) // 2))
                self._block_index.append((ci, k))    # (channel, degree)
        self.widths = widths
        self.n_blocks = len(widths)
        self.register_buffer(
            "block_of", torch.repeat_interleave(
                torch.arange(self.n_blocks), torch.tensor(widths)))
        self.F = int(sum(widths))

        self.phi = nn.Linear(h, H * self.n_blocks)   # receiver gains
        self.psi = nn.Linear(h, H * self.n_blocks)   # sender gains
        self.v = nn.Linear(h, H * cfg.d_v)
        self.out = nn.Linear(H * cfg.d_v, h)
        self.norm = nn.LayerNorm(h)

        # Signed per-head mixture over (channel, degree), initialised AT THE
        # PHYSICS rather than at arbitrary scales:
        #   * DC, degree 1  -> 1.0   = ordinary superposition, droop = sum Z_ij I_j
        #   * Re/Im, degree 2 -> equal per frequency = |Z(w)|^2, the magnitude
        #     of the influence coefficient across several frequencies
        #   * everything else small
        # This is a STARTING POINT, not a constraint: alpha is fully
        # learnable and signed, so training moves off it freely. Comparing a
        # trained alpha against this pattern is how we read out which
        # frequency mixture the data actually wanted.
        #
        # Heads all start on the same pattern with different overall gains,
        # so symmetry is broken without assigning any head a role.
        base = torch.full((self.n_blocks,), 0.05)
        for b, (c, k) in enumerate(self._block_index):
            if c == 0 and k == 1:
                base[b] = 1.0                      # DC superposition
            elif c > 0 and k == 2:
                base[b] = 1.0                      # |Z(w)|^2, Re/Im tied
        scales = torch.logspace(-0.5, 0.0, H).view(H, 1)
        self.alpha = nn.Parameter(
            scales * base.view(1, -1)
            + 0.02 * torch.randn(H, self.n_blocks))
        nn.init.zeros_(self.phi.weight); nn.init.ones_(self.phi.bias)
        nn.init.zeros_(self.psi.weight); nn.init.ones_(self.psi.bias)

    def _maps(self, hstate, chans):
        """-> (Phi, Psi) [N, H, F] from invariant channel factor pairs."""
        N, H = hstate.shape[0], self.cfg.heads
        ua, ub = [], []
        for (A, B, _) in chans:
            for k in range(self.K + 1):
                if k == 0:
                    ua.append(A.new_ones(N, 1)); ub.append(B.new_ones(N, 1))
                elif k == 1:
                    ua.append(A); ub.append(B)
                else:
                    ua.append(_sym_square(A)); ub.append(_sym_square(B))
        U = torch.cat(ua, -1)                                  # [N, F]
        V = torch.cat(ub, -1)
        gphi = self.phi(hstate).view(N, H, self.n_blocks) * self.alpha
        gpsi = self.psi(hstate).view(N, H, self.n_blocks)
        Phi = U.unsqueeze(1) * gphi[..., self.block_of]
        Psi = V.unsqueeze(1) * gpsi[..., self.block_of]
        return Phi, Psi

    def forward(self, hstate, chans, naive: bool = False):
        N, H = hstate.shape[0], self.cfg.heads
        v = self.v(hstate).view(N, H, self.cfg.d_v)
        Phi, Psi = self._maps(hstate, chans)
        if naive:
            score = torch.einsum("ihd,jhd->hij", Phi, Psi)     # TEST PATH ONLY
            out = torch.einsum("hij,jhe->ihe", score, v)
        else:
            cache = torch.einsum("jhD,jhe->hDe", Psi, v)
            out = torch.einsum("ihD,hDe->ihe", Phi, cache)
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
        self._omegas_layout = [0.0] + [1.0] * (cfg.n_freq - 1)
        self.n_inv = invariant_channel_count(self._omegas_layout)
        if n_ch is None:
            n_ch = (self.n_inv if cfg.invariant
                    else 1 + 4 * (cfg.n_freq - 1))
        self.n_ch = n_ch
        h = cfg.hidden_dim
        # +n_ch: PER-CHANNEL log self-impedance. This is the multivariate
        # enabler — with only the pair score z_ij available, any scalar
        # monotone function of it leaves the per-observer ordering over
        # sources unchanged (measured). Letting phi/psi see z_ii and z_jj
        # is what allows the score to reorder at all.
        if cfg.score == "simple":
            n_selfz = 1                     # DC self-impedance only
        elif cfg.score == "dynamic_kernel":
            n_selfz = self.n_inv
        else:
            n_selfz = n_ch
        self.n_extra = 2 if cfg.local_rc else 0
        self.encoder = nn.Sequential(
            nn.Linear(N_NODE_FEATURES + self.n_extra + n_selfz, h),
            nn.ReLU(), nn.Linear(h, h)
        )
        if cfg.score == "simple":
            self.attn = SimpleDistanceAttention(cfg, cfg.m_factor)
        elif cfg.score == "dynamic_kernel":
            dims = ([cfg.m_factor]
                    + [2 * cfg.m_factor] * (self.n_inv - 1))
            self.attn = DynamicKernelAttention(cfg, dims)
        elif cfg.score == "kernel":
            self.attn = KernelImpedanceAttention(cfg, n_ch)
        else:
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

    def embed(self, x, p, s, n_elec: int):
        """Normalise factors and run the input MLP -> (hstate, p, s).

        Exposed so probes can inspect the pre-attention state without
        re-implementing the feature construction (which has drifted once).
        """
        if self.cfg.invariant:
            p, s = invariant_factor_tensors(p, s, self._omegas_layout)
        p, s = self.normalize_factors(p, s, n_elec)
        selfz = (p * s).sum(-1).abs().clamp_min(1e-30).log10()     # [N, n_ch]
        return self.encoder(torch.cat([x, selfz], -1)), p, s

    def embed_invariant(self, x, p, s, n_elec: int):
        """Dynamic-kernel embedding: regroup to invariant channels first.

        Self-impedance here is A_i . B_i on the INVARIANT channels, not
        (p*s) on the raw ones — the raw AC channels are probe-draw
        artifacts (see DynamicKernelAttention).
        """
        # Normalise with a scale built from the DC channel only. The generic
        # normalize_factors averages (p*s) over ALL raw channels, and the
        # raw AC channels are not basis-invariant, so that scale itself
        # varies with the probe draw and leaks non-invariance into every
        # downstream feature (measured: 1.4e-3 model-output drift between
        # two probe seeds at exact rank, versus 1e-15 for the channels).
        if p.shape[0] > n_elec:
            diag = (p[n_elec:, 0] * s[n_elec:, 0]).sum(-1).abs().mean().clamp_min(1e-30)
            scale = diag.sqrt()
            p, s = p / scale, s / scale
        chans = invariant_channels(p, s, self._omegas_layout)

        # Self-impedance for the ENCODER is taken before per-frequency
        # normalisation, so the model keeps the absolute impedance level
        # (which carries grid size and operating point). Only the factors
        # handed to the kernel are made dimensionless.
        selfz = torch.stack([(A * B).sum(-1) for A, B, _ in chans], -1)
        selfz = selfz.abs().clamp_min(1e-30).log10()

        if self.cfg.freq_norm:
            chans = self._normalize_per_frequency(chans, n_elec)
        return self.encoder(torch.cat([x, selfz], -1)), chans

    @staticmethod
    def _frob_scale(A, B, eps=1e-30):
        """RMS entry of Z over ALL pairs, without forming ``[N, N]``.

        ``Z = A B^T``, so ``||Z||_F^2 = tr(A^T A . B^T B)`` — two ``[d, d]``
        Gram matrices, O(N d^2 + d^3). Dividing by ``N^2`` gives the mean
        squared entry, which is size-normalised.

        This is the scale that matters: the *off-diagonal* entries are what
        enter the score, and measurement showed the diagonal does not track
        them across frequency (diagonal impedance falls faster with omega
        than off-diagonal does, so diagonal normalisation over-corrects and
        can anti-align the ranges it was meant to align).
        """
        N = A.shape[0]
        f2 = (A.transpose(-2, -1) @ A * (B.transpose(-2, -1) @ B)).sum()
        return (f2.abs() / (N * N)).clamp_min(eps).sqrt()

    def _normalize_per_frequency(self, chans, n_elec: int, eps: float = 1e-30):
        """Divide each frequency's factors by its own impedance scale.

            s_w^2 = mean_{i in loads} |Z_ii(w)|^2

        Rationale: features at different frequencies differ by orders of
        magnitude in raw units, so without this a single frequency dominates
        the score for purely numerical reasons.

        Properties, all required:
        * **basis-invariant** — ``Z_ii`` is a diagonal entry of ``Z``, so it
          is independent of the factor basis (unlike a raw AC channel).
        * **per-node, never a pair matrix** — computed from ``A_i . B_i``,
          O(N) and no ``[N, N]`` anywhere.
        * **differentiable** in R and C through the factors.
        * **size-independent** — a mean over load nodes, not a sum, and
          dividing by the mean self-impedance removes the growth of
          absolute impedance with grid size.
        * **per frequency** — the Re and Im channels of one frequency share
          one scale, since they are one physical frequency.
        * **factorization-preserving** — scaling only the observer factor
          ``A`` by a scalar leaves every degree an inner product, and makes
          degree ``k`` scale as ``(Z/s)^k``, i.e. dimensionless at every
          degree from a single scalar.
        """
        if len(chans) == 0 or chans[0][0].shape[0] <= n_elec:
            return chans
        mode = self.cfg.freq_norm_mode
        out, c = [], 0
        while c < len(chans):
            if c == 0:                                  # DC: Z_ii real
                A, B, nm = chans[0]
                if mode == "diag":
                    s_w = (((A[n_elec:] * B[n_elec:]).sum(-1)) ** 2
                           ).mean().clamp_min(eps).sqrt()
                else:
                    s_w = self._frob_scale(A, B, eps)
                out.append((A / s_w, B, nm))
                c += 1
            else:                                       # Re/Im of one freq
                Ar, Br, nr = chans[c]
                Ai, Bi, ni = chans[c + 1]
                if mode == "diag":
                    re = (Ar[n_elec:] * Br[n_elec:]).sum(-1)
                    im = (Ai[n_elec:] * Bi[n_elec:]).sum(-1)
                    s_w = (re ** 2 + im ** 2).mean().clamp_min(eps).sqrt()
                else:
                    # one scale per FREQUENCY, shared by its Re and Im
                    # channels, since they are one physical frequency
                    s_w = (self._frob_scale(Ar, Br, eps) ** 2
                           + self._frob_scale(Ai, Bi, eps) ** 2
                           ).clamp_min(eps).sqrt()
                out.append((Ar / s_w, Br, nr))
                out.append((Ai / s_w, Bi, ni))
                c += 2
        return out

    def forward(self, x, p, s, n_elec: int, naive: bool = False, fdc=None):
        """x [N, F]; p, s [N, n_ch, m]; loads are rows [n_elec:].

        Returns one droop prediction per load node, shape [L].
        """
        if self.cfg.score == "simple":
            if fdc is None:
                raise ValueError("score='simple' needs the symmetric DC "
                                 "factor (dc_symmetric_factor)")
            fn = fdc / fdc.norm(dim=-1, keepdim=True).mean().clamp_min(1e-30)
            zdc = (fn * fn).sum(-1, keepdim=True).clamp_min(1e-30).log10()
            hstate = self.encoder(torch.cat([x, zdc], -1))
            hstate = self.attn(hstate, fn, naive=naive)
            return self.decoder(hstate[n_elec:]).squeeze(-1)
        if self.cfg.score == "dynamic_kernel":
            hstate, chans = self.embed_invariant(x, p, s, n_elec)
            hstate = self.attn(hstate, chans, naive=naive)
            return self.decoder(hstate[n_elec:]).squeeze(-1)
        hstate, p, s = self.embed(x, p, s, n_elec)
        if isinstance(self.attn, KernelImpedanceAttention):
            if fdc is not None:
                fdc = fdc / fdc.norm(dim=-1, keepdim=True).mean().clamp_min(1e-30)
            hstate = self.attn(hstate, p, s, naive=naive, fdc=fdc)
        else:
            hstate = self.attn(hstate, p, s, naive=naive)
        return self.decoder(hstate[n_elec:]).squeeze(-1)
