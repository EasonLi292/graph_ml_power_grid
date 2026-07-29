"""Differentiable multi-frequency impedance factors for global attention.

Produces compact observer/source factors ``p, s`` with ``p_i . s_j ~= Z_ij``
for the nodal admittance ``Y(w) = B^T diag(w(w)) B`` of the PDN, entirely
inside PyTorch autograd so gradients reach every R, C and L.

Mechanism (uniform at every frequency, no hand-designed time constants) —
randomized subspace iteration:

    X  = solve(Y, Omega)              Omega [n_free, m], fixed per graph
    repeat q:  X <- solve(Y, X)       power iterations
    Qr = qr(X).Q
    T  = Qr^T solve(Y, Qr)            [m, m] small and dense
    p_i = Qr_i                        s_j = (Qr T^T)_j

Measured against the DC solver on this track (q=2): droop reconstructed
from rank-m factors is within 1.4 % (3,7) / 2.3 % (13,13) at m=16, i.e.
optimal-rank accuracy and **invariant to grid size** — the O(1)-reach
property that a fixed message-passing depth cannot provide.

Clamped pads get zero factors (voltage sources have zero transfer
impedance). Load nodes take the oriented terminal difference
``p_load = p_vdd - p_vss`` (see ``load_factors``), which makes the DC
superposition ``droop_l = sum_l' (u_l . u_l') I_l'`` exact — verified to
2e-11 in ``scripts/probes/impedance_attention_checks.py``.

See docs/IMPEDANCE_ATTENTION_DESIGN.md. NOTE: the prototype builds a dense
``Y`` (fine for n_free <= ~400 here); the sparse iterative replacement is
the documented unresolved scaling item.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .grid_construction import PDNGraph


@dataclass
class BranchSystem:
    """Topology + boundary of one grid, independent of knob values.

    Index convention for the flat node axis:
        [0, n_elec)            electrical nodes (mesh_top then mesh_bot)
        [n_elec, n_elec + L)   load nodes
    """
    n_elec: int                  # electrical nodes (top + bot)
    n_free: int                  # non-clamped electrical nodes
    free_of: torch.Tensor        # [n_elec] -> free index, -1 if clamped
    r_edges: torch.Tensor        # [E_r, 2] resistive branch endpoints (flat elec idx)
    c_edges: torch.Tensor        # [E_c, 2] capacitive branch endpoints
    l_edges: torch.Tensor        # [E_l, 2] inductive branch endpoints (IBM)
    load_terminals: torch.Tensor  # [L, 2] (vdd_node, vss_node) flat elec idx
    node_type: torch.Tensor      # [n_elec] 0=mesh_top 1=mesh_bot 2=pad
    is_vdd: torch.Tensor         # [n_elec] bool

    @property
    def n_loads(self) -> int:
        return int(self.load_terminals.shape[0])

    @property
    def n_nodes(self) -> int:
        return self.n_elec + self.n_loads


def branch_system(g: PDNGraph) -> BranchSystem:
    """Extract topology/boundary from a ``PDNGraph`` (no knob values)."""
    top0, bot0 = 0, g.n_top_nodes
    n_elec = g.n_top_nodes + g.n_bot_nodes

    r_edges = np.concatenate([
        np.stack([top0 + g.top_edges[:, 0], top0 + g.top_edges[:, 1]], 1),
        np.stack([bot0 + g.bot_edges[:, 0], bot0 + g.bot_edges[:, 1]], 1),
        np.stack([top0 + g.via_pairs[:, 0], bot0 + g.via_pairs[:, 1]], 1),
    ]).astype(np.int64)
    c_edges = np.stack([bot0 + g.decap_pairs[:, 0],
                        bot0 + g.decap_pairs[:, 1]], 1).astype(np.int64)
    loads = np.stack([bot0 + g.load_pairs[:, 0],
                      bot0 + g.load_pairs[:, 1]], 1).astype(np.int64)

    pad = np.concatenate([g.vdd_pad_top_idx, g.vss_pad_top_idx]).astype(int)
    free_of = np.full(n_elec, -1, dtype=np.int64)
    free = np.setdiff1d(np.arange(n_elec), pad)
    free_of[free] = np.arange(free.size)

    node_type = np.zeros(n_elec, dtype=np.int64)
    node_type[bot0:] = 1
    node_type[pad] = 2
    is_vdd = np.concatenate([g.top_is_vdd, g.bot_is_vdd]).astype(bool)

    t = torch.from_numpy
    return BranchSystem(
        n_elec=n_elec, n_free=int(free.size), free_of=t(free_of),
        r_edges=t(r_edges), c_edges=t(c_edges),
        l_edges=torch.zeros((0, 2), dtype=torch.int64),
        load_terminals=t(loads), node_type=t(node_type), is_vdd=t(is_vdd),
    )


def _incidence_apply(sys_: BranchSystem, edges: torch.Tensor,
                     w: torch.Tensor, n_free: int, dtype):
    """Accumulate ``B^T diag(w) B`` for one branch family into a dense Y.

    Dense is a prototype choice; the identical stamping works on a sparse
    layout (see the design note's unresolved item 1).
    """
    Y = torch.zeros((n_free, n_free), dtype=dtype, device=w.device)
    if edges.numel() == 0:
        return Y
    ia = sys_.free_of[edges[:, 0]]
    ib = sys_.free_of[edges[:, 1]]
    both = (ia >= 0) & (ib >= 0)
    if both.any():
        a, b, wv = ia[both], ib[both], w[both]
        Y.index_put_((a, a), wv, accumulate=True)
        Y.index_put_((b, b), wv, accumulate=True)
        Y.index_put_((a, b), -wv, accumulate=True)
        Y.index_put_((b, a), -wv, accumulate=True)
    # branches with one clamped terminal contribute only the live diagonal
    for live, dead in ((ia, ib), (ib, ia)):
        m = (live >= 0) & (dead < 0)
        if m.any():
            Y.index_put_((live[m], live[m]), w[m], accumulate=True)
    return Y


def admittance(sys_: BranchSystem, R: torch.Tensor, C: torch.Tensor,
               omega: torch.Tensor, L: torch.Tensor | None = None):
    """Complex nodal admittance ``Y(omega)`` over free nodes.

    Differentiable in R, C, L and omega. ``omega`` is a 0-dim tensor.
    """
    cdt = torch.complex128 if R.dtype == torch.float64 else torch.complex64
    Y = _incidence_apply(sys_, sys_.r_edges,
                         (1.0 / R).to(cdt), sys_.n_free, cdt)
    if C.numel() and float(omega.detach().abs()) > 0:
        Y = Y + _incidence_apply(sys_, sys_.c_edges,
                                 (1j * omega.to(cdt) * C.to(cdt)),
                                 sys_.n_free, cdt)
    if L is not None and L.numel() and float(omega.detach().abs()) > 0:
        Y = Y + _incidence_apply(sys_, sys_.l_edges,
                                 1.0 / (1j * omega.to(cdt) * L.to(cdt)),
                                 sys_.n_free, cdt)
    return Y


def _subspace_factors(Y: torch.Tensor, probes: torch.Tensor, n_power: int):
    """Randomized subspace iteration -> (p, s) with ``p_i^T s_j ~= Z_ij``.

    Galerkin projection onto ``range(Qr)``: ``Z ~= Qr (Qr^H Z Qr) Qr^H``.

    ``torch.linalg.qr`` returns a **unitary** ``Qr`` (``Qr^H Qr = I``), not a
    complex-orthogonal one (``Qr^T Qr != I`` — measured 0.94 at w>0), so the
    conjugate transpose is required in ``T`` and the conjugate in ``s``.
    Using plain transposes silently gives ~100 % error at every w>0 while
    staying exact at DC, where ``Y`` is real. Note ``p^T s`` below uses a
    plain transpose by construction: the conjugation lives in ``s``.

    Returns factors over FREE nodes only, shape ``[n_free, m]`` each.
    """
    X = torch.linalg.solve(Y, probes.to(Y.dtype))
    for _ in range(n_power):
        X = torch.linalg.solve(Y, X)
    Qr, _ = torch.linalg.qr(X)
    ZQ = torch.linalg.solve(Y, Qr)
    T = Qr.conj().transpose(-2, -1) @ ZQ          # [m, m] = Qr^H Z Qr
    return Qr, Qr.conj() @ T.transpose(-2, -1), T


def symmetric_factor(Y: torch.Tensor, probes: torch.Tensor, n_power: int):
    """Symmetric factor ``F`` with ``F_i . F_j ~= Z_ij`` for real SPD ``Y``.

    The asymmetric ``(p, s)`` pair reconstructs ``Z`` but does not make
    ``z_ii + z_jj - 2 z_ij`` a Euclidean distance, which random Fourier
    features require. Since ``Z`` is symmetric PSD at DC, ``T = Qr^T Z Qr``
    is too, so ``F = Qr T^{1/2}`` gives both

        F_i . F_j       = Z_ij
        ||F_i - F_j||^2 = Z_ii + Z_jj - 2 Z_ij = R_eff(i, j)

    Real DC only — at ``w > 0`` ``T`` is complex symmetric, not PSD, and
    has no real square root.
    """
    _, _, T = _subspace_factors(Y, probes, n_power)
    Tr = T.real if T.is_complex() else T
    Tr = 0.5 * (Tr + Tr.transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(Tr)
    root = evecs @ torch.diag_embed(evals.clamp_min(0).sqrt()) @ evecs.transpose(-2, -1)
    X = torch.linalg.solve(Y, probes.to(Y.dtype))
    for _ in range(n_power):
        X = torch.linalg.solve(Y, X)
    Qr, _ = torch.linalg.qr(X)
    return (Qr.real if Qr.is_complex() else Qr) @ root


def channel_count(omegas) -> int:
    """Number of REAL channels emitted by :func:`impedance_factors`.

    A real (DC) system needs one channel; a complex one needs four, because
    both parts of ``Z`` require cross terms:

        Re(Z) = <p_re, s_re> - <p_im, s_im>
        Im(Z) = <p_re, s_im> + <p_im, s_re>

    Emitting only the diagonal pairs (re,re) and (im,im) makes ``Im(Z)``
    unrepresentable — and ``|Im Z| / |Re Z|`` is ~0.30 at the load
    frequency, i.e. phase information the transient target depends on.
    """
    return sum(1 if float(w) == 0.0 else 4 for w in omegas)


def impedance_factors(
    sys_: BranchSystem,
    R: torch.Tensor,
    C: torch.Tensor,
    omegas: torch.Tensor,
    m: int = 16,
    n_power: int = 2,
    L: torch.Tensor | None = None,
    seed: int = 0,
):
    """Per-frequency observer/source factors on the FULL node axis.

    Returns ``(p, s)`` of shape ``[n_nodes, C_ch, m]`` with
    ``C_ch = channel_count(omegas)`` real channels: one at DC, four per
    non-zero frequency (the (re,re), (im,im), (re,im), (im,re) pairings
    needed to span both ``Re Z`` and ``Im Z``). Load-node factors are the
    oriented terminal difference; clamped pads are zero.
    """
    dev, dt = R.device, R.dtype
    gen = torch.Generator(device="cpu").manual_seed(seed)
    probes = torch.randn(sys_.n_free, m, generator=gen,
                         dtype=torch.float64).to(device=dev, dtype=dt)

    p_ch, s_ch = [], []
    for f in range(omegas.shape[0]):
        Y = admittance(sys_, R, C, omegas[f], L)
        pf, sf, _ = _subspace_factors(Y, probes, n_power)
        if float(omegas[f]) == 0.0:
            p_ch += [pf.real]
            s_ch += [sf.real]
        else:
            # pairs spanning Re(Z) = rr - ii and Im(Z) = ri + ir
            p_ch += [pf.real, pf.imag, pf.real, pf.imag]
            s_ch += [sf.real, sf.imag, sf.imag, sf.real]
    p_free = torch.stack(p_ch, 1).to(dt)      # [n_free, C_ch, m]
    s_free = torch.stack(s_ch, 1).to(dt)

    n_ch = p_free.shape[1]
    p_e = p_free.new_zeros((sys_.n_elec, n_ch, m))
    s_e = s_free.new_zeros((sys_.n_elec, n_ch, m))
    live = sys_.free_of >= 0
    p_e[live] = p_free[sys_.free_of[live]]
    s_e[live] = s_free[sys_.free_of[live]]

    p_l, s_l = load_factors(sys_, p_e, s_e)
    return torch.cat([p_e, p_l], 0), torch.cat([s_e, s_l], 0)


def dc_symmetric_factor(sys_: BranchSystem, R: torch.Tensor, C: torch.Tensor,
                        m: int = 16, n_power: int = 2, seed: int = 0):
    """DC symmetric factor on the FULL node axis (loads = oriented difference).

    ``F_i . F_j ~= Z_ij`` and ``||F_i - F_j||^2 ~= R_eff(i, j)``, so a
    Gaussian kernel of this distance can be approximated by random Fourier
    features. Clamped pads are zero, as elsewhere.
    """
    dev, dt = R.device, R.dtype
    gen = torch.Generator(device="cpu").manual_seed(seed)
    probes = torch.randn(sys_.n_free, m, generator=gen,
                         dtype=torch.float64).to(device=dev, dtype=dt)
    Y = admittance(sys_, R, C, torch.zeros((), dtype=dt, device=dev))
    F = symmetric_factor(Y, probes, n_power).to(dt)
    out = F.new_zeros((sys_.n_elec, F.shape[1]))
    live = sys_.free_of >= 0
    out[live] = F[sys_.free_of[live]]
    a, b = sys_.load_terminals[:, 0], sys_.load_terminals[:, 1]
    return torch.cat([out, out[a] - out[b]], 0)


def load_factors(sys_: BranchSystem, p_e: torch.Tensor, s_e: torch.Tensor):
    """Oriented terminal difference: VDD terminal minus VSS terminal.

    The sign convention is applied identically to both factors, which is
    what makes ``droop_l = sum_l' (p_l . s_l') I_l'`` come out exact.
    """
    a, b = sys_.load_terminals[:, 0], sys_.load_terminals[:, 1]
    return p_e[a] - p_e[b], s_e[a] - s_e[b]


def node_features(sys_: BranchSystem, loads: torch.Tensor,
                  log_R_stats=None) -> torch.Tensor:
    """Shared node-feature matrix for electrical AND load nodes.

    Node type is an input feature, never a separate parameter block:
        [is_top, is_bot, is_pad, is_load, is_vdd, I_peak, freq, duty,
         sin phase, cos phase]
    Electrical nodes zero the load block; load nodes zero the rail block.
    """
    n_e, L = sys_.n_elec, sys_.n_loads
    dt = loads.dtype if loads.numel() else torch.float32
    x = torch.zeros((n_e + L, 10), dtype=dt, device=loads.device)
    x[torch.arange(n_e), sys_.node_type] = 1.0          # is_top/is_bot/is_pad
    x[:n_e, 4] = sys_.is_vdd.to(dt)
    if L:
        x[n_e:, 3] = 1.0
        ip, fr, du, ph = loads[:, 0], loads[:, 1], loads[:, 2], loads[:, 3]
        x[n_e:, 5] = torch.log10(ip.clamp_min(1e-12)) + 4.0   # ~O(1)
        x[n_e:, 6] = torch.log10(fr.clamp_min(1e-12)) - 9.0
        x[n_e:, 7] = du
        x[n_e:, 8] = torch.sin(2 * np.pi * ph)
        x[n_e:, 9] = torch.cos(2 * np.pi * ph)
    return x


def knob_tensors(g: PDNGraph, ww_top, ww_bot, C_decap, Rsheet_top, Rsheet_bot,
                 R_via: float):
    """Assemble the differentiable branch-value vectors (R, C) from knobs.

    ``ww_top``/``ww_bot`` are per-physical-edge widths, ``C_decap`` either a
    scalar tensor or a per-site vector. Ordering matches ``branch_system``:
    top straps, bot straps, vias.
    """
    R_top = Rsheet_top * g.pitch_top / ww_top
    R_bot = Rsheet_bot * g.pitch_bot / ww_bot
    R_via_v = torch.full((g.via_pairs.shape[0],), float(R_via),
                         dtype=R_top.dtype, device=R_top.device)
    R = torch.cat([R_top, R_bot, R_via_v])
    n_d = g.decap_pairs.shape[0]
    C = (C_decap.expand(n_d) if C_decap.dim() == 0 else C_decap)
    return R, C
