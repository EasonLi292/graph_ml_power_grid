"""Incremental impedance updates: score a proposal without refactorizing.

Why this exists
---------------
Scoring one candidate modification currently costs a full rebuild of the
impedance factors, which is dominated by one dense LU per frequency. That
made the surrogate 1.3-36x SLOWER than just running the transient simulator
(``scripts/probes/inference_cost.py``), which removes the entire reason to
have a surrogate in the loop.

The fix is not a cheaper factorization, it is not redoing one. A circuit
modification changes every entry of ``Z`` -- but the CHANGE is low RANK.
Conductance enters the nodal admittance as

    Y = B^T diag(w) B  =  sum_e  w_e a_e a_e^T ,   a_e = e_u - e_v

so altering r branches is exactly a rank-r update

    Y' = Y + A D A^T ,   A = [a_1 ... a_r],  D = diag(w'_k - w_k).

Block Woodbury then gives the updated inverse ACTION from the base
factorization plus an r x r solve, with no N x N object anywhere:

    U    = Z A                                   (r back-substitutions)
    Z' B = Z B - U (I + D A^T U)^{-1} D (A^T Z B)

and ``A^T Z B`` is read off ``Z B`` by index arithmetic because ``A`` has
two non-zeros per column, so it costs O(r k), not O(N r k).

What is rank-r, and what is not
-------------------------------
    one wire width / via resistance      rank 1
    a wire section, strap or rail        rank = number of SEGMENTS in it
    add or remove an edge                rank 1   (old or new admittance 0)
    resize one decap                     rank 1   at w > 0, rank 0 at DC
    add / remove one decap               rank 1   at w > 0, rank 0 at DC
    move a decap                         rank 2   (remove + add)
    GLOBAL decap scaling                 rank = number of decap SITES

The last line is the one that matters: a global decap change is NOT low
rank, and this module does not pretend otherwise -- it falls back to
refactorization on a measured rank threshold. Likewise a "rail change" is
only cheap because a rail is a handful of segments, not because it is one
named action.

Capacitive branches carry admittance ``jwC``, so at DC their delta is
identically zero and no update is required at all. Changing a current
source does not touch ``Y`` and bypasses this module entirely.

Exactness
---------
This is an exact identity, not an approximation: the updated inverse action
agrees with a full refactorization to floating point (see
``scripts/probes/woodbury_checks.py``). Any error you see downstream comes
from the rank-m factor sketch, which is the same approximation the base
path already makes, and is reported separately.

The core ``I + D A^T U`` becomes singular exactly when the modification
disconnects the circuit -- removing a bridge of conductance g gives
``a^T Z a = R_eff = 1/g``, so the core hits zero -- so the conditioning
check IS the disconnection check, not a separate heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .impedance_factors import BranchSystem, admittance

# Structural kinds of a branch change.
KIND_G = "G"    # conductance, frequency-independent  (wire, via, generic edge)
KIND_C = "C"    # capacitance, admittance jwC         (decap)


# --------------------------------------------------------------- solve ops

class SolveOp:
    """Applies ``Z = Y^{-1}`` to a matrix without ever forming ``Z``.

    Everything downstream -- subspace iteration, the Galerkin projection,
    the DC symmetric factor -- touches ``Y`` only through this one call, so
    an incremental update is a drop-in replacement for a refactorization.

    Implementations must be immutable: the same base is reused across many
    candidate proposals and must never be mutated by evaluating one.
    """

    n: int
    dtype: torch.dtype

    def solve(self, B: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def depth(self) -> int:
        """Number of chained Woodbury levels above the dense base."""
        return 0


class SingularCircuit(RuntimeError):
    """The modified circuit has no solution at all.

    Raised when a proposal leaves some node with no conductive path to a
    clamped pad — ``Y`` is then exactly singular and there is no droop to
    predict, by refactorization or by any other means. Distinct from the
    near-singular case, which the rcond guard turns into a safe fallback.
    """


class DenseLUSolve(SolveOp):
    """Base factorization: one dense LU, reused for every subsequent solve.

    Dense is the prototype choice inherited from ``impedance_factors``; the
    interface is what matters, and a sparse factorization drops in here
    without touching anything above.
    """

    def __init__(self, Y: torch.Tensor):
        self.n = int(Y.shape[-1])
        self.dtype = Y.dtype
        self._lu, self._piv, info = torch.linalg.lu_factor_ex(Y)
        if int(info) != 0:
            raise SingularCircuit(
                f"admittance matrix is singular at pivot {int(info) - 1}: the "
                f"circuit is disconnected (some node has no path to a pad)")

    def solve(self, B: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lu_solve(self._lu, self._piv, B.to(self.dtype))


class DirectSolve(SolveOp):
    """Refactorize on every call -- the baseline the update is measured against."""

    def __init__(self, Y: torch.Tensor):
        self.n = int(Y.shape[-1])
        self.dtype = Y.dtype
        self._Y = Y

    def solve(self, B: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(self._Y, B.to(self.dtype))


class WoodburySolve(SolveOp):
    """``Z' = (Y + A D A^T)^{-1}`` applied through an immutable base solver.

    Two algebraically identical core forms are supported:

    ``nonsym`` (default)
        ``core = I + D A^T U``, applied as ``core^{-1} D``. Defined for
        ``D = 0``, which the symmetric form is not, so it is the only form
        that keeps a CONTINUOUS derivative through a zero-magnitude change
        -- needed for the gradient check at delta = 0, where ``d = 0`` but
        ``dd/dwidth != 0``.

    ``sym``
        ``core = D^{-1} + A^T U``, symmetric, better conditioned when every
        ``|d_k|`` is comfortably non-zero. Selected by measurement, not by
        assumption; see ``docs/INCREMENTAL_IMPEDANCE.md``.

    ``A^T Z B`` is evaluated by gathering rows of ``Z B`` rather than by a
    matrix product, since ``A`` has at most two non-zeros per column.
    """

    def __init__(self, base: SolveOp, ia: torch.Tensor, ib: torch.Tensor,
                 d: torch.Tensor, u_cols: torch.Tensor | None = None,
                 core_form: str = "nonsym", rcond_min: float = 1e-8):
        self.n, self.dtype = base.n, base.dtype
        self._base = base
        self._ia, self._ib = ia, ib          # free-node indices, -1 = clamped
        self._d = d.to(self.dtype)
        self._depth = base.depth + 1

        A = incidence_matrix(ia, ib, self.n, self.dtype)
        U = base.solve(A) if u_cols is None else u_cols.to(self.dtype)
        self._U = U
        AtU = gather_rows(U, ia, ib)          # [r, r], = A^T Z A

        r = ia.shape[0]
        eye = torch.eye(r, dtype=self.dtype, device=U.device)
        if core_form == "sym":
            M = torch.diag(1.0 / self._d)
            core = M + AtU
            self._apply_core = lambda X: torch.linalg.solve(core, X)
        else:
            M = self._d.unsqueeze(1) * AtU
            core = eye + M
            self._apply_core = lambda X: torch.linalg.solve(
                core, self._d.unsqueeze(1) * X)
        self._core = core
        self.core_rcond = _rcond(core.detach(), M.detach())
        self.singular = not (self.core_rcond > rcond_min)

    def solve(self, B: torch.Tensor) -> torch.Tensor:
        ZB = self._base.solve(B)
        return ZB - self._U @ self._apply_core(gather_rows(ZB, self._ia, self._ib))

    @property
    def depth(self) -> int:
        return self._depth


def _rcond(core: torch.Tensor, M: torch.Tensor) -> float:
    """How close the core is to singular, on a scale the update sets.

    The textbook condition number is useless here: the core is ``1 x 1`` for
    every single-branch action, and ``cond`` of a 1x1 matrix is identically
    1.0 no matter how near zero the entry is. Removing a bridge would then
    sail through the check that exists to catch exactly that.

    So compare the smallest singular value of ``core = I + M`` against the
    scale ``M`` itself sets, ``max(1, ||M||)``. Widening a wire gives
    ``rcond ~ 1``; approaching the conductance change that disconnects the
    circuit drives it to zero, in the 1x1 case as much as the block case.
    Solve error tracks ``eps / rcond`` — measured in
    ``scripts/probes/woodbury_checks.py`` check 10, which is where the
    default threshold comes from.
    """
    sv = torch.linalg.svdvals(core)
    scale = torch.clamp(torch.linalg.matrix_norm(M, ord=2), min=1.0)
    r = float(sv.min() / scale)
    return r if r == r else 0.0            # NaN core -> treat as singular


def incidence_matrix(ia: torch.Tensor, ib: torch.Tensor, n: int,
                     dtype) -> torch.Tensor:
    """``A`` of shape ``[n, r]``; a clamped terminal simply drops its entry.

    That reproduces the boundary handling in ``_incidence_apply`` exactly: a
    branch with one clamped end contributes only ``w`` to the live diagonal,
    which is what ``a a^T`` gives when ``a`` is a single unit vector.
    """
    r = ia.shape[0]
    A = torch.zeros((n, r), dtype=dtype, device=ia.device)
    k = torch.arange(r, device=ia.device)
    ma, mb = ia >= 0, ib >= 0
    if ma.any():
        A[ia[ma], k[ma]] += 1.0
    if mb.any():
        A[ib[mb], k[mb]] -= 1.0
    return A


def gather_rows(M: torch.Tensor, ia: torch.Tensor,
                ib: torch.Tensor) -> torch.Tensor:
    """``A^T M`` by indexing -- O(r k), never O(n r k)."""
    pad = torch.zeros((1, M.shape[1]), dtype=M.dtype, device=M.device)
    Mp = torch.cat([M, pad], 0)
    n = M.shape[0]
    return Mp[torch.where(ia >= 0, ia, n)] - Mp[torch.where(ib >= 0, ib, n)]


# ---------------------------------------------------------------- proposals

@dataclass(frozen=True)
class BranchChange:
    """One branch change: endpoints, old value, new value.

    ``u, v`` are FLAT ELECTRICAL node indices (the ``BranchSystem`` axis).
    For ``KIND_G`` the values are conductances; for ``KIND_C`` they are
    capacitances, whose admittance ``jwC`` is formed per frequency. Values
    are tensors so autograd reaches wire width and capacitance.

    Adding a branch is ``old = 0``; removing one is ``new = 0``. Both are
    rank 1 -- the node set is what must stay fixed, not the edge set.
    """
    u: int
    v: int
    kind: str
    old: torch.Tensor
    new: torch.Tensor


@dataclass
class Proposal:
    """A candidate modification: branch changes plus an optional load change.

    ``loads`` changes the current sources only. Sources do not appear in
    ``Y`` at all, so a load-only proposal has rank 0 and skips the impedance
    update entirely rather than doing a rank-0 no-op.
    """
    changes: list[BranchChange] = field(default_factory=list)
    loads: torch.Tensor | None = None
    label: str = ""
    new_nodes: int = 0          # > 0 forces refactorization; nothing supports it

    @property
    def rank(self) -> int:
        return len(self.changes)

    def touches_Y(self) -> bool:
        return bool(self.changes)


def deltas_at(changes, omega: float, dtype) -> torch.Tensor:
    """Admittance deltas of every change at one frequency.

    Capacitive branches contribute ``jw dC``, which is identically zero at
    DC -- the reason a decap action needs no DC update at all.
    """
    out = []
    for c in changes:
        dv = (c.new - c.old).to(dtype)
        out.append(dv if c.kind == KIND_G else dv * (1j * omega))
    return torch.stack(out) if out else torch.zeros(0, dtype=dtype)


# ---------------------------------------------------- base + update manager

@dataclass
class FallbackPolicy:
    """When to stop updating and just refactorize.

    ``max_rank`` is CALIBRATED, not assumed -- see
    ``scripts/probes/woodbury_crossover.py``, which measures where the
    update stops paying and writes the threshold to
    ``docs/analysis/woodbury_crossover.json``.
    """
    max_rank: int = 64
    max_branch_frac: float = 0.25
    rcond_min: float = 1e-8         # see _rcond; error tracks eps / rcond
    max_depth: int = 8              # chained accepted updates before refactor
    core_form: str = "nonsym"

    @classmethod
    def calibrated(cls, path="docs/analysis/woodbury_crossover.json", **kw):
        """Load the measured rank threshold if a calibration run exists."""
        import json
        from pathlib import Path
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text())
            kw.setdefault("max_rank", int(d["recommended_max_rank"]))
        return cls(**kw)


class IncrementalImpedance:
    """Immutable base factorization + many candidate proposals.

    Evaluating a proposal never mutates the base, so alternatives are
    independent and can be scored in any order. ``ZA`` columns are cached
    per (frequency, branch), because a branch's incidence vector does not
    depend on how much its value changed -- re-scoring the same strap at
    six magnitudes solves for ``U`` once.
    """

    def __init__(self, sys_: BranchSystem, R: torch.Tensor, C: torch.Tensor,
                 omegas: torch.Tensor, L: torch.Tensor | None = None,
                 policy: FallbackPolicy | None = None):
        self.sys = sys_
        self.omegas = omegas
        self.policy = policy or FallbackPolicy()
        self._L = L
        self.R, self.C = R, C
        self._build_base(R, C)
        self.n_branches = int(sys_.r_edges.shape[0] + sys_.c_edges.shape[0])
        self.stats = dict(updated=0, refactored=0, u_hits=0, u_misses=0)

    # -- base ------------------------------------------------------------
    def _build_base(self, R, C):
        self.solvers = [DenseLUSolve(admittance(self.sys, R, C, w, self._L))
                        for w in self.omegas]
        self._u_cache = [{} for _ in self.omegas]

    def refactorize(self, R=None, C=None):
        """Rebuild the base from scratch; clears every cache."""
        self.R = self.R if R is None else R
        self.C = self.C if C is None else C
        self._build_base(self.R, self.C)
        self.stats["refactored"] += 1

    # -- fallback decision ------------------------------------------------
    def _reject(self, prop: Proposal) -> str | None:
        if prop.new_nodes:
            return "new nodes"
        if prop.rank > self.policy.max_rank:
            return f"rank {prop.rank} > max_rank {self.policy.max_rank}"
        if self.n_branches and prop.rank / self.n_branches > self.policy.max_branch_frac:
            return (f"rank {prop.rank} is "
                    f"{prop.rank / self.n_branches:.0%} of all branches")
        return None

    # -- U cache ----------------------------------------------------------
    def _u_columns(self, f: int, ia, ib):
        """``Z a_e`` per change, from cache where possible."""
        cache = self._u_cache[f]
        keys = [(int(a), int(b)) for a, b in zip(ia, ib)]
        miss = [k for k in dict.fromkeys(keys) if k not in cache]
        if miss:
            mi = torch.tensor([k[0] for k in miss], device=ia.device)
            mb = torch.tensor([k[1] for k in miss], device=ia.device)
            dtype = self.solvers[f].dtype
            cols = self.solvers[f].solve(
                incidence_matrix(mi, mb, self.solvers[f].n, dtype))
            for j, k in enumerate(miss):
                cache[k] = cols[:, j]
        self.stats["u_hits"] += len(keys) - len(miss)
        self.stats["u_misses"] += len(miss)
        return torch.stack([cache[k] for k in keys], 1)

    # -- the operation ----------------------------------------------------
    def solvers_for(self, prop: Proposal, R_new=None, C_new=None,
                    sys_new=None):
        """Per-frequency solve operators for the modified circuit.

        Returns ``(solvers, info)``. ``info['mode']`` is ``update``,
        ``bypass`` (``Y`` untouched) or ``refactor`` (with a reason).

        ``R_new``/``C_new``/``sys_new`` are consulted ONLY on the
        refactorization path; the update path derives everything from the
        proposal itself, which is what keeps the base immutable. ``sys_new``
        is required when the proposal adds or removes a branch, since that
        changes the edge set the fallback has to stamp.
        """
        info = {"mode": "update", "rank": prop.rank, "reason": None,
                "rcond": [], "depth": []}

        if not prop.touches_Y():
            info["mode"] = "bypass"
            return list(self.solvers), info

        reason = self._reject(prop)
        if reason is None:
            ia = torch.tensor([self.sys.free_of[c.u] for c in prop.changes])
            ib = torch.tensor([self.sys.free_of[c.v] for c in prop.changes])
            out, bad = [], None
            for f, w in enumerate(self.omegas):
                d = deltas_at(prop.changes, float(w), self.solvers[f].dtype)
                if not bool(d.detach().any()) and not d.requires_grad:
                    # e.g. a purely capacitive change at DC: no update at all
                    out.append(self.solvers[f])
                    info["rcond"].append(1.0)
                    continue
                U = self._u_columns(f, ia, ib)
                op = WoodburySolve(self.solvers[f], ia, ib, d, u_cols=U,
                                   core_form=self.policy.core_form,
                                   rcond_min=self.policy.rcond_min)
                info["rcond"].append(op.core_rcond)
                if op.singular:
                    bad = (f"core near-singular at w[{f}] "
                           f"(rcond {op.core_rcond:.2e} <= "
                           f"{self.policy.rcond_min:.0e}); the modification "
                           f"is at or past the point of disconnecting the "
                           f"circuit")
                    break
                out.append(op)
            if bad is None:
                self.stats["updated"] += 1
                info["depth"] = [s.depth for s in out]
                return out, info
            reason = bad

        # -- safe fallback
        info["mode"], info["reason"] = "refactor", reason
        if R_new is None and C_new is None:
            raise ValueError(
                "fallback to refactorization needs the modified circuit; pass "
                "R_new/C_new (and sys_new when the proposal adds or removes a "
                "branch) to solvers_for(). Reason: " + str(reason))
        sysm = self.sys if sys_new is None else sys_new
        R_new = self.R if R_new is None else R_new
        C_new = self.C if C_new is None else C_new
        if R_new.shape[0] != sysm.r_edges.shape[0]:
            raise ValueError(
                f"R_new has {R_new.shape[0]} entries but sys_new has "
                f"{sysm.r_edges.shape[0]} resistive branches — an add/remove "
                f"proposal must pass its modified BranchSystem as sys_new")
        self.stats["refactored"] += 1
        try:
            ops = [DenseLUSolve(admittance(sysm, R_new, C_new, w, self._L))
                   for w in self.omegas]
        except SingularCircuit as exc:
            raise SingularCircuit(
                f"{exc}; the proposal '{prop.label or '(unlabelled)'}' is "
                f"invalid, not merely hard to update incrementally") from None
        return ops, info

    def accept(self, prop: Proposal, R_new=None, C_new=None, sys_new=None):
        """Commit a proposal to the base, refactorizing when drift warrants.

        Chained Woodbury levels each add O(n r) to every later solve and
        accumulate rounding, so the base is rebuilt once the chain reaches
        ``policy.max_depth``.
        """
        solvers, info = self.solvers_for(prop, R_new=R_new, C_new=C_new,
                                         sys_new=sys_new)
        self.solvers = solvers
        self._u_cache = [{} for _ in self.omegas]
        if sys_new is not None:
            self.sys = sys_new
        if max((s.depth for s in solvers), default=0) >= self.policy.max_depth:
            if R_new is None and C_new is None:
                raise ValueError("periodic refactorization needs (R, C)")
            self.refactorize(R_new, C_new)
            info["refactored_after_accept"] = True
        return info
