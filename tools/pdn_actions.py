"""Circuit modifications expressed as rank-r branch changes.

Each action returns a :class:`Candidate` carrying BOTH representations of
the same modification:

* a :class:`~tools.incremental_impedance.Proposal` — the branch changes, for
  the incremental update path;
* the fully modified circuit (``sys``, ``R``, ``C``, widths, per-site
  capacitance, loads) — for exact refactorization, for the simulator, and
  for the local R/C node features.

Keeping both in one object is what makes the exactness tests possible: the
update path and the refactorization path are handed the *same* modification
by construction, not by two hand-written descriptions of it that could
drift apart.

Rank, per family
----------------
    wire_width(one edge)       1        via_resistance(one via)     1
    wire_section(k segments)   k        strap / rail (k segments)   k
    remove_edge / add_edge     1        decap_resize(one site)      1 (0 at DC)
    decap_add / decap_remove   1        decap_move                  2
    decap_multi(k sites)       k        decap_global(S sites)       S
    load_*                     0  -- current sources are not in Y at all

``decap_global`` is deliberately included and deliberately NOT cheap: its
rank is the number of decap SITES, so it is the family that must trip the
fallback rather than quietly produce a dense update. A "rail change" is
cheap only because a rail is a handful of segments — the rank is the
segment count, never the name of the action.

Adding and removing a decap are ordinary value changes here, because every
candidate site already exists in the grid (``DECAP_ROW_STRIDE = 1``) and a
site holding zero capacitance is electrically identical to an empty slot.
Adding and removing a *resistive* edge genuinely changes the edge set, so
those candidates carry ``topology_changed=True`` and are checked against
refactorization only — the fixed-topology grid builder cannot simulate them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from .grid_construction import PDNGraph
from .impedance_factors import BranchSystem, knob_tensors
from .incremental_impedance import KIND_C, KIND_G, BranchChange, Proposal


@dataclass
class Candidate:
    """One proposed modification, in both representations."""
    label: str
    family: str
    proposal: Proposal
    sys: BranchSystem
    R: torch.Tensor
    C: torch.Tensor
    wt: torch.Tensor
    wb: torch.Tensor
    C_sites: torch.Tensor
    loads: torch.Tensor
    topology_changed: bool = False

    @property
    def rank(self) -> int:
        return self.proposal.rank


class ActionSpace:
    """Actions over one fixed base circuit. Never mutates the base.

    Branch ordering follows ``branch_system`` / ``knob_tensors`` exactly:
    ``r_edges = [top straps | bot straps | vias]`` and ``c_edges = decap
    sites``. Those offsets are the only coupling between an action and the
    factor pipeline, so they are computed once here rather than open-coded
    at each call site.
    """

    def __init__(self, g: PDNGraph, sys_: BranchSystem, wt: torch.Tensor,
                 wb: torch.Tensor, C_sites: torch.Tensor, loads: torch.Tensor,
                 Rsheet_top: float, Rsheet_bot: float, R_via: float):
        self.g, self.sys = g, sys_
        self.wt, self.wb, self.C_sites, self.loads = wt, wb, C_sites, loads
        self.Rsheet_top, self.Rsheet_bot, self.R_via = Rsheet_top, Rsheet_bot, R_via
        self.n_top_e = int(g.top_edges.shape[0])
        self.n_bot_e = int(g.bot_edges.shape[0])
        self.n_via = int(g.via_pairs.shape[0])
        self.R, self.C = self.knobs(wt, wb, C_sites)

    # -- plumbing ---------------------------------------------------------
    def knobs(self, wt, wb, C_sites):
        return knob_tensors(self.g, wt, wb, C_sites, self.Rsheet_top,
                            self.Rsheet_bot, self.R_via)

    def _r_endpoints(self, j: int):
        e = self.sys.r_edges[j]
        return int(e[0]), int(e[1])

    def _c_endpoints(self, k: int):
        e = self.sys.c_edges[k]
        return int(e[0]), int(e[1])

    def _cand(self, label, family, changes, wt, wb, C_sites, loads=None,
              sys_=None, topology_changed=False, new_nodes=0):
        loads = self.loads if loads is None else loads
        sys_ = self.sys if sys_ is None else sys_
        R, C = self.knobs(wt, wb, C_sites)
        # Callers that change the EDGE SET (add / remove) or that touch a
        # non-width knob (via R) overwrite ``R`` after construction; the
        # width path is already correct here.
        return Candidate(label=label, family=family,
                         proposal=Proposal(changes=list(changes), loads=loads,
                                           label=label, new_nodes=new_nodes),
                         sys=sys_, R=R, C=C, wt=wt, wb=wb, C_sites=C_sites,
                         loads=loads, topology_changed=topology_changed)

    # -- resistive value changes -----------------------------------------
    def _scale_conductance(self, rows, factor, wt, wb, label, family):
        """Common path: widths already scaled, emit the branch changes."""
        R_new, _ = self.knobs(wt, wb, self.C_sites)
        changes = []
        for j in rows:
            u, v = self._r_endpoints(int(j))
            changes.append(BranchChange(u, v, KIND_G,
                                        old=1.0 / self.R[int(j)],
                                        new=1.0 / R_new[int(j)]))
        return self._cand(label, family, changes, wt, wb, self.C_sites)

    def wire_width(self, layer: int, idx, factor: float, label=None):
        """Scale the width of one or more strap segments. Rank = len(idx)."""
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        wt, wb = self.wt.clone(), self.wb.clone()
        if layer == 0:
            wt[idx] = wt[idx] * factor
            rows = idx
        else:
            wb[idx] = wb[idx] * factor
            rows = idx + self.n_top_e
        fam = {1: "ww_edge"}.get(idx.size, "ww_multi")
        return self._scale_conductance(
            rows, factor, wt, wb,
            label or f"ww L{layer} x{factor:g} on {idx.size} seg", fam)

    def via_resistance(self, idx, factor: float):
        """Scale via resistance. Vias are r_edges after both strap blocks."""
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        rows = idx + self.n_top_e + self.n_bot_e
        changes = []
        for j in rows:
            u, v = self._r_endpoints(int(j))
            g_old = 1.0 / self.R[int(j)]
            changes.append(BranchChange(u, v, KIND_G, old=g_old,
                                        new=g_old / factor))
        R = self.R.clone()
        R[rows] = R[rows] * factor
        c = self._cand(f"via x{factor:g} on {idx.size}", "via", changes,
                       self.wt, self.wb, self.C_sites)
        c.R = R                       # vias are not a width knob
        return c

    def remove_resistive_edge(self, j: int):
        """Delete one resistive branch. Rank 1, with ``new = 0``.

        The reference circuit DROPS the row from ``r_edges``; the update
        represents the same thing as a conductance going to zero. Those must
        agree, which is the test that "remove" is genuinely rank 1.
        """
        u, v = self._r_endpoints(j)
        changes = [BranchChange(u, v, KIND_G, old=1.0 / self.R[j],
                                new=torch.zeros((), dtype=self.R.dtype))]
        keep = torch.ones(self.R.shape[0], dtype=torch.bool)
        keep[j] = False
        sys2 = replace(self.sys, r_edges=self.sys.r_edges[keep])
        cand = self._cand(f"remove r_edge {j}", "edge_remove", changes,
                          self.wt, self.wb, self.C_sites, sys_=sys2,
                          topology_changed=True)
        cand.R = self.R[keep]
        return cand

    def add_resistive_edge(self, u: int, v: int, R_new: torch.Tensor):
        """Insert one resistive branch between EXISTING nodes. Rank 1."""
        changes = [BranchChange(u, v, KIND_G,
                                old=torch.zeros((), dtype=self.R.dtype),
                                new=1.0 / R_new)]
        row = torch.tensor([[u, v]], dtype=self.sys.r_edges.dtype)
        sys2 = replace(self.sys, r_edges=torch.cat([self.sys.r_edges, row], 0))
        cand = self._cand(f"add r_edge {u}-{v}", "edge_add", changes,
                          self.wt, self.wb, self.C_sites, sys_=sys2,
                          topology_changed=True)
        cand.R = torch.cat([self.R, R_new.reshape(1)])
        return cand

    # -- capacitive changes ----------------------------------------------
    def _decap_change(self, sites, new_vals, label, family):
        sites = np.atleast_1d(np.asarray(sites, dtype=int))
        C_new = self.C_sites.clone()
        C_new[sites] = new_vals
        changes = []
        for k in sites:
            u, v = self._c_endpoints(int(k))
            changes.append(BranchChange(u, v, KIND_C,
                                        old=self.C_sites[int(k)],
                                        new=C_new[int(k)]))
        return self._cand(label, family, changes, self.wt, self.wb, C_new)

    def decap_resize(self, site: int, factor: float):
        """Rank 1 at w > 0, rank 0 at DC (capacitors carry no DC admittance)."""
        return self._decap_change(site, self.C_sites[site] * factor,
                                  f"decap resize {site} x{factor:g}",
                                  "decap_resize")

    def decap_remove(self, site: int):
        return self._decap_change(site, torch.zeros((), dtype=self.C_sites.dtype),
                                  f"decap remove {site}", "decap_remove")

    def decap_add(self, site: int, value: torch.Tensor):
        """Populate an empty slot. Identical machinery to a resize from zero."""
        return self._decap_change(site, value, f"decap add {site}", "decap_add")

    def decap_move(self, src: int, dst: int):
        """Remove plus add — rank 2, and total capacitance is preserved."""
        sites = np.array([src, dst])
        vals = torch.stack([torch.zeros((), dtype=self.C_sites.dtype),
                            self.C_sites[dst] + self.C_sites[src]])
        return self._decap_change(sites, vals, f"decap move {src}->{dst}",
                                  "decap_move")

    def decap_multi(self, sites, factor: float):
        sites = np.atleast_1d(np.asarray(sites, dtype=int))
        return self._decap_change(sites, self.C_sites[sites] * factor,
                                  f"decap x{factor:g} on {sites.size}",
                                  "decap_multi")

    def decap_global(self, factor: float):
        """Every site at once. Rank = number of sites — NOT a low-rank action."""
        sites = np.arange(self.C_sites.shape[0])
        return self._decap_change(sites, self.C_sites * factor,
                                  f"decap global x{factor:g}", "decap_global")

    def decap_redistribute(self, take, give, frac: float):
        """Move a fraction of capacitance from one set of sites to another.

        Total capacitance is held FIXED, which is the action that separates
        "how much decap" from "where the decap is" — the distinction the
        pilot's global-scaling family could not express.
        """
        take = np.atleast_1d(np.asarray(take, dtype=int))
        give = np.atleast_1d(np.asarray(give, dtype=int))
        moved = self.C_sites[take].sum() * frac
        sites = np.concatenate([take, give])
        vals = torch.cat([self.C_sites[take] * (1.0 - frac),
                          self.C_sites[give] + moved / give.size])
        return self._decap_change(sites, vals,
                                  f"decap redistribute {frac:g} "
                                  f"{take.size}->{give.size}",
                                  "decap_redistribute")

    # -- source changes (rank 0) -----------------------------------------
    def load_scale(self, idx, factor: float):
        """Current sources do not enter ``Y``; rank 0, bypasses the update."""
        loads = self.loads.clone()
        loads[np.atleast_1d(np.asarray(idx, dtype=int)), 0] *= factor
        return self._cand(f"load I x{factor:g}", "load_one", [], self.wt,
                          self.wb, self.C_sites, loads=loads)

    def load_freq(self, factor: float):
        loads = self.loads.clone()
        loads[:, 1] *= factor
        return self._cand(f"load f x{factor:g}", "load_freq", [], self.wt,
                          self.wb, self.C_sites, loads=loads)


# ----------------------------------------------------------- group helpers

def column_groups(edges: np.ndarray, n: int) -> list[np.ndarray]:
    """Vertical-edge indices grouped by column — one full strap per group."""
    src, dst = edges[:, 0], edges[:, 1]
    vert = (dst - src) == n
    col = src % n
    return [idx for c in range(n)
            if (idx := np.nonzero(vert & (col == c))[0]).size]


def wire_section(group: np.ndarray, rng, frac: float = 1 / 3) -> np.ndarray:
    """A contiguous run of segments inside one strap."""
    k = max(2, int(round(group.size * frac)))
    st = int(rng.integers(max(1, group.size - k + 1)))
    return group[st:st + k]
