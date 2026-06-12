"""Per-relation convolutions for the heterogeneous PDN encoder.

* ``EdgeAwareConv`` — generic edge-conditioned MLP message. Kept as the
  baseline / ablation; ``EncoderConfig.conv_type = "edge_aware"`` selects
  this for every relation.
* ``AdmittanceConv`` — physics-shaped message passing. For relations
  whose physical law is ``i = Y · (v_j − v_i)`` (resistor strap, via,
  capacitor decap), the message is
  ``gate(edge_attr) ⊙ delta(x_j − x_i)`` — a Hadamard product separating
  "how strong is this edge" (gate) from "what is the differential I'm
  propagating" (delta). For load relations (current sources, not
  admittance), the generic edge-aware form is used.
* ``_make_conv`` — per-relation factory dispatching the right conv class
  + kind from ``EDGE_TYPES``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeAwareConv(MessagePassing):
    """Generic edge-conditioned message passing.

    msg_ij = MLP([x_i || x_j || edge_attr]); aggr = sum;
    upd_i  = MLP([x_i || agg_i]).
    Works on bipartite (src ≠ dst) edges via the standard PyG convention.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__(aggr="sum")
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # x is either Tensor (homogeneous) or (src, dst) tuple (bipartite)
        if isinstance(x, torch.Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.upd_mlp(torch.cat([x[1], out], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class AdmittanceConv(MessagePassing):
    """Physics-shaped message passing.

    Three flavours, selected at construction:

    * ``kind="conductance"`` — for **resistor** edges (strap, via),
      where the physical law is ``i = g · (v_j − v_i)`` with
      ``g = 1/R``. The gate is a *deterministic* function of R, not a
      learned MLP, so the conv cannot use R-magnitude as a topology
      fingerprint shortcut::

          gate_ij = exp(−α · z(log10 R_ij))           # scalar in [E, 1]
          msg_ij  = gate_ij · delta_mlp(x_j − x_i)

      ``z(·)`` is the analytic z-score from ``InputNormalizer`` (so the
      argument is mean-0 unit-std across the dataset's R range). ``α``
      is a single learnable scalar per conv that tunes the sharpness
      of the conductance dependence. At α=1 the gate is approximately
      ``1/R`` up to a constant.

    * ``kind="admittance"`` — for **decap** edges. C contributes a
      frequency-dependent reactive impedance whose effect on droop is
      messier than a single scalar; the gate stays a learned MLP::

          msg_ij = gate_mlp(edge_attr) ⊙ delta_mlp(x_j − x_i)

    * ``kind="source"`` — for **load** edges (current sources, not
      admittance). Same generic edge-aware form as ``EdgeAwareConv``.

    The update step is shared: ``upd_i = MLP([x_i || agg_i])``.
    """

    def __init__(self, hidden_dim: int, kind: str) -> None:
        super().__init__(aggr="sum")
        if kind not in ("admittance", "conductance", "source"):
            raise ValueError(f"unknown kind: {kind!r}")
        h = hidden_dim
        self.kind = kind
        if kind == "admittance":
            self.delta_mlp = nn.Sequential(
                nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)
            )
        elif kind == "conductance":
            self.delta_mlp = nn.Sequential(
                nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)
            )
            # Single learnable scalar. At α=1 the gate ≈ 1/R (the
            # actual physical conductance) up to a constant.
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:  # source
            self.msg_mlp = nn.Sequential(
                nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h)
            )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h)
        )

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.upd_mlp(torch.cat([x[1], out], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        if self.kind == "admittance":
            return self.gate_mlp(edge_attr) * self.delta_mlp(x_j - x_i)
        if self.kind == "conductance":
            # edge_attr is [E, 1] — the z-scored log10(R). Gate is a
            # deterministic function of R, no MLP. The model can only
            # learn α and the delta channels.
            gate = torch.exp(-self.alpha * edge_attr)
            return gate * self.delta_mlp(x_j - x_i)
        # source
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


def _make_conv(et: tuple[str, str, str], hidden_dim: int, conv_type: str) -> nn.Module:
    """Per-relation conv factory.

    For ``conv_type="admittance"`` (the post-revert physics setup):
      * **strap / via** → ``AdmittanceConv(kind="conductance")`` —
        deterministic gate ``exp(−α · z(log10 R))`` so the conv cannot
        use R-magnitude as a topology fingerprint.
      * **decap** → ``AdmittanceConv(kind="admittance")`` — learned
        gate from C (less leak there since C is a sample-level knob,
        not topology-derived).
      * **load** → ``AdmittanceConv(kind="source")``.

    For ``conv_type="edge_aware"``: every relation uses ``EdgeAwareConv``
    (baseline ablation).
    """
    if conv_type == "edge_aware":
        return EdgeAwareConv(hidden_dim)
    if conv_type == "admittance":
        rel = et[1]
        if rel in ("strap", "via"):
            return AdmittanceConv(hidden_dim, kind="conductance")
        if rel == "decap":
            return AdmittanceConv(hidden_dim, kind="admittance")
        if rel == "load":
            return AdmittanceConv(hidden_dim, kind="source")
        raise ValueError(f"unknown relation: {rel!r}")
    raise ValueError(f"unknown conv_type: {conv_type!r}")
