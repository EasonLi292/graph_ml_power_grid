"""Build heterogeneous PDN graphs.

The builder produces a framework-agnostic ``PDNGraph`` (numpy + scalars).
``to_hetero_data`` converts it to a PyG ``HeteroData`` for the ML pipeline
(lazy import — solver-side ground-truth generation does not need torch).

Indexing convention: row-major. Node ``i`` of an n×n mesh sits at
``(row, col) = (i // n, i % n)``.

Geometry vs. circuit
--------------------
Per-segment resistances ``R_top`` and ``R_bot`` are not free parameters of
the regular grid — they are derived from sheet resistance and strap
geometry::

    R_seg = Rsheet × (segment_length / wire_width)

where ``segment_length`` is the mesh pitch on that layer. The top mesh
is coarser than the bottom mesh, so ``pitch_top > pitch_bot`` and the
top-layer per-segment R picks up that geometric factor automatically.
``build_regular_pdn`` does this derivation; downstream code (solver, GNN)
sees the derived per-segment scalars.

Pad patterns
------------
``pad_pattern`` selects how Vdd is supplied to the top mesh. For the
default 4×4 M_top:

* ``"corner"``      — 4 corners (worst-case, far-from-pad center droop).
* ``"checker"``     — 8 alternating M_top nodes ((r + c) % 2 == 0).
* ``"edge_strip"``  — all 12 boundary nodes (no interior pads).
* ``"distributed"`` — 4 corners + interior 2×2 = 8 pads, mixed.

Loads that sit directly on a Vdd-pad via are filtered out — those nodes
are effectively tied to Vdd and don't represent realistic instance
placement. The number of surviving loads therefore varies per pattern.

Per-load heterogeneity
----------------------
Each load may draw its own ``(I_peak, freq, duty, phase)`` waveform. The
sample-level convention is a single global ``freq`` (one clock domain)
with ``(I_peak, duty, phase)`` varying per load. ``build_regular_pdn``
accepts either a precomputed ``loads`` array ``[n_loads, 4]`` or scalar
defaults that get broadcast to every load.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


PadPattern = Literal["corner", "checker", "edge_strip", "distributed"]
PAD_PATTERNS: tuple[PadPattern, ...] = ("corner", "checker", "edge_strip", "distributed")


@dataclass
class PDNGraph:
    n_top: int
    n_bot: int

    Vdd: float

    # Geometry (length units arbitrary; only ratios enter the circuit).
    pitch_top: float
    pitch_bot: float
    wire_width: float

    # Sheet resistance and the derived per-segment scalars actually stamped
    # by the solver / fed to the GNN.
    Rsheet_top: float
    Rsheet_bot: float
    R_top: float
    R_bot: float
    R_via: float
    C_decap: float

    # Per-load waveform parameters: [n_loads, 4] = (I_peak, freq, duty, phase).
    # ``freq`` is broadcast from the single global clock in the standard
    # dataset, but the solver doesn't assume that — it processes loads
    # independently.
    loads: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=float))

    pad_pattern: PadPattern = "corner"

    vdd_pad_top_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    via_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    load_attach_bot_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    decap_attach_bot_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    top_edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    bot_edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))

    top_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    bot_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    @property
    def n_top_nodes(self) -> int:
        return self.n_top * self.n_top

    @property
    def n_bot_nodes(self) -> int:
        return self.n_bot * self.n_bot

    @property
    def n_loads(self) -> int:
        return int(self.load_attach_bot_idx.shape[0])

    @property
    def n_decaps(self) -> int:
        return int(self.decap_attach_bot_idx.shape[0])


def _mesh_edges(n: int) -> np.ndarray:
    """Undirected (u, v) edges of an n×n mesh in row-major indexing."""
    edges = []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            if c + 1 < n:
                edges.append((i, i + 1))
            if r + 1 < n:
                edges.append((i, i + n))
    return np.asarray(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)


def _pad_indices(pattern: PadPattern, n_top: int) -> np.ndarray:
    """Top-mesh node indices that act as Vdd supply pads for ``pattern``."""
    if pattern == "corner":
        if n_top < 2:
            return np.array([0], dtype=int)
        return np.array(
            [0, n_top - 1, n_top * (n_top - 1), n_top * n_top - 1], dtype=int
        )
    if pattern == "checker":
        idx = [r * n_top + c for r in range(n_top) for c in range(n_top) if (r + c) % 2 == 0]
        return np.asarray(idx, dtype=int)
    if pattern == "edge_strip":
        idx = [
            r * n_top + c
            for r in range(n_top)
            for c in range(n_top)
            if r == 0 or r == n_top - 1 or c == 0 or c == n_top - 1
        ]
        return np.asarray(idx, dtype=int)
    if pattern == "distributed":
        # 4 corners + interior 2×2 (for n_top=4: corners {0,3,12,15} + interior
        # {5,6,9,10}). For other n_top, fall back to corners + a central
        # 2×2 block around (n_top/2, n_top/2).
        corners = [0, n_top - 1, n_top * (n_top - 1), n_top * n_top - 1] if n_top >= 2 else [0]
        c0 = (n_top - 2) // 2 if n_top >= 4 else 0
        interior = [(c0 + dr) * n_top + (c0 + dc) for dr in (0, 1) for dc in (0, 1)] if n_top >= 4 else []
        idx = sorted(set(corners + interior))
        return np.asarray(idx, dtype=int)
    raise ValueError(f"unknown pad_pattern: {pattern!r}")


def build_regular_pdn(
    n_top: int = 4,
    n_bot: int = 7,
    Vdd: float = 1.0,
    Rsheet_top: float = 0.05,
    Rsheet_bot: float = 0.2,
    wire_width: float = 0.5,
    R_via: float = 0.05,
    C_decap: float = 1e-10,
    I_peak: float = 1e-3,
    freq: float = 1e9,
    duty: float = 0.5,
    phase: float = 0.0,
    loads: np.ndarray | None = None,
    pad_pattern: PadPattern = "corner",
    load_stride: int = 2,
    decap_stride: int = 2,
    decap_offset: int = 1,
) -> PDNGraph:
    """Build the maximally-regular 2-layer PDN.

    Defaults: 4×4 M_top, 7×7 M_bot (via step=2 aligns cleanly), 4 corner
    pads, loads on the even M_bot sub-grid, decaps on the odd sub-grid.
    Loads coincident with a Vdd-pad via are removed.

    Per-segment R is derived from sheet resistance + geometry::

        R_top = Rsheet_top × (pitch_top / wire_width)
        R_bot = Rsheet_bot × (pitch_bot / wire_width)

    If ``loads`` is supplied, it must be a ``[n_loads_after_filtering, 4]``
    array of per-load ``(I_peak, freq, duty, phase)``. Otherwise the
    scalar defaults are broadcast to every load.
    """
    if n_top > 1 and (n_bot - 1) % (n_top - 1) != 0:
        raise ValueError(
            f"(n_bot-1) must be a multiple of (n_top-1) for clean via alignment; "
            f"got n_top={n_top}, n_bot={n_bot}."
        )
    if pad_pattern not in PAD_PATTERNS:
        raise ValueError(f"pad_pattern must be one of {PAD_PATTERNS}, got {pad_pattern!r}")
    if wire_width <= 0:
        raise ValueError(f"wire_width must be positive, got {wire_width}")

    top_edges = _mesh_edges(n_top)
    bot_edges = _mesh_edges(n_bot)

    pitch_bot = 1.0
    pitch_top = pitch_bot * (n_bot - 1) / max(n_top - 1, 1)
    R_top = Rsheet_top * (pitch_top / wire_width)
    R_bot = Rsheet_bot * (pitch_bot / wire_width)

    top_pos = np.array(
        [[c * pitch_top, r * pitch_top] for r in range(n_top) for c in range(n_top)],
        dtype=float,
    )
    bot_pos = np.array(
        [[c * pitch_bot, r * pitch_bot] for r in range(n_bot) for c in range(n_bot)],
        dtype=float,
    )

    vdd_pad_top_idx = _pad_indices(pad_pattern, n_top)

    step = (n_bot - 1) // max(n_top - 1, 1) if n_top > 1 else 0
    via_pairs = []
    for r in range(n_top):
        for c in range(n_top):
            ti = r * n_top + c
            br, bc = r * step, c * step
            via_pairs.append((ti, br * n_bot + bc))
    via_pairs = np.asarray(via_pairs, dtype=int)

    pad_set = set(int(i) for i in vdd_pad_top_idx)
    pad_via_bot = {int(b) for t, b in via_pairs if int(t) in pad_set}

    load_lattice = [
        r * n_bot + c
        for r in range(0, n_bot, load_stride)
        for c in range(0, n_bot, load_stride)
    ]
    load_attach = np.asarray(
        [i for i in load_lattice if i not in pad_via_bot], dtype=int
    )

    decap_attach = np.asarray(
        [
            r * n_bot + c
            for r in range(decap_offset, n_bot, decap_stride)
            for c in range(decap_offset, n_bot, decap_stride)
        ],
        dtype=int,
    )

    n_loads = load_attach.shape[0]
    if loads is None:
        loads_arr = np.tile(
            np.array([[I_peak, freq, duty, phase]], dtype=float), (n_loads, 1)
        )
    else:
        loads_arr = np.asarray(loads, dtype=float)
        if loads_arr.shape != (n_loads, 4):
            raise ValueError(
                f"loads shape {loads_arr.shape} != expected ({n_loads}, 4) for "
                f"pad_pattern={pad_pattern!r}"
            )

    return PDNGraph(
        n_top=n_top,
        n_bot=n_bot,
        Vdd=Vdd,
        pitch_top=pitch_top,
        pitch_bot=pitch_bot,
        wire_width=wire_width,
        Rsheet_top=Rsheet_top,
        Rsheet_bot=Rsheet_bot,
        R_top=R_top,
        R_bot=R_bot,
        R_via=R_via,
        C_decap=C_decap,
        loads=loads_arr,
        pad_pattern=pad_pattern,
        vdd_pad_top_idx=vdd_pad_top_idx,
        via_pairs=via_pairs,
        load_attach_bot_idx=load_attach,
        decap_attach_bot_idx=decap_attach,
        top_edges=top_edges,
        bot_edges=bot_edges,
        top_pos=top_pos,
        bot_pos=bot_pos,
    )


def to_hetero_data(g: PDNGraph):
    """Convert a ``PDNGraph`` to a ``torch_geometric.data.HeteroData``.

    Resistor edges are emitted bidirectionally so the encoder can pass
    messages along straps in either direction; capacitor and load edges
    are directed (gnd is always the sink).
    """
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()

    # ----- nodes -----
    is_pad = np.zeros(g.n_top_nodes, dtype=np.float32)
    is_pad[g.vdd_pad_top_idx] = 1.0
    data["mesh_top"].x = torch.from_numpy(
        np.column_stack([g.top_pos.astype(np.float32), is_pad])
    )
    data["mesh_bot"].x = torch.from_numpy(g.bot_pos.astype(np.float32))
    data["load"].x = torch.from_numpy(g.loads.astype(np.float32))
    data["gnd"].x = torch.zeros((1, 1), dtype=torch.float32)

    def _bidir(edges: np.ndarray, value: float):
        u, v = edges[:, 0], edges[:, 1]
        ei = np.stack([np.concatenate([u, v]), np.concatenate([v, u])], axis=0)
        ea = np.full((ei.shape[1], 1), value, dtype=np.float32)
        return torch.from_numpy(ei.astype(np.int64)), torch.from_numpy(ea)

    ei, ea = _bidir(g.top_edges, g.R_top)
    data["mesh_top", "R_top", "mesh_top"].edge_index = ei
    data["mesh_top", "R_top", "mesh_top"].edge_attr = ea

    ei, ea = _bidir(g.bot_edges, g.R_bot)
    data["mesh_bot", "R_bot", "mesh_bot"].edge_index = ei
    data["mesh_bot", "R_bot", "mesh_bot"].edge_attr = ea

    via_top = g.via_pairs[:, 0].astype(np.int64)
    via_bot = g.via_pairs[:, 1].astype(np.int64)
    via_attr = torch.full((via_top.size, 1), g.R_via, dtype=torch.float32)
    data["mesh_top", "R_via", "mesh_bot"].edge_index = torch.from_numpy(
        np.stack([via_top, via_bot])
    )
    data["mesh_top", "R_via", "mesh_bot"].edge_attr = via_attr.clone()
    data["mesh_bot", "R_via", "mesh_top"].edge_index = torch.from_numpy(
        np.stack([via_bot, via_top])
    )
    data["mesh_bot", "R_via", "mesh_top"].edge_attr = via_attr.clone()

    decap_src = g.decap_attach_bot_idx.astype(np.int64)
    decap_dst = np.zeros_like(decap_src)
    decap_attr = torch.full((decap_src.size, 1), g.C_decap, dtype=torch.float32)
    data["mesh_bot", "C_decap", "gnd"].edge_index = torch.from_numpy(
        np.stack([decap_src, decap_dst])
    )
    data["mesh_bot", "C_decap", "gnd"].edge_attr = decap_attr.clone()
    # Reverse so mesh_bot also receives the C_decap signal — without this,
    # mesh_bot has no path by which decap capacitance reaches its prediction.
    data["gnd", "C_decap_rev", "mesh_bot"].edge_index = torch.from_numpy(
        np.stack([decap_dst, decap_src])
    )
    data["gnd", "C_decap_rev", "mesh_bot"].edge_attr = decap_attr.clone()

    load_ids = np.arange(g.n_loads, dtype=np.int64)
    load_attach = g.load_attach_bot_idx.astype(np.int64)
    data["mesh_bot", "I_in", "load"].edge_index = torch.from_numpy(
        np.stack([load_attach, load_ids])
    )
    # Reverse so load.x (I_peak, freq, duty, phase) propagates to mesh_bot.
    data["load", "I_in_rev", "mesh_bot"].edge_index = torch.from_numpy(
        np.stack([load_ids, load_attach])
    )

    data["load", "I_out", "gnd"].edge_index = torch.from_numpy(
        np.stack([load_ids, np.zeros_like(load_ids)])
    )
    data["gnd", "I_out_rev", "load"].edge_index = torch.from_numpy(
        np.stack([np.zeros_like(load_ids), load_ids])
    )

    return data
