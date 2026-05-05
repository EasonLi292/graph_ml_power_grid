"""Build heterogeneous PDN graphs.

The builder produces a framework-agnostic ``PDNGraph`` (numpy + scalars).
``to_hetero_data`` converts it to a PyG ``HeteroData`` for the ML pipeline
(lazy import — solver-side ground-truth generation does not need torch).

Indexing convention: row-major. Node ``i`` of an n×n mesh sits at
``(row, col) = (i // n, i % n)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PDNGraph:
    n_top: int
    n_bot: int

    Vdd: float
    R_top: float
    R_bot: float
    R_via: float
    C_decap: float

    I_peak: float
    freq: float
    duty: float
    phase: float

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


def build_regular_pdn(
    n_top: int = 4,
    n_bot: int = 7,
    Vdd: float = 1.0,
    R_top: float = 0.1,
    R_bot: float = 0.5,
    R_via: float = 0.05,
    C_decap: float = 1e-10,
    I_peak: float = 1e-3,
    freq: float = 1e9,
    duty: float = 0.5,
    phase: float = 0.0,
    load_stride: int = 2,
    decap_stride: int = 2,
    decap_offset: int = 1,
) -> PDNGraph:
    """Build the maximally-regular 2-layer PDN.

    Defaults: 4×4 M_top, 7×7 M_bot (via step=2 aligns cleanly), 4 corner pads,
    loads on the even sub-grid (16), decaps on the odd sub-grid (9). All
    R/C values and the load waveform are scalars shared across the mesh.
    """
    if n_top > 1 and (n_bot - 1) % (n_top - 1) != 0:
        raise ValueError(
            f"(n_bot-1) must be a multiple of (n_top-1) for clean via alignment; "
            f"got n_top={n_top}, n_bot={n_bot}."
        )

    top_edges = _mesh_edges(n_top)
    bot_edges = _mesh_edges(n_bot)

    bot_pitch = 1.0
    top_pitch = bot_pitch * (n_bot - 1) / max(n_top - 1, 1)
    top_pos = np.array(
        [[c * top_pitch, r * top_pitch] for r in range(n_top) for c in range(n_top)],
        dtype=float,
    )
    bot_pos = np.array(
        [[c * bot_pitch, r * bot_pitch] for r in range(n_bot) for c in range(n_bot)],
        dtype=float,
    )

    vdd_pad_top_idx = np.array(
        [0, n_top - 1, n_top * (n_top - 1), n_top * n_top - 1], dtype=int
    )

    step = (n_bot - 1) // max(n_top - 1, 1) if n_top > 1 else 0
    via_pairs = []
    for r in range(n_top):
        for c in range(n_top):
            ti = r * n_top + c
            br, bc = r * step, c * step
            via_pairs.append((ti, br * n_bot + bc))
    via_pairs = np.asarray(via_pairs, dtype=int)

    load_attach = [
        r * n_bot + c
        for r in range(0, n_bot, load_stride)
        for c in range(0, n_bot, load_stride)
    ]
    decap_attach = [
        r * n_bot + c
        for r in range(decap_offset, n_bot, decap_stride)
        for c in range(decap_offset, n_bot, decap_stride)
    ]

    return PDNGraph(
        n_top=n_top,
        n_bot=n_bot,
        Vdd=Vdd,
        R_top=R_top,
        R_bot=R_bot,
        R_via=R_via,
        C_decap=C_decap,
        I_peak=I_peak,
        freq=freq,
        duty=duty,
        phase=phase,
        vdd_pad_top_idx=vdd_pad_top_idx,
        via_pairs=via_pairs,
        load_attach_bot_idx=np.asarray(load_attach, dtype=int),
        decap_attach_bot_idx=np.asarray(decap_attach, dtype=int),
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
    data["load"].x = torch.from_numpy(
        np.tile(
            np.array([[g.I_peak, g.freq, g.duty, g.phase]], dtype=np.float32),
            (g.n_loads, 1),
        )
    )
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
