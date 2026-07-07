"""Build the heterogeneous PDN graph for the two-rail (Vdd + Vss) design.

Physical model
--------------
Two metal layers, ``M_top`` (coarse, ``n_top × n_top`` intersections) and
``M_bot`` (fine, ``n_bot × n_bot``). Both layers carry strips running in
the same direction (vertical); each column is one net (Vdd or Vss) end
to end. This deliberately avoids the perpendicular-strip aliasing trap
where the step-N via mapping only hits one net's columns.

Bot-mesh column pattern is fixed at the period-4 pattern ``V V G G V V G``
for ``n_bot = 7`` (cols 0,1,4,5 Vdd; cols 2,3,6 Vss). The top-mesh column
pattern is derived from the bot pattern by sampling at the via step::

    step = (n_bot - 1) // (n_top - 1)
    top_col_net[c] = bot_col_net[c * step]

so vias are *always same-net* by construction, for every valid ``n_top``
in ``{2, 3, 4, 7}``.

Within-layer edges
~~~~~~~~~~~~~~~~~~
* **Vertical straps** between consecutive rows of the same column (always
  same net).
* **Horizontal rungs** between adjacent columns at the same row, only
  when both columns are the same net. The rungs give a 2D mesh per net
  for current spreading; Vdd↔Vss adjacencies have no rung (that would
  short the supply).

Cross-layer / cross-net edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Vias** at ``(top_r, top_c) ↔ (top_r·step, top_c·step)``. Same-net by
  construction.
* **Decaps** on ``M_bot`` between every Vdd-Vss column pair at every
  row in the decap stride set. Period-4 pattern → boundary col pairs
  ``(1,2), (3,4), (5,6)``.
* **Loads** on ``M_bot`` at the same column-pair boundaries, on a
  *different* row stride set so loads and decaps don't sit on top of
  each other.

Pads
~~~~
Every row-0 node on ``M_top`` is a bump (``is_pad = 1``) — one bump per
column. Their nets are read from the column pattern, so each Vdd column
gets a Vdd bump and each Vss column gets a Vss bump. (Using only "4
corner" bumps would leave one of the nets entirely floating at most
``n_top`` values for our bot pattern.)

Returns: a framework-agnostic ``PDNGraph`` (numpy + scalars).
``to_hetero_data`` converts it to PyG ``HeteroData`` (lazy import).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Bot column pattern per die size (1 = Vdd, 0 = Vss). One fixed pattern
# per ``n_bot``, so topology within a die size varies only through
# ``n_top`` (top track density).
#
# * ``n_bot=7`` — V V G G V V V. The "3-cluster" choice (Vdd{0,1},
#   Vss{2,3}, Vdd{4,5,6}) gives 2 cross-net boundaries per row (14 loads)
#   while keeping every cluster reachable from the via tap-sets at
#   n_top ∈ {3, 4, 7}::
#
#     n_top=7 (step=1)  taps  {0,1,2,3,4,5,6}  → every cluster ✓
#     n_top=4 (step=2)  taps  {0, 2, 4, 6}     → every cluster ✓
#     n_top=3 (step=3)  taps  {0, 3, 6}        → every cluster ✓
#
#   n_top=2 (step=6) is *not* supported (taps {0, 6} are both Vdd, so
#   the Vss cluster {2,3} would float).
#
# * ``n_bot=13`` — the 7-pattern tiled: V V G G V V V V G G V V V
#   (5 clusters, 4 cross-net boundaries per row → 52 loads). Valid
#   n_top: {5, 7, 13} — verified every cluster is tapped. n_top=4
#   (step=4, taps {0,4,8,12}) misses Vss cluster {2,3} → floating DC
#   island; n_top ∈ {2, 3} tap only Vdd. ``_top_col_pattern`` raises on
#   those rather than silently building a singular grid.
BOT_COL_PATTERNS: dict[int, tuple[int, ...]] = {
    7:  (1, 1, 0, 0, 1, 1, 1),
    13: (1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1),
}
# Back-compat alias (the original single-die-size constant).
BOT_COL_PATTERN: tuple[int, ...] = BOT_COL_PATTERNS[7]

# Stride-1, offset-0 on both: every row carries one load and one decap
# at the single Vdd↔Vss boundary. 1 boundary × 7 rows = 7 of each.
# Loads and decaps overlap at the same (row, boundary), which is fine
# electrically — they're two parallel branches between the same Vdd
# and Vss bot nodes, the way a real cell row mixes switching cells and
# decap fillers.
DECAP_ROW_STRIDE: int = 1
DECAP_ROW_OFFSET: int = 0
LOAD_ROW_STRIDE: int = 1
LOAD_ROW_OFFSET: int = 0


@dataclass
class PDNGraph:
    n_top: int
    n_bot: int
    Vdd: float

    # Geometry (length units arbitrary; only ratios enter the circuit).
    pitch_top: float
    pitch_bot: float
    wire_width: float

    # Sheet resistance + derived per-segment scalars actually stamped by
    # the solver / fed to the GNN.
    Rsheet_top: float
    Rsheet_bot: float
    R_top: float
    R_bot: float
    R_via: float
    C_decap: float

    # Per-load waveforms: [n_loads, 4] = (I_peak, freq, duty, phase).
    loads: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=float))

    # Per-node net assignment: 1 = Vdd, 0 = Vss. Index matches row-major
    # node ordering within each mesh.
    top_is_vdd: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    bot_is_vdd: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))

    # Per-node pad flag (1 only at the corner bumps of M_top).
    top_is_pad: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))

    # Top-mesh node indices that act as Vdd / Vss bumps (subsets of the
    # 4 corners). Kept as separate arrays so the solver can clamp them
    # to Vdd / 0 respectively.
    vdd_pad_top_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    vss_pad_top_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    # Within-layer strap edges (vertical + same-net rungs). Both arrays
    # are (E, 2) of (u, v) endpoint indices into the per-layer node list.
    top_edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    bot_edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))

    # Cross-layer vias: (top_idx, bot_idx) pairs.
    via_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))

    # Cross-net decap and load endpoints (Vdd-side, Vss-side) on M_bot.
    # Each row k is one device between bot[u] (Vdd) and bot[v] (Vss).
    decap_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    load_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))

    # Shared positions per mesh.
    top_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    bot_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    # Optional PER-EDGE strap resistances (heterogeneous wire width). When
    # set, these override the scalar ``R_top`` / ``R_bot`` everywhere R is
    # consumed (solver stamps, GNN edge_attr). Index-aligned with
    # ``top_edges`` / ``bot_edges`` (one R per *physical* edge, before the
    # bidirectional packing in ``to_hetero_data``). ``None`` ⇒ uniform R.
    R_top_edges: np.ndarray | None = None
    R_bot_edges: np.ndarray | None = None

    @property
    def n_top_nodes(self) -> int:
        return self.n_top * self.n_top

    @property
    def n_bot_nodes(self) -> int:
        return self.n_bot * self.n_bot

    @property
    def n_loads(self) -> int:
        return int(self.load_pairs.shape[0])

    @property
    def n_decaps(self) -> int:
        return int(self.decap_pairs.shape[0])


def _bot_col_pattern(n_bot: int) -> np.ndarray:
    """Return the fixed Vdd/Vss column pattern for the bot mesh."""
    if n_bot not in BOT_COL_PATTERNS:
        raise ValueError(
            f"no bot column pattern defined for n_bot={n_bot}; "
            f"supported: {sorted(BOT_COL_PATTERNS)}. Extend BOT_COL_PATTERNS "
            f"if you add a die size."
        )
    return np.asarray(BOT_COL_PATTERNS[n_bot], dtype=np.int8)


def _top_col_pattern(n_top: int, n_bot: int, bot_pat: np.ndarray) -> np.ndarray:
    """Sample the bot column pattern at the via step to get top pattern.

    Same-net via alignment is automatic: top col c shares net with bot col
    ``c * step``. Raises if any same-net column *cluster* of the bot
    pattern has no via tap — an untapped cluster has no DC path to any
    pad (its only connections are cross-net load/decap branches), which
    makes the operating-point solve singular.
    """
    if n_top > 1 and (n_bot - 1) % (n_top - 1) != 0:
        raise ValueError(
            f"(n_bot - 1) must be a multiple of (n_top - 1); got n_top={n_top}, n_bot={n_bot}"
        )
    step = (n_bot - 1) // max(n_top - 1, 1)
    taps = {c * step for c in range(n_top)}
    start = 0
    for i in range(1, n_bot + 1):
        if i == n_bot or bot_pat[i] != bot_pat[start]:
            cluster = set(range(start, i))
            if not cluster & taps:
                raise ValueError(
                    f"n_top={n_top} on n_bot={n_bot}: via taps {sorted(taps)} miss "
                    f"the {'Vdd' if bot_pat[start] else 'Vss'} cluster cols "
                    f"{sorted(cluster)} — that net segment would float (no DC "
                    f"path to a pad). Pick an n_top whose tap-set covers every "
                    f"cluster."
                )
            start = i
    return np.asarray([bot_pat[c * step] for c in range(n_top)], dtype=np.int8)


def _per_layer_edges(n: int, col_is_vdd: np.ndarray) -> np.ndarray:
    """Vertical column straps + horizontal same-net rungs for an n×n mesh.

    * Vertical: every (r, c) ↔ (r+1, c). Always same-net (one column = one net).
    * Horizontal: (r, c) ↔ (r, c+1) only if columns c and c+1 are the same net.
    """
    edges: list[tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            # vertical
            if r + 1 < n:
                edges.append((i, i + n))
            # horizontal rung, same-net only
            if c + 1 < n and col_is_vdd[c] == col_is_vdd[c + 1]:
                edges.append((i, i + 1))
    return np.asarray(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)


def _via_pairs(n_top: int, n_bot: int) -> np.ndarray:
    """Top↔bot via pairs at the step mapping (same-net by construction)."""
    step = (n_bot - 1) // max(n_top - 1, 1) if n_top > 1 else 0
    pairs: list[tuple[int, int]] = []
    for r in range(n_top):
        for c in range(n_top):
            ti = r * n_top + c
            bi = (r * step) * n_bot + (c * step)
            pairs.append((ti, bi))
    return np.asarray(pairs, dtype=int)


def _cross_net_pairs_on_bot(
    n_bot: int,
    bot_col_is_vdd: np.ndarray,
    row_stride: int,
    row_offset: int,
) -> np.ndarray:
    """Cross-net (Vdd, Vss) bot node pairs at Vdd→Vss column boundaries.

    For our period-4 bot pattern, the boundaries are (cols (1,2), (3,4),
    (5,6)). At every row in the chosen stride set, emit one pair per
    boundary with the Vdd col first.
    """
    pairs: list[tuple[int, int]] = []
    boundaries: list[tuple[int, int]] = []
    for c in range(n_bot - 1):
        a, b = bot_col_is_vdd[c], bot_col_is_vdd[c + 1]
        if a != b:
            vdd_c, vss_c = (c, c + 1) if a == 1 else (c + 1, c)
            boundaries.append((vdd_c, vss_c))
    for r in range(row_offset, n_bot, row_stride):
        for vdd_c, vss_c in boundaries:
            pairs.append((r * n_bot + vdd_c, r * n_bot + vss_c))
    return np.asarray(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)


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
    ww_top_edges: np.ndarray | None = None,
    ww_bot_edges: np.ndarray | None = None,
) -> PDNGraph:
    """Build the two-rail PDN graph.

    Per-segment R is derived from sheet resistance + geometry::

        R_top = Rsheet_top × (pitch_top / wire_width)
        R_bot = Rsheet_bot × (pitch_bot / wire_width)

    ``loads`` is an optional ``[n_loads, 4]`` array of per-load
    ``(I_peak, freq, duty, phase)``; if omitted, the scalar defaults are
    broadcast to every load.

    ``ww_top_edges`` / ``ww_bot_edges`` are optional per-edge wire widths
    (heterogeneous metal). When given they must be index-aligned with the
    layer's ``*_edges`` array; each strap edge then gets its own
    ``R = Rsheet × pitch / width_edge``. The scalar ``wire_width`` still
    sets the uniform fallback and the warmup-relevant ``R_bot`` is taken
    at the *thinnest* bot edge (largest R, slowest τ) so labels never
    under-warm.
    """
    if wire_width <= 0:
        raise ValueError(f"wire_width must be positive, got {wire_width}")

    bot_pat = _bot_col_pattern(n_bot)
    top_pat = _top_col_pattern(n_top, n_bot, bot_pat)

    # Per-node net assignment (row-major).
    top_is_vdd = np.tile(top_pat, n_top).reshape(n_top, n_top).ravel()
    bot_is_vdd = np.tile(bot_pat, n_bot).reshape(n_bot, n_bot).ravel()

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

    # Pad bumps at row 0 of every top column — one bump per column. Each
    # bump's net follows the column's net pattern, so every net is
    # guaranteed at least one clamp regardless of the column pattern.
    pad_nodes = np.arange(n_top, dtype=int)  # row-0 nodes of an n_top × n_top mesh
    top_is_pad = np.zeros(n_top * n_top, dtype=np.int8)
    top_is_pad[pad_nodes] = 1
    vdd_pad_top_idx = pad_nodes[top_is_vdd[pad_nodes] == 1].astype(int)
    vss_pad_top_idx = pad_nodes[top_is_vdd[pad_nodes] == 0].astype(int)

    top_edges = _per_layer_edges(n_top, top_pat)
    bot_edges = _per_layer_edges(n_bot, bot_pat)
    via_pairs = _via_pairs(n_top, n_bot)

    # Optional per-edge (heterogeneous) strap resistances.
    R_top_edges = R_bot_edges = None
    if ww_top_edges is not None:
        ww_top_edges = np.asarray(ww_top_edges, dtype=float)
        if ww_top_edges.shape[0] != top_edges.shape[0]:
            raise ValueError(
                f"ww_top_edges length {ww_top_edges.shape[0]} != "
                f"#top_edges {top_edges.shape[0]}"
            )
        if np.any(ww_top_edges <= 0):
            raise ValueError("ww_top_edges must be positive")
        R_top_edges = Rsheet_top * (pitch_top / ww_top_edges)
    if ww_bot_edges is not None:
        ww_bot_edges = np.asarray(ww_bot_edges, dtype=float)
        if ww_bot_edges.shape[0] != bot_edges.shape[0]:
            raise ValueError(
                f"ww_bot_edges length {ww_bot_edges.shape[0]} != "
                f"#bot_edges {bot_edges.shape[0]}"
            )
        if np.any(ww_bot_edges <= 0):
            raise ValueError("ww_bot_edges must be positive")
        R_bot_edges = Rsheet_bot * (pitch_bot / ww_bot_edges)
        # Conservative scalar R_bot (thinnest edge → slowest τ) for warmup.
        R_bot = float(R_bot_edges.max())

    decap_pairs = _cross_net_pairs_on_bot(
        n_bot, bot_pat, DECAP_ROW_STRIDE, DECAP_ROW_OFFSET
    )
    load_pairs = _cross_net_pairs_on_bot(
        n_bot, bot_pat, LOAD_ROW_STRIDE, LOAD_ROW_OFFSET
    )

    n_loads = load_pairs.shape[0]
    if loads is None:
        loads_arr = np.tile(
            np.array([[I_peak, freq, duty, phase]], dtype=float), (n_loads, 1)
        )
    else:
        loads_arr = np.asarray(loads, dtype=float)
        if loads_arr.shape != (n_loads, 4):
            raise ValueError(
                f"loads shape {loads_arr.shape} != expected ({n_loads}, 4)"
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
        top_is_vdd=top_is_vdd.astype(np.int8),
        bot_is_vdd=bot_is_vdd.astype(np.int8),
        top_is_pad=top_is_pad,
        vdd_pad_top_idx=vdd_pad_top_idx,
        vss_pad_top_idx=vss_pad_top_idx,
        top_edges=top_edges,
        bot_edges=bot_edges,
        via_pairs=via_pairs,
        decap_pairs=decap_pairs,
        load_pairs=load_pairs,
        top_pos=top_pos,
        bot_pos=bot_pos,
        R_top_edges=R_top_edges,
        R_bot_edges=R_bot_edges,
    )


# Edge attribute layout, shared across every relation:
#   col 0 — R (resistance, Ω)  — non-zero for strap and via
#   col 1 — C (capacitance, F) — non-zero for decap
#   col 2 — I_peak (A)        ┐
#   col 3 — freq (Hz)         │
#   col 4 — duty (fraction)   │ non-zero for load
#   col 5 — phase (∈ [0, 1])  ┘
EDGE_ATTR_DIM = 6
EDGE_ATTR_COLS = ("R", "C", "I_peak", "freq", "duty", "phase")

# Node feature layout: [is_vdd, is_pad]. Layer identity is already encoded
# by the heterogeneous node type (mesh_top vs mesh_bot), and physical
# adjacency / segment scale are carried by edge_index and the per-segment
# resistances — so coordinates and a layer one-hot would be redundant
# inputs the model could overfit to. We expose only the rail (``is_vdd``)
# and the boundary-condition marker (``is_pad``).
NODE_FEATURE_DIM = 2


def to_hetero_data(g: PDNGraph):
    """Convert a ``PDNGraph`` to a ``torch_geometric.data.HeteroData``.

    Two node types — ``mesh_top``, ``mesh_bot`` — and four logical
    relations carrying the same 6-dim edge attribute schema
    ``EDGE_ATTR_COLS``::

        within-layer same-net mesh:
            mesh_top ↔ mesh_top   strap  (R only)
            mesh_bot ↔ mesh_bot   strap  (R only)

        cross-layer same-net via:
            mesh_top ↔ mesh_bot   via    (R only)

        cross-net devices (on M_bot, Vdd-Vss pairs at the same row):
            mesh_bot ↔ mesh_bot   decap  (C only)
            mesh_bot ↔ mesh_bot   load   (I_peak, freq, duty, phase)

    Same-layer bidir is packed into a single ``edge_index``; cross-layer
    via is two PyG relations sharing the ``via`` name. Decap and load
    each live within ``mesh_bot`` and are also packed bidir (the
    encoder's message passing is symmetric in src/dst).
    """
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()

    def _node_features(is_vdd: np.ndarray, is_pad: np.ndarray) -> np.ndarray:
        # [is_vdd, is_pad]. Layer identity comes from the node type, and
        # connectivity / scale from edge_index + resistances, so neither
        # coordinates nor a layer one-hot are exposed to the GNN.
        return np.column_stack([
            is_vdd.astype(np.float32),
            is_pad.astype(np.float32),
        ])

    data["mesh_top"].x = torch.from_numpy(
        _node_features(g.top_is_vdd, g.top_is_pad)
    )
    # mesh_bot has no pads, so its ``is_pad`` column is all zeros.
    data["mesh_bot"].x = torch.from_numpy(
        _node_features(g.bot_is_vdd, np.zeros(g.n_bot_nodes, dtype=np.int8))
    )

    def _const_attr(n: int, **kwargs) -> np.ndarray:
        a = np.zeros((n, EDGE_ATTR_DIM), dtype=np.float32)
        for k, v in kwargs.items():
            a[:, EDGE_ATTR_COLS.index(k)] = v
        return a

    def _load_attr(loads: np.ndarray) -> np.ndarray:
        a = np.zeros((loads.shape[0], EDGE_ATTR_DIM), dtype=np.float32)
        a[:, EDGE_ATTR_COLS.index("I_peak")] = loads[:, 0]
        a[:, EDGE_ATTR_COLS.index("freq")] = loads[:, 1]
        a[:, EDGE_ATTR_COLS.index("duty")] = loads[:, 2]
        a[:, EDGE_ATTR_COLS.index("phase")] = loads[:, 3]
        return a

    def _bidir(pairs: np.ndarray) -> np.ndarray:
        u, v = pairs[:, 0], pairs[:, 1]
        return np.stack([np.concatenate([u, v]), np.concatenate([v, u])], axis=0)

    def _strap_attr(n_dir_edges: int, R_scalar: float, R_edges: np.ndarray | None) -> np.ndarray:
        """Edge-attr with R in col 0. Per-edge R (index-aligned with the
        physical edges) is bidir-tiled to match ``_bidir`` packing; else
        the scalar is broadcast."""
        a = np.zeros((n_dir_edges, EDGE_ATTR_DIM), dtype=np.float32)
        if R_edges is not None:
            a[:, EDGE_ATTR_COLS.index("R")] = np.concatenate([R_edges, R_edges])
        else:
            a[:, EDGE_ATTR_COLS.index("R")] = R_scalar
        return a

    # ----- strap edges (within-layer same-net, bidir packed) -----
    ei_top = _bidir(g.top_edges).astype(np.int64) if g.top_edges.size else np.empty((2, 0), dtype=np.int64)
    data["mesh_top", "strap", "mesh_top"].edge_index = torch.from_numpy(ei_top)
    data["mesh_top", "strap", "mesh_top"].edge_attr = torch.from_numpy(
        _strap_attr(ei_top.shape[1], g.R_top, g.R_top_edges)
    )
    ei_bot = _bidir(g.bot_edges).astype(np.int64) if g.bot_edges.size else np.empty((2, 0), dtype=np.int64)
    data["mesh_bot", "strap", "mesh_bot"].edge_index = torch.from_numpy(ei_bot)
    data["mesh_bot", "strap", "mesh_bot"].edge_attr = torch.from_numpy(
        _strap_attr(ei_bot.shape[1], g.R_bot, g.R_bot_edges)
    )

    # ----- via edges (top ↔ bot, two directed relations sharing 'via') -----
    via_top = g.via_pairs[:, 0].astype(np.int64)
    via_bot = g.via_pairs[:, 1].astype(np.int64)
    via_attr = _const_attr(via_top.size, R=g.R_via)
    data["mesh_top", "via", "mesh_bot"].edge_index = torch.from_numpy(np.stack([via_top, via_bot]))
    data["mesh_top", "via", "mesh_bot"].edge_attr = torch.from_numpy(via_attr.copy())
    data["mesh_bot", "via", "mesh_top"].edge_index = torch.from_numpy(np.stack([via_bot, via_top]))
    data["mesh_bot", "via", "mesh_top"].edge_attr = torch.from_numpy(via_attr.copy())

    # ----- decap edges (cross-net within M_bot, bidir packed) -----
    if g.n_decaps > 0:
        ei_dec = _bidir(g.decap_pairs).astype(np.int64)
        dec_attr_one_dir = _const_attr(g.n_decaps, C=g.C_decap)
        dec_attr = np.concatenate([dec_attr_one_dir, dec_attr_one_dir], axis=0)
    else:
        ei_dec = np.empty((2, 0), dtype=np.int64)
        dec_attr = np.empty((0, EDGE_ATTR_DIM), dtype=np.float32)
    data["mesh_bot", "decap", "mesh_bot"].edge_index = torch.from_numpy(ei_dec)
    data["mesh_bot", "decap", "mesh_bot"].edge_attr = torch.from_numpy(dec_attr)

    # ----- load edges (cross-net within M_bot, directed Vdd→Vss) -----
    # Load relation is directed (not bidir-packed) so PyG auto-offsets
    # the edge_index correctly under batching, and the regressor head
    # can read ``edge_index[0]`` for Vdd-side endpoints and
    # ``edge_index[1]`` for Vss-side endpoints. The load is physically a
    # current source from Vdd to Vss, so a single-direction message —
    # "Vdd-side has this current draw" → Vss-side — captures the load
    # information; the GNN can propagate it further via the symmetric
    # strap / decap relations.
    if g.n_loads > 0:
        load_src = g.load_pairs[:, 0].astype(np.int64)
        load_dst = g.load_pairs[:, 1].astype(np.int64)
        ei_ld = np.stack([load_src, load_dst])
        ld_attr = _load_attr(g.loads.astype(np.float32))
    else:
        ei_ld = np.empty((2, 0), dtype=np.int64)
        ld_attr = np.empty((0, EDGE_ATTR_DIM), dtype=np.float32)
    data["mesh_bot", "load", "mesh_bot"].edge_index = torch.from_numpy(ei_ld)
    data["mesh_bot", "load", "mesh_bot"].edge_attr = torch.from_numpy(ld_attr)

    return data
