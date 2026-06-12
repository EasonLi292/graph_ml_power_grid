"""PyG schema constants for the two-rail PDN graph.

Kept in one place so every other submodule (normalizer, convs, encoder)
imports the exact same tuples — avoids per-relation key drift across
files.
"""
from __future__ import annotations

from tools.grid_construction import EDGE_ATTR_DIM


# Two node types. The Vdd / Vss distinction lives in the *feature* of
# each node (``is_vdd`` flag), not in the node type — every intersection
# in the layout corresponds to exactly one PyG node.
NODE_TYPES: tuple[str, ...] = ("mesh_top", "mesh_bot")


# Four logical relations:
#
#   * ``strap`` — within-layer same-net edges (vertical column straps +
#     same-net horizontal rungs). One relation per layer, bidir packed.
#   * ``via`` — cross-layer same-net edges. Two directed PyG relations
#     sharing the relation name.
#   * ``decap`` — cross-net edges on M_bot (Vdd↔Vss at the same row).
#     One bidir-packed mesh_bot ↔ mesh_bot relation.
#   * ``load`` — cross-net edges on M_bot. One bidir-packed mesh_bot ↔
#     mesh_bot relation. Carries the per-load (I, freq, duty, phase).
EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("mesh_top", "strap", "mesh_top"),
    ("mesh_bot", "strap", "mesh_bot"),
    ("mesh_top", "via", "mesh_bot"),
    ("mesh_bot", "via", "mesh_top"),
    ("mesh_bot", "decap", "mesh_bot"),
    ("mesh_bot", "load", "mesh_bot"),
)


# Normalized edge-attribute dimension: the raw 6 columns
# ``[R, C, I_peak, freq, duty, phase]`` become 7 after the load phase is
# encoded as ``(sin 2πφ, cos 2πφ)``.
EDGE_ATTR_DIM_NORMALIZED: int = EDGE_ATTR_DIM + 1
