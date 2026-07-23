"""Patch dataset over the IBM transient per-node graphs.

Turns each big ``datasets/ibmpg/graphs/<bench>.npz`` graph (25k–1.5M
nodes) into many training patches. Because the grids are near-planar
meshes (degree ≈ 4), exact receptive fields are cheap: a patch is a BFS
ball of radius ``seed_radius`` (the *seed* nodes, which carry the loss)
plus a further ``halo`` hops of context (= model depth), so every seed
sees its complete ``halo``-hop neighborhood — no sampled-fanout
truncation bias.

Preprocessing per graph:

* **Short merge** — resistors below ``r_short`` (pg3t has a 6.5e-10 Ω
  edge) are contracted (union-find), parallel edges re-collapsed by
  conductance sum. Kills the 12-decade R range without changing the
  electrical answer at the µV level.
* **Node features** (13-dim): one-hot metal layer (6), ``is_clamp``,
  ``net_vdd``, ``has_load`` + z(log10 I), ``has_cap`` + z(log10 C),
  z(log10 static drop). The static drop (from ``v_dc``) is the global
  context a k-hop patch cannot see — effective resistance to the pads
  from one cheap linear solve.
* **Targets**: ``log10(max(droop, 1e-6))`` per node; loss mask =
  seed ∩ ``is_grid``.

Normalization stats are computed over the *training* benchmarks only
(pass the same ``FeatureStats`` to held-out grids).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GRAPHS_DIR = Path("datasets/ibmpg/graphs")
R_SHORT = 1e-6          # ohms; contract resistor edges below this
LOG_EPS = 1e-12
DROOP_FLOOR = 1e-6      # volts; floor before log10 target
SDROP_FLOOR = 1e-6      # volts; floor for the static-drop feature
N_LAYER_ONEHOT = 7      # metal layers 0..6 across the benchmark set
N_GLOBAL_FEATS = 6      # per-grid scalars broadcast to every node
N_LOCAL_FEATS = 8       # clamp, vdd, has_load, zI, has_cap, zSd, has_rg, zGg
NODE_FEATURE_DIM = N_LAYER_ONEHOT + N_LOCAL_FEATS + N_GLOBAL_FEATS
VDD_NOMINAL = 1.8


# ---------------------------------------------------------------------------
# Graph loading + short merge
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self, n: int) -> None:
        self.p = np.arange(n)

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


@dataclass
class IBMGraph:
    bench: str
    x_raw: dict                 # raw per-node arrays (post-merge)
    droop: np.ndarray           # [N] volts
    is_grid: np.ndarray         # [N] bool
    edges: dict                 # rel -> (edge_index [2,E], value [E])  undirected, u<v
    n_nodes: int


def load_graph(bench: str, graphs_dir: Path = GRAPHS_DIR) -> IBMGraph:
    d = np.load(graphs_dir / f"{bench}.npz")
    n = int(d["n_nodes"])

    # --- contract shorts (R < R_SHORT) ---
    R_ei, R_val = d["R_edge_index"], d["R_value"].astype(np.float64)
    short = R_val < R_SHORT
    if short.any():
        uf = _UF(n)
        for a, b in R_ei[:, short].T:
            uf.union(int(a), int(b))
        root = np.array([uf.find(i) for i in range(n)])
        # compact remap
        uniq, remap = np.unique(root, return_inverse=True)
        n_new = uniq.size
    else:
        remap = np.arange(n)
        n_new = n

    def merge_edges(ei: np.ndarray, val: np.ndarray, conductance: bool):
        if ei.size == 0:
            return np.zeros((2, 0), np.int64), np.zeros(0, np.float32)
        a, b = remap[ei[0]], remap[ei[1]]
        keep = a != b
        a, b, v = a[keep], b[keep], val[keep].astype(np.float64)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * n_new + hi
        order = np.argsort(key)
        key, lo, hi, v = key[order], lo[order], hi[order], v[order]
        grp = np.concatenate([[True], key[1:] != key[:-1]])
        idx = np.cumsum(grp) - 1
        agg = np.zeros(idx[-1] + 1 if idx.size else 0)
        # resistors combine by conductance sum; C/L just sum (parallel C) —
        # good enough for the handful of merged parallels.
        np.add.at(agg, idx, 1.0 / v if conductance else v)
        vals = (1.0 / agg if conductance else agg).astype(np.float32)
        ei_new = np.stack([lo[grp], hi[grp]])
        return ei_new, vals

    edges = {
        "R": merge_edges(R_ei[:, ~short] if short.any() else R_ei,
                         R_val[~short] if short.any() else R_val, conductance=True),
        "C": merge_edges(d["C_edge_index"], d["C_value"], conductance=False),
        "L": merge_edges(d["L_edge_index"], d["L_value"], conductance=False),
    }

    def pool_max(arr):   # merged node takes max (droop) / or of flags
        out = np.zeros(n_new, dtype=np.float64)
        np.maximum.at(out, remap, arr.astype(np.float64))
        return out

    def pool_sum(arr):
        out = np.zeros(n_new, dtype=np.float64)
        np.add.at(out, remap, arr.astype(np.float64))
        return out

    # node-to-ground conductance (backfilled by scripts/patch_ibmpg_rg.py)
    g_gnd_full = np.zeros(n, dtype=np.float64)
    if "Rg_node" in d.files:
        g_gnd_full[d["Rg_node"]] = 1.0 / d["Rg_value"].astype(np.float64)
    # package inductors to ground (DC shorts to the reference)
    l_gnd_full = np.zeros(n, dtype=bool)
    if "Lg_node" in d.files:
        l_gnd_full[d["Lg_node"]] = True

    x_raw = {
        "layer":    pool_max(d["layer"]).astype(np.int8),
        "is_clamp": pool_max(d["is_clamp"]) > 0,
        "net_vdd":  pool_max(d["net_vdd"]) > 0,
        "load_I":   pool_sum(d["load_I"]).astype(np.float32),
        "cap_gnd":  pool_sum(d["cap_gnd"]).astype(np.float32),
        "g_gnd":    pool_sum(g_gnd_full).astype(np.float32),
        "l_gnd":    pool_max(l_gnd_full) > 0,
        "v_dc":     pool_max(d["v_dc"]).astype(np.float32),
    }
    # load-timing backfill (scripts/patch_ibmpg_timing.py): exact
    # quasi-static timing peak + sparse binned load waveforms
    if "tqs_peak" in d.files:
        x_raw["tqs_peak"] = pool_max(d["tqs_peak"]).astype(np.float32)
        wn = remap[d["wave_node"]]
        uniq, inv = np.unique(wn, return_inverse=True)
        agg = np.zeros((uniq.size, d["wave_bins"].shape[1]))
        np.add.at(agg, inv, d["wave_bins"].astype(np.float64))  # currents add on merge
        x_raw["wave_node"] = uniq.astype(np.int64)
        x_raw["wave_bins"] = agg.astype(np.float32)
    return IBMGraph(
        bench=bench,
        x_raw=x_raw,
        droop=pool_max(d["droop"]).astype(np.float32),
        is_grid=pool_max(d["is_grid"]) > 0,
        edges=edges,
        n_nodes=n_new,
    )


# ---------------------------------------------------------------------------
# Normalization stats (train benchmarks only)
# ---------------------------------------------------------------------------

@dataclass
class FeatureStats:
    mu: dict = field(default_factory=dict)
    sigma: dict = field(default_factory=dict)

    def z(self, name: str, x: np.ndarray) -> np.ndarray:
        return (x - self.mu[name]) / self.sigma[name]


def compute_stats(graphs: list[IBMGraph]) -> FeatureStats:
    s = FeatureStats()

    def add(name: str, vals: np.ndarray) -> None:
        vals = vals[np.isfinite(vals)]
        s.mu[name] = float(vals.mean())
        s.sigma[name] = float(vals.std() + 1e-6)

    add("logR", np.log10(np.concatenate([g.edges["R"][1] for g in graphs]) + LOG_EPS))
    add("logC", np.log10(np.concatenate([g.edges["C"][1] for g in graphs]) + LOG_EPS))
    add("logL", np.log10(np.concatenate([g.edges["L"][1] for g in graphs]) + LOG_EPS))
    li = np.concatenate([g.x_raw["load_I"][g.x_raw["load_I"] > 0] for g in graphs])
    add("logI", np.log10(li + LOG_EPS))
    cg = np.concatenate([g.x_raw["cap_gnd"][g.x_raw["cap_gnd"] > 0] for g in graphs])
    add("logCg", np.log10(cg + LOG_EPS))
    gg = np.concatenate([g.x_raw["g_gnd"][g.x_raw["g_gnd"] > 0] for g in graphs])
    add("logGg", np.log10(gg + LOG_EPS) if gg.size else np.zeros(1))
    sd = np.concatenate([_static_drop(g) for g in graphs])
    add("logSd", np.log10(np.maximum(sd, SDROP_FLOOR)))
    # per-grid global scalars (one value per graph; z over the train set).
    # These carry the grid's overall droop *scale* — total switching
    # current, total decap, size, pad count, median static drop — which a
    # local patch cannot infer but which is trivially computable for any
    # new grid.
    for name in _GLOBAL_NAMES:
        add(name, np.array([_global_scalars(g)[name] for g in graphs]))
    return s


_GLOBAL_NAMES = ("gLogI", "gLogC", "gLogN", "gLogClamp", "gLogSd", "gLogL")


def _global_scalars(g: IBMGraph) -> dict[str, float]:
    grid = g.is_grid
    return {
        "gLogI":     float(np.log10(g.x_raw["load_I"].sum() + LOG_EPS)),
        "gLogC":     float(np.log10(g.x_raw["cap_gnd"].sum() + LOG_EPS)),
        "gLogN":     float(np.log10(max(int(grid.sum()), 1))),
        "gLogClamp": float(np.log10(int(g.x_raw["is_clamp"].sum()) + 1)),
        "gLogSd":    float(np.log10(max(np.median(_static_drop(g)[grid]), SDROP_FLOOR))),
        "gLogL":     float(np.log10(g.edges["L"][0].shape[1] + 1)),
    }


def _static_drop(g: IBMGraph) -> np.ndarray:
    """Per-node DC deviation from its own rail's nominal (volts, >= 0)."""
    v = g.x_raw["v_dc"]
    return np.where(g.x_raw["net_vdd"], VDD_NOMINAL - v, v - 0.0).clip(min=0.0)


def _node_features(g: IBMGraph, stats: FeatureStats) -> np.ndarray:
    n = g.n_nodes
    x = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    lay = np.clip(g.x_raw["layer"], 0, N_LAYER_ONEHOT - 1)
    x[np.arange(n), lay] = 1.0
    c = N_LAYER_ONEHOT
    x[:, c + 0] = g.x_raw["is_clamp"]
    x[:, c + 1] = g.x_raw["net_vdd"]
    has_load = g.x_raw["load_I"] > 0
    x[has_load, c + 2] = 1.0
    x[has_load, c + 3] = stats.z("logI", np.log10(g.x_raw["load_I"][has_load] + LOG_EPS))
    has_cap = g.x_raw["cap_gnd"] > 0
    x[has_cap, c + 4] = 1.0
    # z(log static drop) — the global "how far from the pads" signal.
    x[:, c + 5] = stats.z("logSd", np.log10(np.maximum(_static_drop(g), SDROP_FLOOR)))
    has_rg = g.x_raw["g_gnd"] > 0
    x[has_rg, c + 6] = 1.0
    x[has_rg, c + 7] = stats.z("logGg", np.log10(g.x_raw["g_gnd"][has_rg] + LOG_EPS))
    # per-grid globals, broadcast to every node. Clipped to ±5σ: the
    # stats come from only a handful of training grids, so a held-out
    # grid outside their range must saturate, not explode.
    gs = _global_scalars(g)
    for j, name in enumerate(_GLOBAL_NAMES):
        x[:, c + 8 + j] = np.clip(stats.z(name, np.float64(gs[name])), -5.0, 5.0)
    return x


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

class IBMPGPatches:
    """Iterable of HeteroData patches for one benchmark graph.

    Patches tile the grid nodes: every grid node is a seed in exactly one
    patch, each patch carries a ``halo``-hop context ring, and the loss
    mask selects seed ∩ grid nodes. ``shuffle_seed`` changes the tiling.
    """

    def __init__(
        self,
        g: IBMGraph,
        stats: FeatureStats,
        seed_radius: int = 8,
        halo: int = 7,
        shuffle_seed: int = 0,
        residual: bool = False,
    ) -> None:
        self.g = g
        self.stats = stats
        self.halo = halo
        self.seed_radius = seed_radius
        self.residual = residual

        self.x = _node_features(g, stats)
        self.sd_log = np.log10(
            np.maximum(_static_drop(g), DROOP_FLOOR)).astype(np.float32)
        y_abs = np.log10(np.maximum(g.droop, DROOP_FLOOR)).astype(np.float32)
        # residual mode: learn only the dynamic correction on top of the
        # static IR drop — the baseline's ranking becomes the floor, and
        # predicted log droop = model output + sd_log.
        self.y = (y_abs - self.sd_log) if residual else y_abs

        # union adjacency (R+C+L) in CSR for BFS
        n = g.n_nodes
        ei = np.concatenate([g.edges[r][0] for r in ("R", "C", "L")], axis=1)
        src = np.concatenate([ei[0], ei[1]])
        dst = np.concatenate([ei[1], ei[0]])
        order = np.argsort(src, kind="stable")
        self._adj_dst = dst[order]
        self._adj_ptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(self._adj_ptr, src + 1, 1)
        np.cumsum(self._adj_ptr, out=self._adj_ptr)

        # per-relation incident-edge CSR (edge ids by endpoint) for fast
        # induced-subgraph extraction
        self._inc = {}
        for rel in ("R", "C", "L"):
            rei = g.edges[rel][0]
            e_ids = np.arange(rei.shape[1], dtype=np.int64)
            s = np.concatenate([rei[0], rei[1]])
            e2 = np.concatenate([e_ids, e_ids])
            o = np.argsort(s, kind="stable")
            ptr = np.zeros(n + 1, dtype=np.int64)
            np.add.at(ptr, s + 1, 1)
            np.cumsum(ptr, out=ptr)
            self._inc[rel] = (ptr, e2[o])

        self.patches = self._tile(shuffle_seed)

    # --- BFS helpers ---

    def _ball_levels(self, start: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
        """One BFS ball; returns (nodes, depth) within ``radius`` hops."""
        vis = {start}
        frontier = np.array([start], dtype=np.int64)
        nodes = [frontier]
        depths = [np.zeros(1, dtype=np.int16)]
        for d in range(1, radius + 1):
            nxt = []
            for u in frontier:
                nbr = self._adj_dst[self._adj_ptr[u]:self._adj_ptr[u + 1]]
                for v in nbr:
                    if v not in vis:
                        vis.add(v)
                        nxt.append(v)
            if not nxt:
                break
            frontier = np.array(nxt, dtype=np.int64)
            nodes.append(frontier)
            depths.append(np.full(frontier.size, d, dtype=np.int16))
        return np.concatenate(nodes), np.concatenate(depths)

    def _tile(self, shuffle_seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Cover all grid nodes with (seed_set, patch_nodes) tiles.

        One BFS per patch out to ``seed_radius + halo``; nodes within
        ``seed_radius`` (and not already a seed elsewhere) carry the
        loss, the rest are context.
        """
        g = self.g
        rng = np.random.default_rng(shuffle_seed)
        order = rng.permutation(np.flatnonzero(g.is_grid))
        seed_covered = np.zeros(g.n_nodes, dtype=bool)
        patches = []
        for start in order:
            if seed_covered[start]:
                continue
            nodes, depth = self._ball_levels(int(start), self.seed_radius + self.halo)
            seeds = nodes[(depth <= self.seed_radius) & ~seed_covered[nodes]]
            seeds = seeds[g.is_grid[seeds]]
            if seeds.size == 0:
                continue
            seed_covered[seeds] = True
            patches.append((seeds, np.sort(nodes)))
        return patches

    # --- dataset protocol ---

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i: int):
        import torch
        from torch_geometric.data import HeteroData

        seeds, nodes = self.patches[i]
        n_local = nodes.size
        glob2loc = np.full(self.g.n_nodes, -1, dtype=np.int64)
        glob2loc[nodes] = np.arange(n_local)

        data = HeteroData()
        data["node"].x = torch.from_numpy(self.x[nodes])
        data["node"].y = torch.from_numpy(self.y[nodes])
        if self.residual:
            data["node"].sd = torch.from_numpy(self.sd_log[nodes])
        mask = np.zeros(n_local, dtype=bool)
        mask[glob2loc[seeds]] = True
        mask &= self.g.is_grid[nodes]
        data["node"].loss_mask = torch.from_numpy(mask)

        for rel, stat in (("R", "logR"), ("C", "logC"), ("L", "logL")):
            ei, val = self.g.edges[rel]
            ptr, eids = self._inc[rel]
            cand = np.unique(np.concatenate(
                [eids[ptr[u]:ptr[u + 1]] for u in nodes]
            )) if nodes.size and eids.size else np.zeros(0, np.int64)
            if cand.size:
                a, b = glob2loc[ei[0, cand]], glob2loc[ei[1, cand]]
                keep = (a >= 0) & (b >= 0)
                a, b, v = a[keep], b[keep], val[cand[keep]]
            else:
                a = b = np.zeros(0, np.int64); v = np.zeros(0, np.float32)
            z = self.stats.z(stat, np.log10(v.astype(np.float64) + LOG_EPS)).astype(np.float32)
            # undirected -> both directions
            src = np.concatenate([a, b]); dst = np.concatenate([b, a])
            zz = np.concatenate([z, z])
            et = ("node", rel, "node")
            data[et].edge_index = torch.from_numpy(np.stack([src, dst]))
            data[et].edge_attr = torch.from_numpy(zz[:, None])
        return data
