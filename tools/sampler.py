"""Reduced 3-knob parameter sampler for the regular-PDN dataset.

Design choice (paper-grade)
---------------------------
The prior sampler spanned 6 + 3 continuous knobs plus a discrete pad pattern,
which is too much manifold for any realistic dataset size and makes the
downstream generative model impossible to evaluate cleanly. This version cuts
to **three** knobs, holds everything else at physically-realistic constants,
and produces samples whose load/decap *placement* is identical across the
entire dataset — only component values change.

Knobs per sample
~~~~~~~~~~~~~~~~

* Continuous (Latin-Hypercube):
    - ``wire_width``  log-uniform ``[0.2, 1.0]`` (× ``pitch_bot``)
    - ``C_decap``     log-uniform ``[5e-11, 8e-10]`` F per decap site
                         (50–800 pF — a realistic single MIM macro range)
* Discrete (uniform):
    - ``n_top`` ∈ ``{3, 4}`` for the training pool; ``{7}`` held out as OOD.

Fixed across every sample
~~~~~~~~~~~~~~~~~~~~~~~~~

* Topology
    - ``n_bot = 7`` (bottom-mesh density — fixed everywhere)
    - ``pad_pattern = "corner"`` — 4 corner Vdd pads. The corner pattern
      has the property that the four pad-via bot positions are *the same
      four bot indices* for every valid ``n_top``, so the surviving load
      set is constant (12 loads at fixed positions) across every sample.
* Process / electrical (geometric medians of the prior LHS box)
    - ``Rsheet_top ≈ 0.0316 Ω/sq``
    - ``Rsheet_bot ≈ 0.158  Ω/sq``
    - ``R_via      ≈ 0.0632 Ω``
    - ``freq       ≈ 0.894  GHz`` (single clock; ≈ 1 GHz target node)
* Per-load workload (broadcast to every load instance)
    - ``I_peak ≈ 4.47 mA``
    - ``duty   = 0.4``
    - ``phase  = 0.0`` (every load switches in-phase — worst-case analysis)

Why these choices
~~~~~~~~~~~~~~~~~

* ``wire_width`` and ``C_decap`` are the two PDN design knobs reviewers
  expect to see vary; both span a realistic order-of-magnitude range and
  have monotone, recognizable effects on transient droop.
* ``n_top`` is the M_top track-density knob — coarser ``n_top`` means
  wider M_top pitch (longer per-segment R) and fewer top-to-bottom vias.
  The combined constraints — ``(n_bot - 1) % (n_top - 1) == 0`` for via
  alignment AND every bot-net cluster reachable from the via tap-set at
  every supported ``n_top`` — force ``n_top ∈ {3, 4, 7}`` for the
  multi-boundary bot pattern that gives a 14-load target.

Split strategy (generalization-test)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The split is deliberately set up so the model cannot pass test by
memorizing per-topology lookups, and so the held-out topology is
*bracketed* by training topologies (interpolation, not extrapolation):

* Train + Val: ``n_top ∈ {2, 3, 7}`` (uniform discrete LHS over
  (wire_width, C_decap)). Val is held-out only by random split, same
  distribution.
* Test       : ``n_top = 4`` — a supply density bracketed by training
  values on both sides. With three training anchors the model must
  interpolate over a learned supply-density axis rather than memorize
  per-topology lookups.

Encoder compatibility
~~~~~~~~~~~~~~~~~~~~~

``DEFAULT_RANGES`` still contains a ``Param`` entry for every quantity the
GNN's edge attribute pipeline sees, so the analytic ``InputNormalizer`` has
mu/sigma for each column. Fixed quantities use ``lo == hi`` (handled by the
normalizer as a constant column that normalizes to zero).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import qmc


Scale = Literal["linear", "log"]


# ---------------------------------------------------------------------------
# Param / ParamRanges
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    scale: Scale

    @property
    def is_fixed(self) -> bool:
        return self.lo == self.hi

    def median(self) -> float:
        if self.scale == "log":
            return float(np.sqrt(self.lo * self.hi))
        return 0.5 * (self.lo + self.hi)

    def grid(self, n: int) -> np.ndarray:
        if self.scale == "log":
            return np.exp(np.linspace(np.log(self.lo), np.log(self.hi), n))
        return np.linspace(self.lo, self.hi, n)


@dataclass(frozen=True)
class ParamRanges:
    params: tuple[Param, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def d(self) -> int:
        return len(self.params)

    def by_name(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def lhs(self, n: int, seed: int) -> np.ndarray:
        """Return ``[n, d]`` array of parameter values in raw units.

        Fixed params (``lo == hi``) are emitted at their pinned value;
        they consume a column of the LHS but the corresponding ``u`` is
        ignored. This keeps the column layout stable when callers add or
        remove fixed-vs-varying knobs.
        """
        sampler = qmc.LatinHypercube(d=self.d, seed=seed)
        u = sampler.random(n)
        out = np.empty_like(u)
        for j, p in enumerate(self.params):
            if p.is_fixed:
                out[:, j] = p.lo
            elif p.scale == "log":
                lo, hi = np.log(p.lo), np.log(p.hi)
                out[:, j] = np.exp(lo + u[:, j] * (hi - lo))
            else:
                out[:, j] = p.lo + u[:, j] * (p.hi - p.lo)
        return out

    def medians(self) -> dict[str, float]:
        return {p.name: p.median() for p in self.params}


# ---------------------------------------------------------------------------
# The two varying continuous knobs (LHS-sampled per sample)
# ---------------------------------------------------------------------------

GLOBAL_RANGES = ParamRanges(
    params=(
        Param("wire_width", 0.2,   1.0,    "log"),   # × pitch_bot
        Param("C_decap",    5e-11, 8e-10,  "log"),   # F per decap site (50–800 pF)
    )
)


# ---------------------------------------------------------------------------
# Discrete supply-density knob
# ---------------------------------------------------------------------------

# Generalization-test split over (n_top, n_bot) topology anchors.
#
# Valid anchors are constrained by via alignment
# ((n_bot−1) % (n_top−1) == 0) AND full cluster tap coverage — see
# ``BOT_COL_PATTERNS`` in grid_construction. Each supported die size
# admits exactly three densities: {3,4,7} on n_bot=7, {5,7,13} on 13,
# {7,10,19} on 19, {9,13,25} on 25, {11,16,31} on 31, {13,19,37} on 37.
#
# **Why 10 training topologies and not 4.** The previous split trained on
# 4 anchors with 16k samples. Measured over 50 epochs on three different
# architectures, that let the model memorise topology: the val score
# (held-out *samples* of trained topologies) rose +0.675 → +0.969 while
# held-out-*topology* test fell +0.755 → +0.117, a correlation of −0.29
# to −0.49. Diversity, not sample count, is the binding constraint.
#
# The held-out anchors probe three distinct axes:
#
# * ``(4, 7)``  — density interpolation on the small die (bracketed by
#   the trained (3,7) and (7,7)).
# * ``(7, 13)`` — a die size × density combination never trained.
#   These two are UNCHANGED from the 4-anchor split so every prior
#   result on this track stays directly comparable.
# * ``(11, 31)``, ``(31, 31)`` — size extrapolation. Nothing in training
#   exceeds 1250 electrical nodes; these are 1082 and 1922 free nodes on
#   a die 24 % wider than any trained one. This is the axis that matters
#   for the IBM target, and the 4-anchor split could not test it at all.
TRAIN_ANCHORS: tuple[tuple[int, int], ...] = (
    (3, 7), (7, 7),                       # small die, 2 densities
    (5, 13), (13, 13),                    # medium die, 2 densities
    (7, 19), (10, 19), (19, 19),          # large die, 3 densities
    (9, 25), (13, 25), (25, 25),          # larger die, 3 densities
)
TEST_ANCHORS:  tuple[tuple[int, int], ...] = (
    (4, 7), (7, 13),                      # interpolation / cross-die (as before)
    (11, 31), (31, 31),                   # size extrapolation (new)
)
ALL_ANCHORS:   tuple[tuple[int, int], ...] = tuple(
    sorted(set(TRAIN_ANCHORS) | set(TEST_ANCHORS))
)

# Legacy single-die-size aliases (the n_bot=7 subset). Kept for scripts
# that predate the multi-die-size design space (design.py, sweeps on old
# datasets); new code should iterate ALL_ANCHORS.
TRAIN_N_TOP: tuple[int, ...] = tuple(nt for nt, nb in TRAIN_ANCHORS if nb == 7)
TEST_N_TOP:  tuple[int, ...] = tuple(nt for nt, nb in TEST_ANCHORS if nb == 7)
OOD_N_TOP:   tuple[int, ...] = ()
ALL_N_TOP:   tuple[int, ...] = tuple(
    sorted(set(TRAIN_N_TOP) | set(TEST_N_TOP) | set(OOD_N_TOP))
)


def sample_anchor(
    n: int, seed: int, choices: tuple[tuple[int, int], ...] = TRAIN_ANCHORS
) -> np.ndarray:
    """Uniform-discrete draw of (n_top, n_bot) anchors; returns [n, 2] int16."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(choices, dtype=np.int16)
    return pool[rng.integers(low=0, high=len(pool), size=n)]


def sample_n_top(
    n: int, seed: int, choices: tuple[int, ...] = TRAIN_N_TOP
) -> np.ndarray:
    """Uniform-discrete draw of ``n_top`` values; returns int16 array of length ``n``."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(choices, dtype=np.int16)
    return pool[rng.integers(low=0, high=len(pool), size=n)]


def sample_edge_widths(
    n_top: int,
    rng: np.random.Generator,
    n_bot: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent log-uniform per-strap-edge wire widths.

    Each top / bot strap edge draws its own width from the same
    ``wire_width`` band ``[lo, hi]`` used for the global knob, so per-edge
    R stays inside the analytic ``InputNormalizer`` range (no clipping) and
    is *decorrelated from topology* — the whole point of the per-edge
    dataset. Returns ``(ww_top_edges[T], ww_bot_edges[B])`` index-aligned
    with ``build_regular_pdn(n_top).top_edges`` / ``bot_edges``.
    """
    from .grid_construction import build_regular_pdn

    if n_bot is None:
        n_bot = FIXED_N_BOT
    g = build_regular_pdn(n_top=int(n_top), n_bot=int(n_bot))
    ww = GLOBAL_RANGES.by_name("wire_width")
    lo, hi = np.log(ww.lo), np.log(ww.hi)
    n_t = int(g.top_edges.shape[0])
    n_b = int(g.bot_edges.shape[0])
    wt = np.exp(rng.uniform(lo, hi, size=n_t))
    wb = np.exp(rng.uniform(lo, hi, size=n_b))
    return wt.astype(np.float64), wb.astype(np.float64)


# ---------------------------------------------------------------------------
# Fixed-value constants (every sample uses these)
# ---------------------------------------------------------------------------

# Topology
FIXED_N_BOT: int = 7

# Process / electrical (geometric medians of the prior LHS box)
FIXED_RSHEET_TOP: float = float(np.sqrt(0.01 * 0.1))   # ≈ 0.0316 Ω/sq
FIXED_RSHEET_BOT: float = float(np.sqrt(0.05 * 0.5))   # ≈ 0.158  Ω/sq
FIXED_R_VIA:      float = float(np.sqrt(0.02 * 0.2))   # ≈ 0.0632 Ω
FIXED_FREQ:       float = float(np.sqrt(2e8 * 4e9))    # ≈ 0.894 GHz

# Per-load workload (broadcast to every load instance). I_peak chosen so
# that with 14 loads at duty=0.4 the worst-case DC droop sits in the
# 1–20 mV range over the wire_width / C_decap design box — a regime
# where the knobs visibly change the droop without saturating the rail.
FIXED_I_PEAK: float = 1e-4                              # 100 μA per load
FIXED_DUTY:   float = 0.4
FIXED_PHASE:  float = 0.0


FIXED_CONSTANTS: dict[str, float | int] = {
    "n_bot":       FIXED_N_BOT,
    "Rsheet_top":  FIXED_RSHEET_TOP,
    "Rsheet_bot":  FIXED_RSHEET_BOT,
    "R_via":       FIXED_R_VIA,
    "freq":        FIXED_FREQ,
    "I_peak":      FIXED_I_PEAK,
    "duty":        FIXED_DUTY,
    "phase":       FIXED_PHASE,
}


# ---------------------------------------------------------------------------
# Encoder-compatibility: ParamRanges over every column the GNN sees.
#
# Varying knobs (``wire_width``, ``C_decap``) get their real lo/hi. Fixed
# quantities use ``lo == hi``; ``InputNormalizer`` detects that and emits a
# stable constant-zero normalized column.
# ---------------------------------------------------------------------------

DEFAULT_RANGES = ParamRanges(
    params=(
        Param("wire_width", 0.2,              1.0,              "log"),
        Param("C_decap",    5e-11,            8e-10,            "log"),
        Param("Rsheet_top", FIXED_RSHEET_TOP, FIXED_RSHEET_TOP, "log"),
        Param("Rsheet_bot", FIXED_RSHEET_BOT, FIXED_RSHEET_BOT, "log"),
        Param("R_via",      FIXED_R_VIA,      FIXED_R_VIA,      "log"),
        Param("freq",       FIXED_FREQ,       FIXED_FREQ,       "log"),
        Param("I_peak",     FIXED_I_PEAK,     FIXED_I_PEAK,     "log"),
        Param("duty",       FIXED_DUTY,       FIXED_DUTY,       "linear"),
        Param("phase",      FIXED_PHASE,      FIXED_PHASE,      "linear"),
    )
)


# ---------------------------------------------------------------------------
# Derived ranges (for the normalizer's R_top / R_bot stats)
# ---------------------------------------------------------------------------

def derived_R_ranges(
    ranges: ParamRanges, pitch_top: float, pitch_bot: float
) -> dict[str, tuple[float, float, Scale]]:
    """Analytic ranges of the derived per-segment ``R_top`` / ``R_bot``.

    ``R_seg = Rsheet × (pitch / wire_width)``. With ``Rsheet`` fixed (lo == hi)
    and ``wire_width`` log-uniform, ``R_seg`` is log-uniform over the band
    ``[Rsheet · pitch / ww.hi, Rsheet · pitch / ww.lo]``.
    """
    rs_top = ranges.by_name("Rsheet_top")
    rs_bot = ranges.by_name("Rsheet_bot")
    ww = ranges.by_name("wire_width")
    return {
        "R_top": (rs_top.lo * pitch_top / ww.hi, rs_top.hi * pitch_top / ww.lo, "log"),
        "R_bot": (rs_bot.lo * pitch_bot / ww.hi, rs_bot.hi * pitch_bot / ww.lo, "log"),
    }


# ---------------------------------------------------------------------------
# Axis sweeps (for latent-structure / sensitivity plots)
# ---------------------------------------------------------------------------

def axis_sweep(
    axis: str,
    n_points: int,
    ranges: ParamRanges = GLOBAL_RANGES,
) -> tuple[np.ndarray, dict[str, float]]:
    """1-D sweep along ``axis`` with the other continuous knobs at median.

    Returns ``(axis_values, other_continuous_medians)``. Callers combine
    these with the fixed constants and a chosen ``n_top`` to assemble
    per-sample parameter dicts. ``axis`` must be a name in ``ranges``;
    sweeping over the discrete ``n_top`` is done by direct enumeration
    upstream, not via this helper.
    """
    if axis not in ranges.names:
        raise KeyError(f"unknown sweep axis: {axis!r} (must be one of {ranges.names})")
    p = ranges.by_name(axis)
    return p.grid(n_points), ranges.medians()
