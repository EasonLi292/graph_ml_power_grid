"""Latin-Hypercube parameter sampling for the regular-PDN dataset.

Two parameter buckets:

* **Global continuous** (LHS): ``Rsheet_top``, ``Rsheet_bot``,
  ``wire_width``, ``R_via``, ``C_decap``, ``freq``. One value per sample.
* **Per-load continuous** (iid per load): ``I_peak``, ``duty``, ``phase``.
  One value per load instance per sample.

Topology is one discrete knob (``pad_pattern``) drawn uniformly. The
training set sees three of the four patterns; the fourth is reserved for
an out-of-distribution generalization split.

Helpers here also expose:

* ``sample_per_load`` — draws iid per-load (I_peak, duty, phase) for a
  variable number of loads (depends on the chosen pad pattern).
* ``axis_sweep`` — 1-D grids that hold every parameter at its median and
  walk one axis across its full range, used to validate latent-space
  structure rather than to train.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import qmc

from .grid_construction import PAD_PATTERNS


Scale = Literal["linear", "log"]


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    scale: Scale

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
        """Return ``[n, d]`` array of parameter values in raw units."""
        sampler = qmc.LatinHypercube(d=self.d, seed=seed)
        u = sampler.random(n)  # uniform in [0, 1)
        out = np.empty_like(u)
        for j, p in enumerate(self.params):
            if p.scale == "log":
                lo, hi = np.log(p.lo), np.log(p.hi)
                out[:, j] = np.exp(lo + u[:, j] * (hi - lo))
            else:
                out[:, j] = p.lo + u[:, j] * (p.hi - p.lo)
        return out

    def to_dict_list(self, samples: np.ndarray) -> list[dict[str, float]]:
        return [
            {p.name: float(samples[i, j]) for j, p in enumerate(self.params)}
            for i in range(samples.shape[0])
        ]

    def medians(self) -> dict[str, float]:
        return {p.name: p.median() for p in self.params}


# Global continuous design parameters.
GLOBAL_RANGES = ParamRanges(
    params=(
        Param("Rsheet_top", 0.01, 0.1, "log"),     # Ω / square
        Param("Rsheet_bot", 0.05, 0.5, "log"),     # Ω / square
        Param("wire_width", 0.2, 1.0, "log"),      # in units of pitch_bot
        Param("R_via", 0.02, 0.2, "log"),          # Ω per via-stack
        Param("C_decap", 1e-11, 1e-9, "log"),      # F
        Param("freq", 2e8, 4e9, "log"),            # Hz (single clock per sample)
    )
)


# Per-load continuous parameters (iid per load instance within a sample).
PER_LOAD_RANGES = ParamRanges(
    params=(
        Param("I_peak", 1e-3, 2e-2, "log"),
        Param("duty", 0.2, 0.6, "linear"),
        Param("phase", 0.0, 1.0, "linear"),
    )
)


# Back-compat alias for callers that just want "every continuous knob, ranges":
DEFAULT_RANGES = ParamRanges(params=tuple(list(GLOBAL_RANGES.params) + list(PER_LOAD_RANGES.params)))


EXTRAPOLATION_RANGES = ParamRanges(
    params=(
        Param("Rsheet_top", 0.1, 0.3, "log"),
        Param("Rsheet_bot", 0.5, 1.5, "log"),
        Param("wire_width", 0.1, 0.2, "log"),
        Param("R_via", 0.2, 0.4, "log"),
        Param("C_decap", 1e-12, 1e-11, "log"),
        Param("freq", 4e9, 6e9, "log"),
        Param("I_peak", 2e-2, 3e-2, "log"),
        Param("duty", 0.6, 0.8, "linear"),
        Param("phase", 0.0, 1.0, "linear"),
    )
)


# Pattern split: train on the first 3, hold out the 4th for OOD evaluation.
TRAIN_PAD_PATTERNS: tuple[str, ...] = ("corner", "checker", "edge_strip")
OOD_PAD_PATTERNS: tuple[str, ...] = ("distributed",)


def derived_R_ranges(
    ranges: ParamRanges, pitch_top: float, pitch_bot: float
) -> dict[str, tuple[float, float, Scale]]:
    """Analytic ranges of the derived per-segment R_top / R_bot.

    ``R_seg = Rsheet × (pitch / wire_width)``; ``Rsheet`` and ``wire_width``
    are sampled log-uniformly, so ``R_seg`` is also log-uniform.
    """
    rs_top = ranges.by_name("Rsheet_top")
    rs_bot = ranges.by_name("Rsheet_bot")
    ww = ranges.by_name("wire_width")
    return {
        "R_top": (rs_top.lo * pitch_top / ww.hi, rs_top.hi * pitch_top / ww.lo, "log"),
        "R_bot": (rs_bot.lo * pitch_bot / ww.hi, rs_bot.hi * pitch_bot / ww.lo, "log"),
    }


def sample_pad_patterns(n: int, seed: int, choices: tuple[str, ...] = TRAIN_PAD_PATTERNS) -> np.ndarray:
    """Uniform i.i.d. draw of pad-pattern indices into ``PAD_PATTERNS``.

    Indices are with respect to the full ``PAD_PATTERNS`` tuple so they are
    consistent across splits regardless of which patterns each split uses.
    """
    rng = np.random.default_rng(seed)
    pool = np.array([PAD_PATTERNS.index(c) for c in choices], dtype=np.int8)
    pick = rng.integers(low=0, high=len(pool), size=n)
    return pool[pick]


def sample_per_load(
    n_loads_per_sample: list[int],
    seed: int,
    ranges: ParamRanges = PER_LOAD_RANGES,
    max_n_loads: int | None = None,
) -> np.ndarray:
    """iid per-load draws for each sample.

    Returns ``[N, max_n_loads, len(ranges.params)]`` padded with zeros for
    samples that have fewer than ``max_n_loads`` loads.
    """
    N = len(n_loads_per_sample)
    if max_n_loads is None:
        max_n_loads = max(n_loads_per_sample) if n_loads_per_sample else 0
    rng = np.random.default_rng(seed)
    out = np.zeros((N, max_n_loads, ranges.d), dtype=np.float32)
    for i, k in enumerate(n_loads_per_sample):
        u = rng.random((k, ranges.d))
        for j, p in enumerate(ranges.params):
            if p.scale == "log":
                lo, hi = np.log(p.lo), np.log(p.hi)
                out[i, :k, j] = np.exp(lo + u[:, j] * (hi - lo))
            else:
                out[i, :k, j] = p.lo + u[:, j] * (p.hi - p.lo)
    return out


def axis_sweep(
    axis: str,
    n_points: int,
    global_ranges: ParamRanges = GLOBAL_RANGES,
    per_load_ranges: ParamRanges = PER_LOAD_RANGES,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    """1-D sweep along ``axis``; all other knobs held at their medians.

    Returns ``(axis_values, fixed_global_medians, fixed_per_load_medians)``.
    The caller is responsible for assembling these into per-sample dicts.
    """
    if axis in global_ranges.names:
        p = global_ranges.by_name(axis)
    elif axis in per_load_ranges.names:
        p = per_load_ranges.by_name(axis)
    else:
        raise KeyError(f"unknown sweep axis: {axis!r}")
    return p.grid(n_points), global_ranges.medians(), per_load_ranges.medians()
