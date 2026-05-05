"""Latin-Hypercube parameter sampling for the regular-PDN dataset.

Adding an extrapolation distribution later is just a matter of constructing a
new ``ParamRanges`` with shifted bounds and re-using the same sampler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import qmc


Scale = Literal["linear", "log"]


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    scale: Scale


@dataclass(frozen=True)
class ParamRanges:
    params: tuple[Param, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def d(self) -> int:
        return len(self.params)

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


DEFAULT_RANGES = ParamRanges(
    params=(
        Param("R_top", 0.05, 0.5, "log"),
        Param("R_bot", 0.2, 5.0, "log"),
        Param("R_via", 0.02, 0.2, "log"),
        Param("C_decap", 1e-11, 1e-9, "log"),
        Param("I_peak", 1e-3, 2e-2, "log"),
        Param("freq", 2e8, 4e9, "log"),
        Param("duty", 0.2, 0.6, "linear"),
    )
)


EXTRAPOLATION_RANGES = ParamRanges(
    params=(
        Param("R_top", 0.5, 1.0, "log"),
        Param("R_bot", 5.0, 8.0, "log"),
        Param("R_via", 0.2, 0.4, "log"),
        Param("C_decap", 1e-12, 1e-11, "log"),
        Param("I_peak", 2e-2, 3e-2, "log"),
        Param("freq", 4e9, 6e9, "log"),
        Param("duty", 0.6, 0.8, "linear"),
    )
)
