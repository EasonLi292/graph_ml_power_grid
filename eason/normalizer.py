"""Analytic input normalizer keyed off ``ParamRanges``.

Stats come from the parameter ranges in [tools/sampler.py], not from a
fit-on-data pass — every ``Param`` provides ``(lo, hi, scale)``, which is
enough to derive ``(mu, sigma)`` for an exact z-score. Log-scale params
get a ``log10`` first.

The R column the GNN sees on strap edges is the *derived* per-segment
``R_top`` / ``R_bot``, not the raw ``Rsheet``. Those are computed
analytically from ``derived_R_ranges`` and the per-``n_top`` pitch, with
the range taken as the union over every ``n_top`` this dataset emits.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from tools.grid_construction import build_regular_pdn
from tools.sampler import (
    ALL_ANCHORS,
    DEFAULT_RANGES,
    ParamRanges,
    derived_R_ranges,
)

from .schema import EDGE_ATTR_DIM_NORMALIZED


class InputNormalizer(nn.Module):
    """Normalize raw edge attributes to ``EDGE_ATTR_DIM_NORMALIZED``-dim.

    Sampled-param stats (``Rsheet_top``, ``wire_width``, ``I_peak``, ...)
    come straight from ``ranges``. Edge attributes the model actually
    sees are the derived per-segment resistances ``R_top`` / ``R_bot``;
    we register stats for those analytically from
    ``derived_R_ranges(ranges, pitch_top, pitch_bot)``.
    """

    def __init__(self, ranges: ParamRanges = DEFAULT_RANGES) -> None:
        super().__init__()
        log_params: set[str] = set()

        def _register(name: str, lo: float, hi: float, scale: str) -> None:
            if scale == "log":
                lo_t, hi_t = math.log10(lo), math.log10(hi)
                log_params.add(name)
            else:
                lo_t, hi_t = lo, hi
            mu = 0.5 * (lo_t + hi_t)
            if lo == hi:
                # Fixed param: sigma must be bounded well away from zero
                # so any float32 jitter at the value doesn't blow up the
                # normalized column. The normalized value is ~ 0 by design.
                sigma = 1.0
            else:
                sigma = (hi_t - lo_t) / math.sqrt(12) + 1e-8
            self.register_buffer(f"mu_{name}", torch.tensor(mu, dtype=torch.float32))
            self.register_buffer(f"sigma_{name}", torch.tensor(sigma, dtype=torch.float32))

        for p in ranges.params:
            _register(p.name, p.lo, p.hi, p.scale)

        # Derived per-segment R_top / R_bot. Pitch_top changes with the
        # (n_top, n_bot) anchor (coarser top mesh → longer segment), so
        # the analytic range needs to span the union over every anchor
        # this dataset emits. Pitch_bot is 1.0 at every current anchor,
        # and the max pitch_top is 3.0 for both die sizes — so these
        # stats are numerically identical to the old single-die-size
        # ones and old checkpoints stay compatible.
        pitch_tops, pitch_bots = [], []
        for nt, nb in ALL_ANCHORS:
            proto_nt = build_regular_pdn(n_top=int(nt), n_bot=int(nb))
            pitch_tops.append(proto_nt.pitch_top)
            pitch_bots.append(proto_nt.pitch_bot)
        agg: dict[str, list] = {}
        for pt, pb in zip(pitch_tops, pitch_bots):
            for name, (lo, hi, scale) in derived_R_ranges(ranges, pt, pb).items():
                if name not in agg:
                    agg[name] = [lo, hi, scale]
                else:
                    agg[name][0] = min(agg[name][0], lo)
                    agg[name][1] = max(agg[name][1], hi)
        for name, (lo, hi, scale) in agg.items():
            _register(name, lo, hi, scale)

        self._log_params = log_params

    def _norm_scalar(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name in self._log_params:
            x = torch.log10(x.clamp_min(1e-15))
        mu = getattr(self, f"mu_{name}")
        sigma = getattr(self, f"sigma_{name}")
        return (x - mu) / sigma

    def normalize_edge_attr(
        self, attr: torch.Tensor, relation: tuple[str, str, str]
    ) -> torch.Tensor:
        """Normalize a 6-dim raw edge attribute to 7-dim.

        Layout in: ``[R, C, I_peak, freq, duty, phase]``.
        Layout out: ``[R_n, C_n, I_n, f_n, d_n, sin 2πφ, cos 2πφ]``.

        Each relation only populates the columns relevant to its
        physical role; the rest stay zero. For load edges, phase becomes
        the usual circular (sin, cos) encoding so φ=0 ≡ φ=1.

        Strap R uses a relation-specific stat: top-strap → ``R_top``,
        bot-strap → ``R_bot``, via → ``R_via``.
        """
        E = attr.shape[0]
        out = torch.zeros((E, EDGE_ATTR_DIM_NORMALIZED), device=attr.device, dtype=attr.dtype)
        rel_name = relation[1]

        if rel_name == "strap":
            stat = "R_top" if relation[0] == "mesh_top" else "R_bot"
            out[:, 0:1] = self._norm_scalar(attr[:, 0:1], stat)
        elif rel_name == "via":
            out[:, 0:1] = self._norm_scalar(attr[:, 0:1], "R_via")
        elif rel_name == "decap":
            out[:, 1:2] = self._norm_scalar(attr[:, 1:2], "C_decap")
        elif rel_name == "load":
            out[:, 2:3] = self._norm_scalar(attr[:, 2:3], "I_peak")
            out[:, 3:4] = self._norm_scalar(attr[:, 3:4], "freq")
            out[:, 4:5] = self._norm_scalar(attr[:, 4:5], "duty")
            ph = attr[:, 5:6]
            out[:, 5:6] = torch.sin(2 * math.pi * ph)
            out[:, 6:7] = torch.cos(2 * math.pi * ph)
        else:
            raise ValueError(f"unknown relation: {relation!r}")

        return out
