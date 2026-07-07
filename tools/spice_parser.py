"""Minimal SPICE parser for the IBM transient power-grid benchmarks.

Handles the element subset the ``ibmpg*t.spice`` netlists actually use:

* ``r<name> a b value``      resistor (Ω)
* ``c<name> a b value``      capacitor (F)
* ``l<name> a b value``      inductor (H)  — package parasitics
* ``v<name> a b value``      independent DC voltage source. value 0 ⇒ a
                              short (merged away by the solver); nonzero ⇒
                              a clamp (the 1.8 V VDD pads).
* ``i<name> a b dc pulse(I1,I2,TD,TR,TF,PW,PER)``  current source (load),
                              time-varying via the SPICE PULSE waveform.
* ``.tran dt tend`` / ``.print tran v(node) ...`` / ``.width`` / ``*`` / ``.end``

Node ``0`` is ground. Node names are kept as strings (e.g. ``n2_18380_8346``,
``_X_n2_...``); the solver maps them to indices after merging shorts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GROUND = "0"
_PULSE_RE = re.compile(r"pulse\s*\(([^)]*)\)", re.IGNORECASE)
_VNODE_RE = re.compile(r"v\(([^)]+)\)", re.IGNORECASE)


@dataclass
class Pulse:
    """SPICE PULSE(I1 I2 TD TR TF PW PER) — vectorized evaluation."""
    i1: float
    i2: float
    td: float
    tr: float
    tf: float
    pw: float
    per: float

    def at(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = np.full(t.shape, self.i1, dtype=float)
        active = t >= self.td
        local = np.where(self.per > 0, np.mod(t - self.td, self.per), t - self.td)
        tr, tf, pw = self.tr, self.tf, self.pw
        # rising edge
        if tr > 0:
            m = active & (local < tr)
            out[m] = self.i1 + (self.i2 - self.i1) * (local[m] / tr)
        # high plateau
        m = active & (local >= tr) & (local < tr + pw)
        out[m] = self.i2
        # falling edge
        if tf > 0:
            m = active & (local >= tr + pw) & (local < tr + pw + tf)
            out[m] = self.i2 + (self.i1 - self.i2) * ((local[m] - tr - pw) / tf)
        # low plateau (remainder of period) already i1 from the fill
        m = active & (local >= tr + pw + tf)
        out[m] = self.i1
        return out


@dataclass
class Circuit:
    resistors: list[tuple[str, str, float]] = field(default_factory=list)
    capacitors: list[tuple[str, str, float]] = field(default_factory=list)
    inductors: list[tuple[str, str, float]] = field(default_factory=list)
    vsources: list[tuple[str, str, float]] = field(default_factory=list)
    isources: list[tuple[str, str, float, Pulse | None]] = field(default_factory=list)
    tran_dt: float | None = None
    tran_tend: float | None = None
    probes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        nodes = set()
        for lst in (self.resistors, self.capacitors, self.inductors, self.vsources):
            for a, b, *_ in lst:
                nodes.add(a); nodes.add(b)
        for a, b, *_ in self.isources:
            nodes.add(a); nodes.add(b)
        nodes.discard(GROUND)
        return (f"R={len(self.resistors)} C={len(self.capacitors)} "
                f"L={len(self.inductors)} V={len(self.vsources)} "
                f"I={len(self.isources)} nodes={len(nodes)} "
                f"tran(dt={self.tran_dt}, tend={self.tran_tend}) probes={len(self.probes)}")


def _parse_pulse(rest: str) -> Pulse | None:
    m = _PULSE_RE.search(rest)
    if not m:
        return None
    nums = [float(x) for x in m.group(1).replace(",", " ").split()]
    nums = (nums + [0.0] * 7)[:7]
    return Pulse(*nums)


def parse_netlist(path: str | Path) -> Circuit:
    circ = Circuit()
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            low = line.lower()
            if low.startswith(".tran"):
                toks = line.split()
                circ.tran_dt = float(toks[1])
                circ.tran_tend = float(toks[2])
                continue
            if low.startswith(".print"):
                circ.probes.extend(_VNODE_RE.findall(line))
                continue
            if line.startswith("."):  # .width, .end, .option, ...
                continue

            kind = low[0]
            toks = line.split()
            if kind in "rcl":
                _, a, b, val = toks[0], toks[1], toks[2], float(toks[3])
                if kind == "r":
                    circ.resistors.append((a, b, val))
                elif kind == "c":
                    circ.capacitors.append((a, b, val))
                else:
                    circ.inductors.append((a, b, val))
            elif kind == "v":
                circ.vsources.append((toks[1], toks[2], float(toks[3])))
            elif kind == "i":
                a, b = toks[1], toks[2]
                dc = float(toks[3]) if len(toks) > 3 and "(" not in toks[3] else 0.0
                pulse = _parse_pulse(line)
                circ.isources.append((a, b, dc, pulse))
            # silently ignore any other element kinds
    return circ


if __name__ == "__main__":
    import sys
    c = parse_netlist(sys.argv[1])
    print(c.summary())
