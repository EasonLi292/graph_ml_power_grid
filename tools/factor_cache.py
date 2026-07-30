"""Impedance-factor caches: versioned, keyed on everything, disposable.

The separation this enforces
---------------------------
Two artifacts with completely different costs and lifetimes keep getting
conflated:

    SIMULATOR LABELS   expensive (hours of transient solves), and a
                       property of the CIRCUIT alone. Independent of factor
                       rank, probe seed, frequency grid, normalization mode
                       and model architecture. Regenerating them because a
                       hyper-parameter changed is pure waste.

    FACTOR CACHES      cheap by comparison, a DERIVED artifact, and a
                       function of the circuit *and* every factor
                       hyper-parameter. Deleting one costs only recompute.

So the label file must never carry a factor parameter, and the factor cache
must carry all of them in its key. A cache key that omits a parameter is
worse than no cache: it silently loads factors built under different
settings, and the resulting run looks fine.

The previous key was ``tag_m{m}_q{n_power}_f{n_freq}_c{n_ch}``. It omitted
the probe seed, the projection mode and the actual frequency VALUES — two
grids with three frequencies each hash identically — so it could serve
mismatched factors without any error. Everything is in the key now, and the
key is prefixed with a format version so a layout change invalidates rather
than misreads.

    from tools.factor_cache import FactorSpec
    spec = FactorSpec(omegas=om, m=16, n_power=2, seed=0, proj="hermitian")
    path = spec.path(cache_dir, tag="train")
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

# Bump when the on-disk layout or the meaning of any field changes. Old
# caches then miss rather than deserialize into the new meaning.
CACHE_FORMAT_VERSION = 2

# Fields of a label/circuit file that would break the separation if present.
FORBIDDEN_IN_LABELS = ("m_factor", "n_power", "probe_seed", "omegas",
                       "freq_norm", "freq_norm_mode", "proj", "score",
                       "factors", "p", "s", "fdc")


@dataclass(frozen=True)
class FactorSpec:
    """Every input that changes the factors. All of it lands in the key."""
    omegas: tuple
    m: int = 16
    n_power: int = 2
    seed: int = 0                # probe draw; factors are basis-dependent
    proj: str = "hermitian"
    with_fdc: bool = True

    def __init__(self, omegas, m=16, n_power=2, seed=0, proj="hermitian",
                 with_fdc=True):
        om = tuple(round(float(w), 6) for w in
                   (omegas.tolist() if torch.is_tensor(omegas) else omegas))
        object.__setattr__(self, "omegas", om)
        object.__setattr__(self, "m", int(m))
        object.__setattr__(self, "n_power", int(n_power))
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "proj", str(proj))
        object.__setattr__(self, "with_fdc", bool(with_fdc))

    def key(self) -> str:
        """Readable prefix plus a hash of the FULL spec, frequencies included."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        h = hashlib.sha1(blob).hexdigest()[:10]
        return (f"v{CACHE_FORMAT_VERSION}_m{self.m}_q{self.n_power}"
                f"_f{len(self.omegas)}_s{self.seed}_{self.proj[:4]}"
                f"{'_fdc' if self.with_fdc else ''}_{h}")

    def path(self, cache_dir, tag: str) -> Path | None:
        if cache_dir is None:
            return None
        return Path(cache_dir) / f"{tag}_{self.key()}.pt"

    def manifest(self) -> dict:
        """What the cache is, written beside it so a stale file is legible."""
        return {"format_version": CACHE_FORMAT_VERSION, "spec": asdict(self),
                "disposable": True,
                "note": "derived from the circuits in the label file; safe to "
                        "delete, costs only recompute. Never a source of truth."}


def save(obj, path: Path):
    """Write a cache plus its manifest."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    path.with_suffix(".json").write_text(json.dumps({"file": path.name}, indent=2))


def purge(cache_dir, keep: FactorSpec | None = None) -> int:
    """Delete every cache except optionally one spec. Returns files removed.

    Exists so that changing rank or normalization is a one-line disposal
    rather than a reason to hesitate.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    keep_key = keep.key() if keep is not None else None
    n = 0
    for p in cache_dir.glob("*.pt"):
        if keep_key and keep_key in p.name:
            continue
        p.unlink()
        p.with_suffix(".json").unlink(missing_ok=True)
        n += 1
    return n


def assert_labels_are_independent(h5_path) -> list:
    """Fail loudly if a label file has picked up a factor hyper-parameter.

    Cheap enough to run in the generator, and it is the only thing standing
    between "labels are reusable" and finding out otherwise after a
    multi-hour regeneration.
    """
    import h5py
    bad = []
    with h5py.File(h5_path, "r") as f:
        names = set(f.attrs.keys())

        def walk(name, obj):
            names.add(name.rsplit("/", 1)[-1])
        f.visititems(walk)
    for k in FORBIDDEN_IN_LABELS:
        if k in names:
            bad.append(k)
    if bad:
        raise ValueError(
            f"{h5_path} carries factor-dependent field(s) {bad}: the "
            f"simulator labels would then be tied to a factor setting and "
            f"could not be reused when it changes")
    return sorted(names)
