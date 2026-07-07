"""Eason's model package — heterogeneous admittance-aware GNN for the PDN.

Self-contained snapshot of the encoder stack as of the train-{2,3,4} /
admittance-conv run. Same public API as ``tools.encoder`` so it can be
swapped in without touching the dataset loader or training loop:

    from tools.eason import EncoderConfig, PDNDroopRegressor

Submodule layout (small files, one job each):

* [schema.py](schema.py)      — node / edge type tuples, normalized
                                edge-attr dimension.
* [normalizer.py](normalizer.py) — ``InputNormalizer``: analytic
                                z-score / log-z-score keyed off
                                ``ParamRanges`` and the per-``n_top``
                                derived R range.
* [convs.py](convs.py)        — ``EdgeAwareConv`` (generic baseline),
                                ``AdmittanceConv`` (physics-shaped:
                                ``gate(edge_attr) ⊙ delta(x_j − x_i)``
                                for admittance edges, generic form for
                                load edges), and the per-relation
                                ``_make_conv`` factory.
* [encoder.py](encoder.py)    — ``EncoderConfig``, ``PDNEncoder``
                                (stacked ``HeteroConv`` + LayerNorm
                                + residual), and ``PDNDroopRegressor``
                                (encoder + per-``mesh_bot`` scalar head).
"""
from .convs import AdmittanceConv, EdgeAwareConv, EdgeConvGated, MonotoneGate
from .encoder import EncoderConfig, PDNDroopRegressor, PDNEncoder
from .normalizer import InputNormalizer
from .schema import EDGE_ATTR_DIM_NORMALIZED, EDGE_TYPES, NODE_TYPES

__all__ = [
    "NODE_TYPES",
    "EDGE_TYPES",
    "EDGE_ATTR_DIM_NORMALIZED",
    "InputNormalizer",
    "EdgeAwareConv",
    "AdmittanceConv",
    "EncoderConfig",
    "PDNEncoder",
    "PDNDroopRegressor",
]
