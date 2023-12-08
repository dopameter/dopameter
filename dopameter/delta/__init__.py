# -*- coding: utf-8 -*-

"""
pydelta library
---------------

Stylometrics in Python

This code is derived from the framework 'PyDelta': https://github.com/cophi-wue/pydelta
It is modified for the usage in DOPA METER.


"""
import warnings

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError: # < Python 3.8
    import importlib_metadata

__title__ = 'delta'
try:
    __version__ = importlib_metadata.version(__name__)
except Exception as e:
    __version__ = None
    warnings.warn(f'Could not determine version: {e}', ImportWarning, source=e)
__author__ = 'Fotis Jannidis, Thorsten Vitt'

from warnings import warn
from dopameter.delta.deltas import registry as functions, normalization, Normalization, \
        DeltaFunction, PDistDeltaFunction, MetricDeltaFunction, \
        CompositeDeltaFunction
from dopameter.delta.cluster import Clustering, FlatClustering

from dopameter.delta.graphics import Dendrogram
from dopameter.delta.util import compare_pairwise, Metadata, TableDocumentDescriber

__all__ = [ "functions",
            "Normalization",
            "normalization",
            "DeltaFunction",
            "PDistDeltaFunction",
            "MetricDeltaFunction",
            "CompositeDeltaFunction",
            "Clustering",
            "FlatClustering",
            "Dendrogram",
            "compare_pairwise",
            "Metadata",
            "TableDocumentDescriber"
            ]

#__all__ = [ "Corpus", "FeatureGenerator", "LETTERS_PATTERN", "WORD_PATTERN",
#           "functions", "Normalization", "normalization",
#           "DeltaFunction", "PDistDeltaFunction",
#           "MetricDeltaFunction", "CompositeDeltaFunction",
#           "Clustering", "FlatClustering",
#           "get_rfe_features", "Dendrogram",
#           "compare_pairwise", "Metadata", "TableDocumentDescriber" ]

try:
        from dopameter.delta.cluster import KMedoidsClustering
        __all__.append("KMedoidsClustering")
except (ImportError, NameError):
        warn("KMedoidsClustering not available")
