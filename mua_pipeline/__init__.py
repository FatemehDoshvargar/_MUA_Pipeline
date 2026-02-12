# mua_pipeline/__init__.py
"""
MUA Pipeline: A Configurable Pipeline for Mass Univariate Aggregation Methods
"""

from .core import FeatureVectorizer, MUA
from .preprocessing import preprocess
from .visualization import plot_results
from .version import __version__

__all__ = [
    'FeatureVectorizer',
    'MUA',
    'preprocess',
    'plot_results',
    '__version__'
]
