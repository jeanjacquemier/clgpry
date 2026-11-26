"""
clgpry: Cluster Likelihood GPRy Package

A Python package for Bayesian inference on cluster cosmology using GPRy,
combining CosmoSIS (CAMB, Consistency) with Firecrown likelihood calculations.

Main Components:
    - ClusterLikelihoodPipeline: Core likelihood evaluation pipeline
    - Utilities for parameter management and visualization
"""

__version__ = "0.1.0"
__author__ = "LSST Dark Energy Science Collaboration"
__license__ = "MIT"


def __getattr__(name):
    """Lazy import of ClusterLikelihoodPipeline to avoid requiring all dependencies."""
    if name == "ClusterLikelihoodPipeline":
        from .pipeline import ClusterLikelihoodPipeline
        return ClusterLikelihoodPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ClusterLikelihoodPipeline",
]
