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

from .pipeline import ClusterLikelihoodPipeline

__all__ = [
    "ClusterLikelihoodPipeline",
]
