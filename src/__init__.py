"""
Netflix Movie Recommendation Engine
====================================
A comprehensive machine learning system for movie recommendations.

Modules:
    - config: Configuration and constants
    - data_preprocessing: Data loading and preprocessing
    - sparse_matrix_handler: Sparse matrix operations
    - feature_engineering: Feature creation and engineering
    - similarity: Similarity computation
    - models: Model training and evaluation
    - visualization: Plotting and analytics
"""

from .config import Config
from .data_preprocessing import DataPreprocessor
from .sparse_matrix_handler import SparseMatrixHandler
from .feature_engineering import FeatureEngineer
from .similarity import SimilarityComputer
from .models import ModelTrainer
from .visualization import Visualizer

__version__ = '1.0.0'
__author__ = 'BHUVANESH SSN'

__all__ = [
    'Config',
    'DataPreprocessor',
    'SparseMatrixHandler',
    'FeatureEngineer',
    'SimilarityComputer',
    'ModelTrainer',
    'Visualizer'
]
