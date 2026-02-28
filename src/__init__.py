"""Heart Disease ML Classification - ML Pipeline Package

This package provides utilities for building, training, evaluating, and deploying
machine learning models for heart disease classification.
"""

from .data_preprocessing import DataPreprocessor
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

__version__ = "1.0.0"
__author__ = "Kanak Acharya"

__all__ = [
    'DataPreprocessor',
    'ModelTrainer',
    'ModelEvaluator'
]
