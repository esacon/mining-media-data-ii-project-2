"""
Pipeline package for Critical Error Detection.

This package contains modular pipeline components for training, evaluation,
prediction, and data analysis.
"""

from .data_analysis import DataAnalysisPipeline
from .evaluation import EvaluationPipeline
from .main import MainPipeline
from .prediction import PredictionPipeline
from .training import TrainingPipeline

__all__ = [
    "MainPipeline",
    "TrainingPipeline",
    "EvaluationPipeline",
    "PredictionPipeline",
    "DataAnalysisPipeline",
]
