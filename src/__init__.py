"""
Critical Error Detection package.

A machine learning package for detecting critical errors in machine translation.
"""

__version__ = "1.0.0"

from src.data import CriticalErrorDataset, CustomDataLoader
from src.models import DistilBERTClassifier, Trainer
from src.utils import get_logger, load_config, setup_logger

__all__ = [
    "DistilBERTClassifier",
    "Trainer",
    "CriticalErrorDataset",
    "CustomDataLoader",
    "get_logger",
    "setup_logger",
    "load_config",
]
