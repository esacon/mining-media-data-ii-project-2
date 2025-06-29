"""
Critical Error Detection package.

A machine learning package for detecting critical errors in machine translation.
"""

__version__ = "0.1.0"

from .data import CriticalErrorDataset, WMT21DataLoader
from .models import DistilBERTClassifier, Trainer
from .utils import get_logger, load_config, setup_logger

__all__ = [
    "DistilBERTClassifier",
    "Trainer",
    "CriticalErrorDataset",
    "WMT21DataLoader",
    "get_logger",
    "setup_logger",
    "load_config",
]
