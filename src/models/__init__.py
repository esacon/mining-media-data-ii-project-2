"""
Models module for WMT21 Task 3 Critical Error Detection.

This module contains the DistilBERT-based model implementations for
binary classification of critical translation errors.
"""

from .distilbert_classifier import DistilBERTClassifier
from .trainer import Trainer

__all__ = ["DistilBERTClassifier", "Trainer"]
