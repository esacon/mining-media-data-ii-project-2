"""
Data module for Critical Error Detection.

This module handles data loading, preprocessing, and dataset creation
for the critical error detection task.
"""

from .data_loader import CustomDataLoader
from .dataset import CriticalErrorDataset

__all__ = ["CriticalErrorDataset", "CustomDataLoader"]
