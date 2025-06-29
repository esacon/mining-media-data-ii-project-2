"""
Data module for WMT21 Task 3 Critical Error Detection.

This module handles data loading, preprocessing, and dataset creation
for the critical error detection task.
"""

from .data_loader import WMT21DataLoader
from .dataset import CriticalErrorDataset

__all__ = ["CriticalErrorDataset", "WMT21DataLoader"]
