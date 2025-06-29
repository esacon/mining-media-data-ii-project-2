"""
Data analysis pipeline for Critical Error Detection.
"""

from pathlib import Path
from typing import Any, Dict, List

from src.data.data_loader import WMT21DataLoader

from .base import BasePipeline


class DataAnalysisPipeline(BasePipeline):
    """Pipeline for analyzing data."""

    def __init__(self, config_path: str, device: str = "auto"):
        super().__init__(config_path, device)

    def run(self, data_dir: str, language_pairs: List[str]) -> Dict[str, Any]:
        """
        Run the data analysis pipeline.

        Args:
            data_dir: Directory containing data files
            language_pairs: List of language pairs to analyze

        Returns:
            Analysis results dictionary
        """
        self.logger.info("Starting data analysis pipeline")

        loader = WMT21DataLoader()
        results = {}

        for language_pair in language_pairs:
            self.logger.info(f"Analyzing {language_pair}")

            analysis = self._analyze_language_pair(loader, data_dir, language_pair)
            results[language_pair] = analysis

        self.logger.info("Data analysis completed successfully")

        return results

    def _analyze_language_pair(
        self, loader: WMT21DataLoader, data_dir: str, language_pair: str
    ) -> Dict[str, Any]:
        """Analyze data for a specific language pair."""

        # Map language pair to file prefix
        prefix_map = {"en-de": "ende", "en-ja": "enja", "en-zh": "enzh", "en-cs": "encs"}

        prefix = prefix_map.get(language_pair, "ende")
        data_path = Path(data_dir)

        # Load data files
        train_file = data_path / f"{prefix}_majority_train.tsv"
        dev_file = data_path / f"{prefix}_majority_dev.tsv"
        test_file = data_path / f"{prefix}_majority_test_blind.tsv"

        analysis = {"language_pair": language_pair, "files_found": {}}

        # Analyze training data
        if train_file.exists():
            self.logger.info(f"Analyzing training data: {train_file}")
            train_df = loader.load_train_dev_data(str(train_file))
            analysis["train"] = {
                "samples": len(train_df),
                "critical_errors": int(train_df["binary_label"].sum()),
                "error_rate": float(train_df["binary_label"].mean()),
                "no_errors": int((train_df["binary_label"] == 0).sum()),
            }
            analysis["files_found"]["train"] = str(train_file)

        # Analyze development data
        if dev_file.exists():
            self.logger.info(f"Analyzing development data: {dev_file}")
            dev_df = loader.load_train_dev_data(str(dev_file))
            analysis["dev"] = {
                "samples": len(dev_df),
                "critical_errors": int(dev_df["binary_label"].sum()),
                "error_rate": float(dev_df["binary_label"].mean()),
                "no_errors": int((dev_df["binary_label"] == 0).sum()),
            }
            analysis["files_found"]["dev"] = str(dev_file)

        # Analyze test data (if available)
        if test_file.exists():
            self.logger.info(f"Analyzing test data: {test_file}")
            test_df = loader.load_test_data(str(test_file))
            analysis["test"] = {
                "samples": len(test_df),
                "note": "Labels not available for test data",
            }
            analysis["files_found"]["test"] = str(test_file)

        # Calculate overall statistics
        if "train" in analysis and "dev" in analysis:
            total_samples = analysis["train"]["samples"] + analysis["dev"]["samples"]
            total_errors = analysis["train"]["critical_errors"] + analysis["dev"]["critical_errors"]
            analysis["overall"] = {
                "total_samples": total_samples,
                "total_critical_errors": total_errors,
                "overall_error_rate": total_errors / total_samples if total_samples > 0 else 0.0,
            }

        return analysis
