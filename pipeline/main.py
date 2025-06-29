"""
Main pipeline orchestrator for Critical Error Detection.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logger

from .data_analysis import DataAnalysisPipeline
from .evaluation import EvaluationPipeline
from .prediction import PredictionPipeline
from .training import TrainingPipeline


class MainPipeline:
    """
    Main pipeline orchestrator that centralizes all operations.

    This class provides a unified interface for training, evaluation,
    prediction, and data analysis operations.
    """

    def __init__(self, config_path: str, device: str = "auto"):
        """
        Initialize the main pipeline.

        Args:
            config_path: Path to configuration file
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        self.config_path = config_path
        self.device = device

        # Load configuration and setup logging
        self.config = load_config(config_path)
        setup_logger(
            name="MainPipeline",
            level_str=self.config["logging"]["level"],
            log_file=f"{self.config['paths']['logs_dir']}/pipeline.log",
            log_to_console=True,
        )
        self.logger = get_logger(__name__)

        # Initialize pipeline components (lazy loading)
        self._training_pipeline = None
        self._evaluation_pipeline = None
        self._prediction_pipeline = None
        self._data_analysis_pipeline = None

        self.logger.info("Main pipeline initialized")

    @property
    def training_pipeline(self) -> TrainingPipeline:
        """Get training pipeline (lazy loading)."""
        if self._training_pipeline is None:
            self._training_pipeline = TrainingPipeline(self.config_path, self.device)
        return self._training_pipeline

    @property
    def evaluation_pipeline(self) -> EvaluationPipeline:
        """Get evaluation pipeline (lazy loading)."""
        if self._evaluation_pipeline is None:
            self._evaluation_pipeline = EvaluationPipeline(self.config_path, self.device)
        return self._evaluation_pipeline

    @property
    def prediction_pipeline(self) -> PredictionPipeline:
        """Get prediction pipeline (lazy loading)."""
        if self._prediction_pipeline is None:
            self._prediction_pipeline = PredictionPipeline(self.config_path, self.device)
        return self._prediction_pipeline

    @property
    def data_analysis_pipeline(self) -> DataAnalysisPipeline:
        """Get data analysis pipeline (lazy loading)."""
        if self._data_analysis_pipeline is None:
            self._data_analysis_pipeline = DataAnalysisPipeline(self.config_path, self.device)
        return self._data_analysis_pipeline

    def train(
        self, data_path: str, language_pair: Optional[str] = None, save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Train a model.

        Args:
            data_path: Path to training data
            language_pair: Optional language pair filter
            save_results: Whether to save training results

        Returns:
            Training results dictionary
        """
        self.logger.info("Starting training operation")

        results = self.training_pipeline.run(data_path, language_pair)

        if save_results:
            output_path = Path(self.config["paths"]["results_dir"]) / "training_results.json"
            self._save_results(results, output_path)

        return results

    def evaluate(
        self,
        model_path: str,
        data_path: str,
        language_pair: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.

        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to evaluation data
            language_pair: Optional language pair filter
            save_results: Whether to save evaluation results

        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Starting evaluation operation")

        results = self.evaluation_pipeline.run(model_path, data_path, language_pair)

        if save_results:
            output_path = Path(self.config["paths"]["results_dir"]) / "evaluation_results.json"
            self._save_results(results, output_path)

        return results

    def predict(
        self,
        model_path: str,
        data_path: str,
        language_pair: Optional[str] = None,
        save_results: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Make predictions on new data.

        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to data for prediction
            language_pair: Optional language pair filter
            save_results: Whether to save prediction results

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.logger.info("Starting prediction operation")

        predictions, probabilities = self.prediction_pipeline.run(
            model_path, data_path, language_pair
        )

        if save_results:
            results = {
                "predictions": predictions,
                "probabilities": probabilities,
                "data_path": data_path,
                "model_path": model_path,
                "language_pair": language_pair,
            }
            output_path = Path(self.config["paths"]["results_dir"]) / "prediction_results.json"
            self._save_results(results, output_path)

        return predictions, probabilities

    def analyze_data(
        self,
        data_dir: Optional[str] = None,
        language_pairs: Optional[List[str]] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze data for given language pairs.

        Args:
            data_dir: Directory containing data files (defaults to config value)
            language_pairs: List of language pairs to analyze (defaults to all)
            save_results: Whether to save analysis results

        Returns:
            Analysis results dictionary
        """
        self.logger.info("Starting data analysis operation")

        # Use defaults from config if not specified
        if data_dir is None:
            data_dir = self.config["data"]["train_data_dir"]

        if language_pairs is None:
            language_pairs = ["en-de", "en-ja", "en-zh", "en-cs"]

        results = self.data_analysis_pipeline.run(data_dir, language_pairs)

        if save_results:
            output_path = Path(self.config["paths"]["results_dir"]) / "data_analysis_results.json"
            self._save_results(results, output_path)

        return results

    def run_full_experiment(
        self, data_path: str, language_pair: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a full experiment: train, then evaluate on development data.

        Args:
            data_path: Path to training data
            language_pair: Optional language pair filter

        Returns:
            Combined results dictionary
        """
        self.logger.info("Starting full experiment")

        # Step 1: Train the model
        self.logger.info("Step 1: Training model")
        training_results = self.train(data_path, language_pair, save_results=False)

        # Step 2: Find the best model checkpoint
        checkpoints_dir = Path(self.config["paths"]["checkpoints_dir"])
        best_model_path = checkpoints_dir / "best_model.pth"

        if not best_model_path.exists():
            # Fallback to any checkpoint
            checkpoints = list(checkpoints_dir.glob("*.pth"))
            if checkpoints:
                best_model_path = checkpoints[0]
            else:
                raise FileNotFoundError("No model checkpoints found after training")

        # Step 3: Evaluate the model
        self.logger.info("Step 2: Evaluating model")
        # For evaluation, we'll use the same data but in evaluation mode
        evaluation_results = self.evaluate(
            str(best_model_path), data_path, language_pair, save_results=False
        )

        # Combine results
        experiment_results = {
            "training": training_results,
            "evaluation": evaluation_results,
            "model_path": str(best_model_path),
            "data_path": data_path,
            "language_pair": language_pair,
        }

        # Save combined results
        output_path = Path(self.config["paths"]["results_dir"]) / "experiment_results.json"
        self._save_results(experiment_results, output_path)

        self.logger.info("Full experiment completed successfully")

        return experiment_results

    def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to: {output_path}")

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config.copy()

    def update_config(self, **kwargs):
        """
        Update configuration values.

        Args:
            **kwargs: Configuration updates (nested keys supported with dots)
        """
        for key, value in kwargs.items():
            # Support nested keys like 'training.batch_size'
            keys = key.split(".")
            config_ref = self.config

            for k in keys[:-1]:
                if k not in config_ref:
                    config_ref[k] = {}
                config_ref = config_ref[k]

            config_ref[keys[-1]] = value
            self.logger.info(f"Updated config: {key} = {value}")

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline configuration and capabilities."""
        return {
            "config_path": self.config_path,
            "device": self.device,
            "model_name": self.config["model"]["name"],
            "data_directory": self.config["data"]["train_data_dir"],
            "output_directories": {
                "checkpoints": self.config["paths"]["checkpoints_dir"],
                "logs": self.config["paths"]["logs_dir"],
                "results": self.config["paths"]["results_dir"],
            },
            "training_config": self.config["training"],
            "available_operations": [
                "train",
                "evaluate",
                "predict",
                "analyze_data",
                "run_full_experiment",
            ],
        }
