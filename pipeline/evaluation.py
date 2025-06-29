"""
Evaluation pipeline for Critical Error Detection.
"""

from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from src.data.dataset import CriticalErrorDataset
from src.models.trainer import Trainer

from .base import BasePipeline


class EvaluationPipeline(BasePipeline):
    """Pipeline for evaluating trained models."""

    def __init__(self, config_path: str, device: str = "auto"):
        super().__init__(config_path, device)
        self.trainer = None

    def run(
        self, model_path: str, data_path: str, language_pair: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the evaluation pipeline.

        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to evaluation data
            language_pair: Optional language pair filter

        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Starting evaluation pipeline")

        # Load model from checkpoint
        self._load_model_from_checkpoint(model_path)

        # Create test data loader
        test_loader = self._create_test_data_loader(data_path, language_pair)

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )

        # Evaluate
        results = self.trainer.evaluate(test_loader)

        # Make predictions
        predictions, probabilities = self.trainer.predict(test_loader)

        # Add predictions to results
        results["predictions"] = predictions
        results["probabilities"] = probabilities

        self.logger.info("Evaluation completed successfully")

        return results

    def _create_test_data_loader(
        self, data_path: str, language_pair: Optional[str] = None
    ) -> DataLoader:
        """Create test data loader."""
        self._load_tokenizer()

        test_dataset = CriticalErrorDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config["model"]["max_seq_length"],
            language_pair=language_pair,
        )

        # Get dataloader parameters
        dl_params = self._get_dataloader_params()

        test_loader = DataLoader(test_dataset, shuffle=False, **dl_params)

        self.logger.info(f"Test samples: {len(test_dataset)}")

        return test_loader
