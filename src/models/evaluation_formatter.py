import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import get_logger


class EvaluationFormatter:
    """
    Formatter for predictions in the official WMT evaluation format.

    Creates files compatible with the official evaluation script format:
    Line 1: disk_footprint_bytes
    Line 2: model_parameters
    Lines 3+: <LP> <METHOD_NAME> <SEGMENT_NUMBER> <SEGMENT_SCORE>
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the evaluation formatter."""
        self.logger = logger or get_logger(__name__)

    def create_evaluation_file(
        self,
        predictions: List[int],
        probabilities: List[float],
        test_ids: List[str],
        language_pair: str,
        method_name: str,
        disk_footprint_bytes: int,
        model_parameters: int,
        output_path: str,
    ) -> str:
        """
        Create an evaluation file in WMT competition format.

        Args:
            predictions: Binary predictions (0 or 1)
            probabilities: Prediction probabilities for class 1
            test_ids: Test sample IDs
            language_pair: Language pair code (e.g., 'en-de')
            method_name: Name of the method/system
            disk_footprint_bytes: Model disk footprint in bytes
            model_parameters: Number of model parameters
            output_path: Path to save evaluation file

        Returns:
            Path to created evaluation file
        """
        self.logger.info(f"Creating evaluation file for {language_pair}")

        # Convert predictions to submission format
        submission_lines = []

        # Line 1: Disk footprint
        submission_lines.append(str(disk_footprint_bytes))

        # Line 2: Model parameters
        submission_lines.append(str(model_parameters))

        # Lines 3+: Predictions in format:
        # <LP> <METHOD_NAME> <SEGMENT_NUMBER> <SEGMENT_SCORE>
        for i, (test_id, pred, prob) in enumerate(
            zip(test_ids, predictions, probabilities)
        ):
            # Convert binary prediction to evaluation format
            segment_score = self._binary_to_evaluation_label(pred)

            line = f"{language_pair.lower()}\t{method_name}\t{test_id}\t{segment_score}"
            submission_lines.append(line)

        # Write evaluation file
        with open(output_path, "w", encoding="utf-8") as f:
            for line in submission_lines:
                f.write(line + "\n")

        self.logger.info(f"Evaluation file saved to: {output_path}")
        self.logger.info(f"Total predictions: {len(predictions)}")
        self.logger.info(
            f"Critical errors detected: {sum(predictions)} "
            f"({sum(predictions)/len(predictions)*100:.1f}%)"
        )

        return output_path

    def create_gold_labels_file(
        self,
        true_labels: List[int],
        test_ids: List[str],
        language_pair: str,
        method_name: str,
        output_path: str,
    ) -> str:
        """
        Create a gold labels file for evaluation.

        Args:
            true_labels: True binary labels
            test_ids: Test sample IDs
            language_pair: Language pair code
            method_name: Method name (usually 'goldlabels')
            output_path: Path to save gold labels file

        Returns:
            Path to created gold labels file
        """
        self.logger.info(f"Creating gold labels file for {language_pair}")

        submission_lines = []

        # Gold labels format doesn't include disk footprint and parameters
        # Lines start directly with predictions
        for test_id, label in zip(test_ids, true_labels):
            segment_score = self._binary_to_evaluation_label(label)
            line = f"{language_pair.lower()}\t{method_name}\t{test_id}\t{segment_score}"
            submission_lines.append(line)

        # Write gold labels file
        with open(output_path, "w", encoding="utf-8") as f:
            for line in submission_lines:
                f.write(line + "\n")

        self.logger.info(f"Gold labels file saved to: {output_path}")

        return output_path

    def _binary_to_evaluation_label(self, binary_pred: int) -> str:
        """
        Convert binary prediction to evaluation label format.

        Args:
            binary_pred: 0 (no error) or 1 (critical error)

        Returns:
            Evaluation label ('NOT' or 'ERR')
        """
        return "ERR" if binary_pred == 1 else "NOT"

    def create_metadata_file(
        self,
        predictions: List[int],
        language_pair: str,
        method_name: str,
        model_info: Dict,
        output_path: str,
    ) -> str:
        """
        Create metadata file with evaluation information.

        Args:
            predictions: Binary predictions
            language_pair: Language pair code
            method_name: Method name
            model_info: Dictionary with model information
            output_path: Path to save metadata file

        Returns:
            Path to created metadata file
        """
        import json

        def json_serializer(obj):
            """Handle numpy types for JSON serialization."""
            import numpy as np

            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )

        metadata = {
            "task": "Critical_Error_Detection",
            "language_pair": language_pair,
            "method_name": method_name,
            "model_info": model_info,
            "submission_stats": {
                "total_predictions": len(predictions),
                "critical_errors_detected": sum(predictions),
                "critical_error_rate": (
                    sum(predictions) / len(predictions) if predictions else 0.0
                ),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=json_serializer)

        self.logger.info(f"Metadata saved to: {output_path}")

        return output_path

    def validate_evaluation_format(self, evaluation_path: str) -> Dict[str, bool]:
        """
        Validate that evaluation file follows correct format.

        Args:
            evaluation_path: Path to evaluation file

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "file_exists": False,
            "correct_header": False,
            "correct_line_format": False,
            "valid_labels": False,
        }

        try:
            # Check if file exists
            if not Path(evaluation_path).exists():
                return validation_results
            validation_results["file_exists"] = True

            with open(evaluation_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) < 3:
                return validation_results

            # Check header lines (disk footprint and model parameters)
            try:
                int(lines[0].strip())  # Disk footprint
                int(lines[1].strip())  # Model parameters
                validation_results["correct_header"] = True
            except ValueError:
                pass

            # Check prediction lines format
            valid_format = True
            valid_labels = True

            for line in lines[2:]:
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    valid_format = False
                    break

                # Check label format
                label = parts[3]
                if label not in ["ERR", "NOT", "err", "not", "BAD", "GOOD", "OK", "ok"]:
                    valid_labels = False
                    break

            validation_results["correct_line_format"] = valid_format
            validation_results["valid_labels"] = valid_labels

        except Exception as e:
            self.logger.error(f"Error validating evaluation file: {e}")

        return validation_results

    def convert_probabilities_to_binary(
        self, probabilities: List[float], threshold: float = 0.5
    ) -> List[int]:
        """
        Convert probabilities to binary predictions.

        Args:
            probabilities: Prediction probabilities for class 1
            threshold: Decision threshold

        Returns:
            Binary predictions
        """
        return [1 if prob >= threshold else 0 for prob in probabilities]
