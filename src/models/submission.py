"""
WMT21 Task 3 Submission Formatter.

This module handles creating submissions in the exact format expected by
the WMT21 evaluation script.
"""

import pandas as pd
from typing import List, Dict, Optional
import logging
from pathlib import Path

from ..utils.logging_utils import get_logger


class WMT21SubmissionFormatter:
    """
    Formatter for WMT21 Task 3 submission files.
    
    Based on the evaluation script format:
    Line 1: disk_footprint_bytes
    Line 2: model_parameters
    Lines 3+: <LP> <METHOD_NAME> <SEGMENT_NUMBER> <SEGMENT_SCORE>
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the submission formatter."""
        self.logger = logger or get_logger(__name__)

    def create_submission(
        self,
        predictions: List[int],
        probabilities: List[float],
        test_ids: List[str],
        language_pair: str,
        method_name: str,
        disk_footprint_bytes: int,
        model_parameters: int,
        output_path: str
    ) -> str:
        """
        Create a WMT21-compatible submission file.
        
        Args:
            predictions: Binary predictions (0 or 1)
            probabilities: Prediction probabilities for class 1
            test_ids: Test sample IDs
            language_pair: Language pair code (e.g., 'en-de')
            method_name: Name of the method/system
            disk_footprint_bytes: Model disk footprint in bytes
            model_parameters: Number of model parameters
            output_path: Path to save submission file
            
        Returns:
            Path to created submission file
        """
        self.logger.info(f"Creating WMT21 submission for {language_pair}")
        
        # Convert predictions to submission format
        submission_lines = []
        
        # Line 1: Disk footprint
        submission_lines.append(str(disk_footprint_bytes))
        
        # Line 2: Model parameters
        submission_lines.append(str(model_parameters))
        
        # Lines 3+: Predictions in format:
        # <LP> <METHOD_NAME> <SEGMENT_NUMBER> <SEGMENT_SCORE>
        for i, (test_id, pred, prob) in enumerate(zip(test_ids, predictions, probabilities)):
            # Convert binary prediction to submission format
            segment_score = self._binary_to_submission_label(pred)
            
            line = f"{language_pair.lower()}\t{method_name}\t{test_id}\t{segment_score}"
            submission_lines.append(line)
        
        # Write submission file
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in submission_lines:
                f.write(line + '\n')
        
        self.logger.info(f"Submission saved to: {output_path}")
        self.logger.info(f"Total predictions: {len(predictions)}")
        self.logger.info(f"Critical errors detected: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        
        return output_path

    def create_gold_labels_submission(
        self,
        true_labels: List[int],
        test_ids: List[str],
        language_pair: str,
        method_name: str,
        output_path: str
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
            segment_score = self._binary_to_submission_label(label)
            line = f"{language_pair.lower()}\t{method_name}\t{test_id}\t{segment_score}"
            submission_lines.append(line)
        
        # Write gold labels file
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in submission_lines:
                f.write(line + '\n')
        
        self.logger.info(f"Gold labels saved to: {output_path}")
        
        return output_path

    def _binary_to_submission_label(self, binary_pred: int) -> str:
        """
        Convert binary prediction to submission label format.
        
        Args:
            binary_pred: 0 (no error) or 1 (critical error)
            
        Returns:
            Submission label ('NOT' or 'ERR')
        """
        return 'ERR' if binary_pred == 1 else 'NOT'

    def create_submission_metadata(
        self,
        predictions: List[int],
        language_pair: str,
        method_name: str,
        model_info: Dict,
        output_path: str
    ) -> str:
        """
        Create metadata file with submission information.
        
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
        
        metadata = {
            'task': 'WMT21_Task3_Critical_Error_Detection',
            'language_pair': language_pair,
            'method_name': method_name,
            'model_info': model_info,
            'submission_stats': {
                'total_predictions': len(predictions),
                'critical_errors_detected': sum(predictions),
                'critical_error_rate': sum(predictions) / len(predictions) if predictions else 0.0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to: {output_path}")
        
        return output_path

    def validate_submission_format(self, submission_path: str) -> Dict[str, bool]:
        """
        Validate that submission file follows correct format.
        
        Args:
            submission_path: Path to submission file
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'file_exists': False,
            'correct_header': False,
            'correct_line_format': False,
            'valid_labels': False
        }
        
        try:
            # Check if file exists
            if not Path(submission_path).exists():
                return validation_results
            validation_results['file_exists'] = True
            
            with open(submission_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 3:
                return validation_results
            
            # Check header lines (disk footprint and model parameters)
            try:
                int(lines[0].strip())  # Disk footprint
                int(lines[1].strip())  # Model parameters
                validation_results['correct_header'] = True
            except ValueError:
                pass
            
            # Check prediction lines format
            valid_format = True
            valid_labels = True
            
            for line in lines[2:]:
                parts = line.strip().split('\t')
                if len(parts) != 4:
                    valid_format = False
                    break
                
                # Check label format
                label = parts[3]
                if label not in ['ERR', 'NOT', 'err', 'not', 'BAD', 'GOOD', 'OK', 'ok']:
                    valid_labels = False
                    break
            
            validation_results['correct_line_format'] = valid_format
            validation_results['valid_labels'] = valid_labels
            
        except Exception as e:
            self.logger.error(f"Error validating submission: {e}")
        
        return validation_results

    def convert_probabilities_to_binary(
        self,
        probabilities: List[float],
        threshold: float = 0.5
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