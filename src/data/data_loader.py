"""
WMT21 Task 3 Data Loader.

This module handles loading and preprocessing of the actual WMT21 Task 3 
Critical Error Detection data format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import ast
import logging
from pathlib import Path

from ..utils.logging_utils import get_logger


class WMT21DataLoader:
    """
    Data loader for WMT21 Task 3 Critical Error Detection format.
    
    Handles the actual format found in the catastrophic_errors data:
    ID  source-sentence  target-sentence  3-scores  aggregated-score
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the data loader."""
        self.logger = logger or get_logger(__name__)

    def load_train_dev_data(self, file_path: str) -> pd.DataFrame:
        """
        Load training or development data with labels.
        
        Args:
            file_path: Path to TSV file with format:
                       ID source target [score1,score2,score3] aggregated_label
                       
        Returns:
            DataFrame with columns: id, source, target, individual_scores, label
        """
        self.logger.info(f"Loading train/dev data from: {file_path}")
        
        # Read TSV file
        df = pd.read_csv(file_path, sep='\t', header=None, 
                        names=['id', 'source', 'target', 'scores', 'label'])
        
        # Parse individual scores (format: [0, 1, 0])
        df['individual_scores'] = df['scores'].apply(self._parse_scores)
        
        # Convert aggregated labels to binary (ERR=1, NOT=0)
        df['binary_label'] = df['label'].apply(self._convert_label_to_binary)
        
        # Add language pair from filename
        df['language_pair'] = self._extract_language_pair(file_path)
        
        self.logger.info(f"Loaded {len(df)} samples")
        self.logger.info(f"Label distribution: {df['binary_label'].value_counts().to_dict()}")
        
        return df

    def load_test_data(self, file_path: str) -> pd.DataFrame:
        """
        Load blind test data without labels.
        
        Args:
            file_path: Path to TSV file with format:
                       ID source target
                       
        Returns:
            DataFrame with columns: id, source, target, language_pair
        """
        self.logger.info(f"Loading test data from: {file_path}")
        
        # Read TSV file
        df = pd.read_csv(file_path, sep='\t', header=None,
                        names=['id', 'source', 'target'])
        
        # Add language pair from filename
        df['language_pair'] = self._extract_language_pair(file_path)
        
        self.logger.info(f"Loaded {len(df)} test samples")
        
        return df

    def _parse_scores(self, scores_str: str) -> List[int]:
        """
        Parse individual annotator scores from string format.
        
        Args:
            scores_str: String like "[0, 1, 0]" or "[1, 1, 1]"
            
        Returns:
            List of integer scores
        """
        try:
            # Use ast.literal_eval to safely parse the list
            scores = ast.literal_eval(scores_str)
            return scores
        except (ValueError, SyntaxError) as e:
            self.logger.warning(f"Failed to parse scores: {scores_str}, error: {e}")
            return [0, 0, 0]  # Default to no errors

    def _convert_label_to_binary(self, label: str) -> int:
        """
        Convert aggregated label to binary format.
        
        Args:
            label: 'ERR' (critical error) or 'NOT' (no critical error)
            
        Returns:
            1 for ERR, 0 for NOT
        """
        label_map = {
            'ERR': 1,  # Critical error present
            'NOT': 0,  # No critical error
            'err': 1,  # Handle lowercase
            'not': 0,
            'BAD': 1,  # Alternative format
            'GOOD': 1, # Alternative format
            'OK': 0,
            'ok': 0,
            'good': 0,
            'bad': 1
        }
        
        return label_map.get(label, 0)

    def _extract_language_pair(self, file_path: str) -> str:
        """
        Extract language pair from filename.
        
        Args:
            file_path: Path like "ende_majority_train.tsv"
            
        Returns:
            Language pair like "en-de"
        """
        filename = Path(file_path).stem
        
        # Extract language code from filename patterns
        if 'ende' in filename:
            return 'en-de'
        elif 'enja' in filename:
            return 'en-ja'
        elif 'enzh' in filename:
            return 'en-zh'
        elif 'encs' in filename:
            return 'en-cs'
        else:
            return 'unknown'

    def get_critical_error_categories(self) -> Dict[str, str]:
        """
        Get the 5 critical error categories as defined by WMT21.
        
        Returns:
            Dictionary mapping category codes to descriptions
        """
        return {
            'TOX': 'Toxicity deviation (hate, violence, profanity)',
            'SAF': 'Safety risks deviation',
            'NAM': 'Named entities deviation', 
            'SEN': 'Sentiment polarity or negation deviation',
            'NUM': 'Units/time/date/numbers deviation'
        }

    def analyze_annotator_agreement(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze inter-annotator agreement on the dataset.
        
        Args:
            df: DataFrame with individual_scores column
            
        Returns:
            Dictionary with agreement statistics
        """
        agreements = []
        
        for scores in df['individual_scores']:
            if len(scores) == 3:
                # Count how many annotators agree
                unique_scores = set(scores)
                if len(unique_scores) == 1:
                    agreements.append(1.0)  # All agree
                elif len(unique_scores) == 2:
                    # Check majority
                    score_counts = {s: scores.count(s) for s in unique_scores}
                    max_count = max(score_counts.values())
                    agreements.append(max_count / 3)  # Majority agreement
                else:
                    agreements.append(0.33)  # All disagree
        
        return {
            'mean_agreement': np.mean(agreements),
            'full_agreement_ratio': sum(1 for a in agreements if a == 1.0) / len(agreements),
            'majority_agreement_ratio': sum(1 for a in agreements if a >= 0.67) / len(agreements)
        }

    def filter_by_language_pair(self, df: pd.DataFrame, language_pair: str) -> pd.DataFrame:
        """
        Filter dataset by specific language pair.
        
        Args:
            df: Input DataFrame
            language_pair: Language pair code (e.g., 'en-de')
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df[df['language_pair'] == language_pair].copy()
        self.logger.info(f"Filtered to {len(filtered_df)} samples for {language_pair}")
        return filtered_df

    def prepare_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for training by ensuring proper format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame ready for training with required columns
        """
        # Ensure we have the required columns
        required_columns = ['source', 'target', 'binary_label', 'language_pair']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create a clean training DataFrame
        training_df = df[['id', 'source', 'target', 'binary_label', 'language_pair']].copy()
        training_df = training_df.rename(columns={'binary_label': 'label'})
        
        # Remove any rows with missing data
        training_df = training_df.dropna()
        
        self.logger.info(f"Prepared {len(training_df)} samples for training")
        
        return training_df 