from typing import Dict, List, Optional, TypedDict


class TrainingResults(TypedDict, total=False):
    """Results from model training."""

    best_mcc: float
    best_model_path: str
    final_model_path: str
    timestamp: str
    language_pair: Optional[str]
    model_prefix: str


class EvaluationResults(TypedDict):
    """Results from model evaluation."""

    accuracy: float
    mcc: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    predictions: List[int]
    probabilities: List[float]


class PredictionResults(TypedDict):
    """Results from model predictions."""

    predictions: List[int]
    probabilities: List[float]
    test_ids: List[str]
    data_path: str
    model_path: str
    language_pair: Optional[str]


class ExperimentResults(TypedDict):
    """Results from a full experiment."""

    training: TrainingResults
    evaluation: EvaluationResults
    model_path: str
    data_path: str
    language_pair: Optional[str]


class LanguagePairAnalysis(TypedDict, total=False):
    """Analysis results for a language pair."""

    language_pair: str
    files_found: Dict[str, str]
    train: Optional[Dict[str, int]]
    dev: Optional[Dict[str, int]]
    test: Optional[Dict[str, int]]
    overall: Optional[Dict[str, int]]


DataAnalysisResults = Dict[str, LanguagePairAnalysis]
