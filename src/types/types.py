"""TypedDict definitions for the project."""

from typing import Dict, List, Literal, Optional, TypedDict


class ModelConfig(TypedDict):
    """Represents the 'model' section of the configuration."""

    name: str
    max_seq_length: int
    num_labels: int


class TrainingConfig(TypedDict):
    """Represents the 'training' section of the configuration."""

    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    warmup_steps: int
    device: Literal["cpu", "cuda", "auto"]
    mixed_precision: bool
    gradient_accumulation_steps: int


class DataConfig(TypedDict):
    """Represents the 'data' section of the configuration."""

    train_data_dir: str
    gold_labels_dir: str
    train_ratio: float
    val_ratio: float
    random_seed: int
    num_workers: int
    pin_memory: bool


class PathsConfig(TypedDict):
    """Represents the 'paths' section of the configuration."""

    checkpoints_dir: str
    logs_dir: str
    results_dir: str


class LoggingConfig(TypedDict):
    """Represents the 'logging' section of the configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_interval: int
    save_interval: int


class QuickTestConfig(TypedDict):
    """Represents the 'quick_test' section of the configuration."""

    sample_size: int
    batch_size: int
    num_epochs: int
    learning_rate: float


class AppConfigRequired(TypedDict):
    """Required fields in the application configuration."""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathsConfig
    logging: LoggingConfig


class AppConfig(AppConfigRequired, total=False):
    """Represents the complete application configuration."""

    quick_test: QuickTestConfig


class DatasetStats(TypedDict):
    """Statistics for a dataset split."""

    samples: int
    critical_errors: int
    error_rate: float
    no_errors: int


class TestDatasetInfo(TypedDict):
    """Information about test dataset."""

    samples: int
    note: str


class OverallStats(TypedDict):
    """Overall statistics across train and dev sets."""

    total_samples: int
    total_critical_errors: int
    overall_error_rate: float


class FilesFound(TypedDict, total=False):
    """Files found for a language pair."""

    train: str
    dev: str
    test: str


class LanguagePairAnalysis(TypedDict, total=False):
    """Analysis results for a language pair."""

    language_pair: str
    files_found: FilesFound
    train: DatasetStats
    dev: DatasetStats
    test: TestDatasetInfo
    overall: OverallStats


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

    loss: float
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


DataAnalysisResults = Dict[str, LanguagePairAnalysis]
