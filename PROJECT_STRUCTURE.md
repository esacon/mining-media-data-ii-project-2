# Project Structure

This document provides a detailed overview of the project architecture, components, and data flow.

## 🏗️ Directory Structure

```
project-2/
├── config.yaml                     # Main configuration file
├── Makefile                        # Build and workflow automation
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Code formatting and tool configuration
├── setup.py                       # Package installation setup
├── Pipfile / Pipfile.lock         # Pipenv dependency management
│
├── data/                           # Data directory
│   ├── catastrophic_errors/        # Training/dev/test datasets
│   │   ├── encs_majority_train.tsv
│   │   ├── encs_majority_dev.tsv
│   │   ├── encs_majority_test_blind.tsv
│   │   ├── ende_majority_*.tsv
│   │   ├── enja_majority_*.tsv
│   │   └── enzh_majority_*.tsv
│   ├── catastrophic_errors_goldlabels/  # Test gold labels
│   │   ├── encs_majority_test_goldlabels.tar.gz
│   │   ├── ende_majority_test_goldlabels.tar.gz
│   │   ├── enja_majority_test_goldlabels.tar.gz
│   │   └── enzh_majority_test_goldlabels.tar.gz
│   └── evaluation/                 # Official evaluation scripts
│       ├── README.md
│       └── sent_evaluate_CED.py
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── runner.py                  # Main pipeline orchestrator
│   ├── data/                      # Data handling components
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Data loading and preprocessing
│   │   └── dataset.py             # PyTorch dataset implementation
│   ├── models/                    # Model components
│   │   ├── __init__.py
│   │   ├── distilbert_classifier.py  # DistilBERT-based classifier
│   │   ├── trainer.py             # Training logic and loops
│   │   └── evaluation_formatter.py   # WMT evaluation format output
│   ├── types/                     # Type definitions
│   │   ├── __init__.py
│   │   └── types.py               # Custom types and data structures
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── logging_utils.py       # Logging utilities
│       └── language_utils.py      # Language pair utilities
│
├── scripts/                       # Command-line scripts
│   ├── run_pipeline.py           # Main CLI entry point
│   ├── setup_dev.sh              # Development environment setup
│   └── test_all_languages.sh     # Multi-language training script
│
├── logs/                         # Training and execution logs
│   └── (generated during runtime)
│
├── results/                      # Output directory
│   ├── checkpoints/              # Model checkpoints
│   │   ├── best_model_*.pt       # Best model during training
│   │   ├── final_model_*.pt      # Final trained model
│   │   └── training_history_*.json  # Training metrics
│   ├── prediction_results.json   # Prediction outputs
│   ├── evaluation_*.tsv          # WMT evaluation format
│   └── evaluation_metadata_*.json # Model metadata
│
└── references/                   # Documentation and references
    └── MMD_II_SoSe25_Assignment2.pdf
```

## 🧩 Component Architecture

### Core Components

#### 1. **Main Pipeline (`src/runner.py`)**
- Central orchestrator for all operations
- Handles training, evaluation, prediction workflows
- Manages configuration and logging
- Coordinates between data, model, and utility components

#### 2. **Data Layer (`src/data/`)**
- **`data_loader.py`**: Loads and preprocesses TSV data files
- **`dataset.py`**: PyTorch Dataset implementation for batch processing
- Handles tokenization, encoding, and data augmentation

#### 3. **Model Layer (`src/models/`)**
- **`distilbert_classifier.py`**: DistilBERT-based binary classifier
- **`trainer.py`**: Training loops, validation, and model checkpointing
- **`evaluation_formatter.py`**: Converts predictions to WMT evaluation format

#### 4. **Utilities (`src/utils/`)**
- **`config.py`**: YAML configuration loading and validation
- **`logging_utils.py`**: Structured logging and progress tracking
- **`language_utils.py`**: Language pair processing utilities

#### 5. **Types (`src/types/`)**
- **`types.py`**: Custom type definitions and data structures
- Ensures type safety across the codebase

### CLI Interface (`scripts/run_pipeline.py`)
Command-line interface that provides:
- Argument parsing and validation
- Command routing (train/evaluate/predict/analyze)
- Debug mode support
- Integration with the main pipeline

## 🔄 Data Flow

### Training Flow
```
TSV Data Files → DataLoader → PyTorch Dataset → DistilBERT → Training Loop → Model Checkpoints
```

1. **Data Loading**: TSV files loaded and validated
2. **Preprocessing**: Text tokenization and encoding
3. **Model Training**: DistilBERT fine-tuning with binary classification head
4. **Checkpointing**: Best models saved based on validation metrics
5. **Logging**: Training metrics and progress tracked

### Evaluation Flow
```
Trained Model → Test Data → Predictions → Metrics Calculation → Results Output
```

1. **Model Loading**: Restore trained model from checkpoint
2. **Data Processing**: Prepare test data for inference
3. **Prediction**: Generate binary classifications
4. **Evaluation**: Calculate MCC, accuracy, F1, precision, recall
5. **Output**: JSON results and optional WMT evaluation format

### Prediction Flow
```
Model + Test Data → Inference → Probability Scores → WMT Format → Evaluation Files
```

1. **Inference**: Generate predictions on blind test data
2. **Post-processing**: Convert probabilities to binary labels
3. **Format Conversion**: Create WMT evaluation format files
4. **Metadata**: Generate model statistics and configuration info

## 📊 Model Architecture

### DistilBERT Classifier
- **Base Model**: `distilbert-base-multilingual-cased`
- **Task**: Binary sequence classification
- **Input Format**: `[CLS] source_text [SEP] target_text [SEP]`
- **Output**: Binary classification (ERR/NOT)
- **Max Sequence Length**: 512 tokens

### Training Configuration
```yaml
model:
  name: "distilbert-base-multilingual-cased"
  num_labels: 2
  max_seq_length: 512

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 3
  weight_decay: 0.01
  warmup_steps: 500

data:
  train_ratio: 0.8
  val_ratio: 0.1
  random_seed: 42
```

## 🔧 Configuration Management

### Configuration Hierarchy
1. **Default Configuration**: Hardcoded defaults in `src/utils/config.py`
2. **YAML File**: Main configuration in `config.yaml`
3. **Command Line**: Override via CLI arguments
4. **Environment Variables**: Runtime overrides

### Debug Mode
- **Purpose**: Fast testing with small datasets
- **Activation**: `--debug-test` flag
- **Effects**: 
  - Reduced dataset size (100 samples)
  - Fewer training epochs
  - Simplified logging

## 📁 Output Files

### Model Outputs
- **Checkpoints**: `results/checkpoints/best_model_epoch_X.pt`
- **Final Model**: `results/checkpoints/final_model_TIMESTAMP_LANG.pt`
- **Training History**: `results/checkpoints/training_history_TIMESTAMP_LANG.json`

### Prediction Outputs
- **JSON Results**: `results/prediction_results.json`
- **WMT Format**: `results/evaluation_LANG_METHOD.tsv`
- **Metadata**: `results/evaluation_metadata_LANG_METHOD.json`

## 🚀 Build System (Makefile)

### Target Categories
1. **Core Operations**: `train`, `evaluate`, `predict`, `analyze`
2. **Debug Mode**: `debug-train`, `debug-evaluate`, `debug-predict`, `debug-experiment`
3. **Development**: `install`, `format`, `lint`, `clean`, `status`

### Variable System
- **LANG**: Language pair selection (en-de, en-ja, en-zh, en-cs)
- **MODEL**: Custom model path override
- **SAMPLE_SIZE**: Dataset size limiting for testing
- **NO_EVAL_FORMAT**: Skip WMT evaluation format generation
