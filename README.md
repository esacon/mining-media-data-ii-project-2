# WMT21 Task 3 - Critical Error Detection

A DistilBERT-based binary classifier for **WMT21 Quality Estimation Task 3: Critical Error Detection**. This model detects whether machine translation output contains critical errors that could have serious implications (health, safety, legal, financial, etc.).

## 🎯 Task Overview

**Task 3** detects critical translation errors in Wikipedia comments translated from English to:
- Czech (en-cs) 
- Japanese (en-ja)
- Chinese (en-zh) 
- German (en-de)

**Labels**: 
- `ERR` = Critical error present
- `NOT` = No critical error

**Evaluation Metric**: Matthews Correlation Coefficient (MCC)

## 🏗️ Project Structure

```
project-2/
├── config.yaml                    # Configuration
├── scripts/
│   └── run_pipeline.py            # Main CLI script
├── src/
│   ├── data/                      # Data loading and processing
│   │   ├── dataset.py            # PyTorch dataset
│   │   └── data_loader.py        # WMT21 data loader
│   ├── models/                   # Model components
│   │   ├── distilbert_classifier.py  # DistilBERT model
│   │   └── trainer.py            # Training logic
│   └── utils/                    # Utilities
│       ├── config.py             # Configuration loading
│       └── logging_utils.py      # Logging
├── data/
│   ├── catastrophic_errors/      # WMT21 training/dev/test data
│   └── evaluation/               # Official evaluation scripts
└── results/                      # Training outputs and checkpoints
```

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or using pipenv
pipenv install
pipenv shell
```

### 2. Train a Model

Train on a specific language pair:

```bash
# Train on English-Czech data
python scripts/run_pipeline.py train \
    --data data/catastrophic_errors \
    --language-pair en-cs

# Train on English-German data  
python scripts/run_pipeline.py train \
    --data data/catastrophic_errors \
    --language-pair en-de
```

### 3. Evaluate a Model

Evaluate your trained model:

```bash
# Evaluate on development set
python scripts/run_pipeline.py evaluate \
    --model results/checkpoints/final_model.pt \
    --data data/catastrophic_errors/encs_majority_dev.tsv \
    --language-pair en-cs
```

### 4. Make Predictions

Generate predictions for test data:

```bash
# Predict on test set
python scripts/run_pipeline.py predict \
    --model results/checkpoints/final_model.pt \
    --data data/catastrophic_errors/encs_majority_test_blind.tsv \
    --language-pair en-cs
```

## ⚙️ Configuration

Main settings in `config.yaml`:

```yaml
# Model settings
model:
  name: "distilbert-base-multilingual-cased"
  num_labels: 2
  max_seq_length: 512

# Training settings  
training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 3
  weight_decay: 0.01

# Data settings
data:
  train_ratio: 0.8
  val_ratio: 0.1
  random_seed: 42
```

## 📊 Model Architecture

- **Base Model**: DistilBERT Multilingual (`distilbert-base-multilingual-cased`)
- **Task**: Binary sequence classification
- **Input**: `[CLS] source_text [SEP] target_text [SEP]`
- **Output**: Probability of critical error (0 = no error, 1 = critical error)

## 📈 Available Commands

### Core Operations
```bash
# Show pipeline summary
python scripts/run_pipeline.py summary

# Analyze data statistics
python scripts/run_pipeline.py analyze \
    --data-dir data/catastrophic_errors \
    --language-pairs en-cs en-de

# Full experiment (train + evaluate)
python scripts/run_pipeline.py experiment \
    --data data/catastrophic_errors \
    --language-pair en-cs
```

### Training Options
```bash
# Train with custom device
python scripts/run_pipeline.py train \
    --data data/catastrophic_errors \
    --language-pair en-cs \
    --device cuda

# Train with custom config
python scripts/run_pipeline.py train \
    --data data/catastrophic_errors \
    --language-pair en-cs \
    --config custom_config.yaml
```

## 💾 Data Format

**Training/Dev files** (`*_majority_train.tsv`, `*_majority_dev.tsv`):
```
ID	source	target	[scores]	label
9026	English text	German translation	[0,0,0]	NOT
3255	English text	Bad translation	[1,1,1]	ERR
```

**Test files** (`*_majority_test_blind.tsv`):
```
ID	source	target
1234	English text	Translation to evaluate
```

## 📊 Evaluation Metrics

- **MCC**: Matthews Correlation Coefficient (primary metric)
- **Accuracy**: Overall classification accuracy  
- **F1**: F1-score for balanced evaluation
- **Precision/Recall**: Per-class performance

## 🔧 Dependencies

Key libraries:
- `torch`: PyTorch framework
- `transformers`: Hugging Face transformers
- `scikit-learn`: Evaluation metrics
- `pandas`: Data processing
- `pyyaml`: Configuration files

## 🎯 Usage Examples

### Python API

```python
from pipeline import MainPipeline

# Initialize pipeline
pipeline = MainPipeline("config.yaml", device="auto")

# Train model
results = pipeline.train("data/catastrophic_errors", "en-cs")

# Evaluate model  
metrics = pipeline.evaluate(
    "results/checkpoints/final_model.pt",
    "data/catastrophic_errors/encs_majority_dev.tsv", 
    "en-cs"
)

# Make predictions
predictions, probabilities = pipeline.predict(
    "results/checkpoints/final_model.pt",
    "data/catastrophic_errors/encs_majority_test_blind.tsv",
    "en-cs" 
)
```

## 📝 Output Files

After training, you'll find:
- `results/checkpoints/final_model.pt` - Final trained model
- `results/checkpoints/best_model_epoch_X.pt` - Best model during training
- `results/checkpoints/training_history.json` - Training metrics
- `logs/` - Training logs

## 🔍 Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce `batch_size` in config.yaml
2. **Model loading errors**: Check PyTorch version compatibility
3. **Data format errors**: Ensure TSV files have correct column structure

**Getting Help:**
```bash
# Show available commands
python scripts/run_pipeline.py --help

# Show command-specific help
python scripts/run_pipeline.py train --help
```

## 🔗 References

- [WMT21 Quality Estimation Task](https://www.statmt.org/wmt21/quality-estimation-task.html)
- [Critical Error Examples](https://statmt.org/wmt21/quality-estimation-task_critical-error-examples.html)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) 