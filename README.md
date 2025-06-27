# WMT21 Task 3 - Critical Error Detection

This project implements a DistilBERT-based classifier for **WMT21 Quality Estimation Task 3: Critical Error Detection**. The model performs binary classification to detect whether machine translation output contains critical errors that could have health, safety, legal, reputation, religious, or financial implications.

## 🎯 Task Overview

**Task 3** focuses on detecting critical translation errors in Wikipedia comments translated from English to:
- Czech (en-cs) → `encs_majority_*.tsv` files
- Japanese (en-ja) → `enja_majority_*.tsv` files
- Chinese (en-zh) → `enzh_majority_*.tsv` files  
- German (en-de) → `ende_majority_*.tsv` files

### Critical Error Categories
Based on the [WMT21 Critical Error Examples](https://statmt.org/wmt21/quality-estimation-task_critical-error-examples.html):
- **TOX**: Toxicity deviation (hate speech, violence, profanity)
- **SAF**: Safety risks deviation
- **NAM**: Named entities deviation
- **SEN**: Sentiment polarity or negation deviation  
- **NUM**: Units/time/date/numbers deviation

### Data Format
- **Training/Dev**: `ID\tsource\ttarget\t[annotator_scores]\taggregated_label`
- **Test**: `ID\tsource\ttarget` (blind evaluation)
- **Labels**: `ERR` (critical error) vs `NOT` (no critical error)
- **Annotation**: 3 annotators with majority voting

### Evaluation Metric
Primary metric: **Matthews Correlation Coefficient (MCC)** as per WMT21 official evaluation

## ✨ Key Features

This implementation provides:

### 🎯 **Task-Specific Design**
- **Real WMT21 Data**: Works directly with your actual `catastrophic_errors/` data files
- **Proper Format Handling**: Parses TSV format with annotator scores and majority voting
- **Official Evaluation**: Includes the exact `sent_evaluate_CED.py` script from WMT21
- **Submission Ready**: Generates files in the exact WMT21 submission format

### 🧠 **Smart Model Architecture** 
- **DistilBERT Multilingual**: Optimized for all 4 language pairs (en-de, en-ja, en-zh, en-cs)
- **MCC Optimization**: Training specifically optimized for Matthews Correlation Coefficient
- **Class Imbalance Handling**: Handles the typical 15-25% critical error rate
- **Efficient Fine-tuning**: Fast training with DistilBERT vs full BERT

### 📊 **Comprehensive Analysis**
- **Inter-annotator Agreement Analysis**: Understand annotation quality
- **Language-specific Statistics**: Compare error rates across language pairs  
- **Data Distribution Insights**: Visualize score patterns and agreements
- **Training Monitoring**: Track MCC, F1, precision, recall during training

### 🚀 **Production Ready**
- **Automated Pipeline**: From data loading to WMT21 submission generation
- **Model Size Tracking**: Automatic calculation of parameters and disk footprint
- **Validation Pipeline**: Ensures submission format compliance
- **Easy Deployment**: Export trained models for inference

## 🏗️ Project Structure

```
project-2/
├── config.yaml                    # ✅ WMT21 Task 3 configuration
├── train.py                       # ✅ Main training script
├── evaluate.py                    # ✅ Evaluation with WMT21 submission format
├── analyze_data.py                # ✅ Data analysis and visualization
├── Makefile                       # ✅ Automation commands
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── src/                           # Source code
│   ├── data/                      # ✅ WMT21 data handling
│   │   ├── __init__.py
│   │   ├── dataset.py             # ✅ PyTorch dataset for WMT21 format
│   │   ├── data_loader.py         # ✅ WMT21 TSV loader with label parsing
│   │   ├── catastrophic_errors/   # ✅ YOUR WMT21 TASK 3 DATA
│   │   │   ├── ende_majority_train.tsv     # English-German training
│   │   │   ├── ende_majority_dev.tsv       # English-German development
│   │   │   ├── ende_majority_test_blind.tsv # English-German test
│   │   │   ├── enja_majority_*.tsv         # English-Japanese data
│   │   │   ├── enzh_majority_*.tsv         # English-Chinese data
│   │   │   ├── encs_majority_*.tsv         # English-Czech data
│   │   │   └── readme                      # Data format description
│   │   ├── catastrophic_errors_goldlabels/ # ✅ Gold test labels
│   │   │   └── *_majority_test_goldlabels.tar.gz
│   │   └── processed/             # Processed data cache
│   │
│   ├── models/                    # ✅ DistilBERT model components
│   │   ├── __init__.py
│   │   ├── distilbert_classifier.py  # ✅ Fine-tuned DistilBERT
│   │   ├── trainer.py             # ✅ Training with MCC optimization
│   │   └── submission.py          # ✅ WMT21 submission formatter
│   │
│   ├── evaluation/                # ✅ WMT21 evaluation
│   │   ├── sent_evaluate_CED.py   # ✅ Official WMT21 evaluation script
│   │   └── README.md              # Evaluation guidelines
│   │
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   └── logging_utils.py       # Logging configuration
│   │
│   └── tests/                     # Unit tests
│
├── results/                       # Training outputs
│   ├── checkpoints/               # Model checkpoints
│   ├── figures/                   # Training plots
│   ├── data_analysis/             # Data analysis results
│   └── submissions/               # WMT21 submission files
│
└── logs/                          # Training logs
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Setup virtual environment and install dependencies
make setup
make install
```

### 2. Explore Your Data

Your project already contains the WMT21 Task 3 dataset! Analyze it first:

```bash
# Quick data exploration
make quick-analysis

# Full data analysis with visualizations
python analyze_data.py --data-dir src/data/catastrophic_errors --output-dir results/data_analysis
```

**Available Data Files**:
- **Training**: `src/data/catastrophic_errors/*_majority_train.tsv` (~7,879 samples each)
- **Development**: `src/data/catastrophic_errors/*_majority_dev.tsv` (~1,000 samples each)  
- **Test (Blind)**: `src/data/catastrophic_errors/*_majority_test_blind.tsv` (1,000 samples each)
- **Gold Labels**: `src/data/catastrophic_errors_goldlabels/*.tar.gz`

### 3. Train Your Model

Start with one language pair (recommended):

```bash
# Train on English-German (good starting point)
python train.py --config config.yaml --language-pair en-de

# Or use Makefile shortcuts
make train-en-de
```

For all language pairs:
```bash
# Train on each language pair
python train.py --config config.yaml --language-pair en-de
python train.py --config config.yaml --language-pair en-ja
python train.py --config config.yaml --language-pair en-zh
python train.py --config config.yaml --language-pair en-cs
```

### 4. Evaluate and Submit

```bash
# Evaluate and create WMT21 submission format
python evaluate.py --model-path results/checkpoints/best_en-de.pt \
                   --data-path src/data/catastrophic_errors/ende_majority_test_blind.tsv \
                   --language-pair en-de \
                   --output-dir results/submissions/

# Use official WMT21 evaluation script
python src/evaluation/sent_evaluate_CED.py \
       results/submissions/submission_en-de.txt \
       path/to/goldlabels_en-de.txt
```

## 📊 Model Architecture

### DistilBERT Configuration
- **Base Model**: `distilbert-base-multilingual-cased`
- **Task**: Binary sequence classification
- **Input Format**: `[CLS] source_text [SEP] target_text [SEP]`
- **Output**: Binary classification (0: no critical error, 1: critical error)

### Key Features
- Multilingual support for all WMT21 Task 3 language pairs
- Fine-tuning with task-specific classification head
- Matthews Correlation Coefficient optimization
- Mixed precision training support
- Comprehensive evaluation metrics

## ⚙️ Configuration

The main configuration is in `config.yaml`:

```yaml
# Model settings
model:
  model_name: "distilbert-base-multilingual-cased"
  num_labels: 2
  max_seq_length: 512
  
  training:
    batch_size: 16
    learning_rate: 2e-5
    num_epochs: 3
    weight_decay: 0.01

# Evaluation metrics
evaluation:
  primary_metric: "mcc"  # Matthews Correlation Coefficient
  additional_metrics:
    - "accuracy"
    - "precision" 
    - "recall"
    - "f1"
    - "auc_roc"
```

## 🔧 Available Commands

### Development
```bash
make setup          # Setup development environment
make install        # Install dependencies
make test           # Run tests
make lint           # Run code linting
make format         # Format code
```

### Training
```bash
make train          # Train model
make train-en-de    # Train on English-German data
make train-en-ja    # Train on English-Japanese data
make train-en-zh    # Train on English-Chinese data
make train-en-cs    # Train on English-Czech data
```

### Analysis & Evaluation
```bash
make analyze-data   # Analyze your WMT21 data with visualizations
make quick-analysis # Quick exploration of data statistics
make evaluate       # Evaluate model with WMT21 metrics
make predict        # Generate WMT21 submission files
```

### Utilities
```bash
make clean          # Clean generated files
make export-model   # Export model for deployment
make status         # Show project status
```

## 📈 Training Process

1. **Data Loading**: Loads TSV/JSON files with source/target pairs and labels
2. **Preprocessing**: Tokenizes text using DistilBERT tokenizer
3. **Model Training**: Fine-tunes DistilBERT with classification head
4. **Evaluation**: Computes MCC and other metrics on validation set
5. **Checkpointing**: Saves best model based on validation MCC

### Training Output
- Model checkpoints in `results/checkpoints/`
- Training logs in `logs/`
- Training history in `results/checkpoints/training_history.json`
- Evaluation plots in `results/figures/`

## 📊 Evaluation Metrics

### Primary Metric
- **MCC (Matthews Correlation Coefficient)**: Primary evaluation metric for WMT21 Task 3

### Additional Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: Visual representation of classification results

## 💾 Data Format

### WMT21 Task 3 Format

**Training/Development Files** (`*_majority_train.tsv`, `*_majority_dev.tsv`):
```
ID	source	target	[annotator_scores]	aggregated_label
9026	Phalin maybe worthy of a section...	Phalin kann vielleicht einen Abschnitt...	[0, 0, 0]	NOT
3255	LOL yeah good one mate...	LOL ja gut ein Freund...	[1, 1, 1]	ERR
```

**Test Files** (`*_majority_test_blind.tsv`):
```
ID	source	target
1234	Original English text	Translated text
5678	Another English text	Translation output
```

### Label System
- **Individual Scores**: `[0, 0, 0]` = 3 annotators, each scoring 0 (no error) or 1 (critical error)
- **Aggregated Labels**: 
  - `NOT` = No critical error (majority voted 0)
  - `ERR` = Critical error present (majority voted 1)
- **Binary Conversion**: `ERR` → 1, `NOT` → 0 for model training

### Annotation Process
- **3 Human Annotators** per sentence pair
- **Majority Voting** for final label (ignoring scores 2/3 for unclear sentences)
- **Critical Error Types**: TOX, SAF, NAM, SEN, NUM (see examples in WMT21 documentation)

## 🔍 Model Performance

The model is optimized for:
- **High MCC scores** (primary WMT21 metric)
- **Balanced precision and recall** for critical error detection
- **Multilingual performance** across all language pairs
- **Efficient inference** suitable for production deployment

## 📦 Dependencies

Key dependencies:
- `torch`: PyTorch deep learning framework
- `transformers`: Hugging Face transformers library
- `scikit-learn`: Metrics and evaluation utilities
- `pandas`: Data manipulation
- `matplotlib/seaborn`: Visualization
- `tqdm`: Progress bars
- `pyyaml`: Configuration management

## 🚀 Deployment

### Export Trained Model
```bash
make export-model
```

This creates a deployment-ready model in `results/export/` with:
- Model weights and configuration
- Tokenizer files
- Metadata for WMT21 submission

### Model Size Information
The exported model includes:
- **Total parameters**: ~134M (DistilBERT base)
- **Disk footprint**: ~500MB
- **Inference speed**: Optimized for real-time critical error detection

## 📝 WMT21 Submission

The evaluation script can generate WMT21-compatible submission files:

```bash
make predict MODEL=results/checkpoints/best_model.pt DATA=test.tsv
```

Output includes:
- `submission.tsv`: Predictions in WMT21 format
- `submission_metadata.json`: Model size and performance info

## 🧪 Testing

Run the test suite:
```bash
make test
```

Tests cover:
- Data loading and preprocessing
- Model architecture
- Training pipeline
- Evaluation metrics
- Edge cases and error handling

## 📊 Example Usage

### Training a Model
```python
from src.models.distilbert_classifier import DistilBERTClassifier
from src.models.trainer import Trainer
from src.data.dataset import CriticalErrorDataset

# Load data and model
dataset = CriticalErrorDataset("train.tsv", tokenizer)
model = DistilBERTClassifier.from_pretrained("distilbert-base-multilingual-cased")

# Train
trainer = Trainer(model, tokenizer, config, device)
results = trainer.train(train_loader, val_loader, "checkpoints/")
```

### Making Predictions
```python
# Load trained model
model, tokenizer, config = load_model("best_model.pt", device)

# Make predictions
predictions, probabilities = trainer.predict(test_loader)
```

## 🤝 Contributing

1. Follow the coding standards (run `make format` and `make lint`)
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass (`make test`)

## 📄 License

This project is part of academic research for WMT21 Quality Estimation Task 3.

## 🔗 References

- [WMT21 Quality Estimation Task](https://www.statmt.org/wmt21/quality-estimation-task.html)
- [Task 3 Data Repository](https://github.com/WMT-QE-Task/wmt-qe-2021-data/tree/main/task3-critical-error-detection)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/transformers/) 