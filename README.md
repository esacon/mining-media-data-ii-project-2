# Critical Error Detection

A DistilBERT-based binary classifier for detecting critical errors in machine translation output.

## 🎯 Task Overview

Detects critical translation errors in English translations to:
- **en-de**: German
- **en-ja**: Japanese  
- **en-zh**: Chinese
- **en-cs**: Czech

**Labels**: `ERR` (critical error) / `NOT` (no critical error)  
**Metric**: Matthews Correlation Coefficient (MCC)

## 🚀 Quick Start

### Setup
```bash
make install
```

### Train a Model
```bash
# Train on English-German
make train LANG=en-de

# Debug training (fast, small sample)
make debug-train LANG=en-de
```

### Evaluate a Model
```bash
# Evaluate trained model
make evaluate LANG=en-de

# Evaluate specific model
make evaluate LANG=en-de MODEL=results/checkpoints/best_model_epoch_2.pt
```

### Generate Predictions
```bash
# Generate predictions (includes WMT evaluation format)
make predict LANG=en-de

# Skip evaluation format files
make predict LANG=en-de NO_EVAL_FORMAT=1
```

### Analyze Data
```bash
# Analyze dataset statistics
make analyze LANG=en-de
```

## 📋 Available Commands

### Core Commands
- `make train LANG=<pair>` - Train model
- `make evaluate LANG=<pair>` - Evaluate model  
- `make predict LANG=<pair>` - Generate predictions
- `make analyze LANG=<pair>` - Analyze data

### Debug Commands (fast testing)
- `make debug-train LANG=<pair>` - Debug training
- `make debug-evaluate LANG=<pair>` - Debug evaluation
- `make debug-predict LANG=<pair>` - Debug prediction
- `make debug-experiment LANG=<pair>` - Full debug pipeline

### Development
- `make install` - Install dependencies
- `make format` - Format code (black + isort)
- `make lint` - Run linting (flake8)
- `make clean` - Clean generated files
- `make status` - Show project status

### Examples
```bash
# Train with limited data for testing
make train LANG=en-de SAMPLE_SIZE=200

# Full experiment workflow
make debug-experiment LANG=en-de

# Check available models
make status
```


## ⚙️ Configuration

Edit `config.yaml` for model and training settings:

```yaml
model:
  name: "distilbert-base-multilingual-cased"
  max_seq_length: 512

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 3
```

## 📁 Project Structure

```
project-2/
├── data/catastrophic_errors/     # Training data
├── results/                      # Model outputs
│   └── checkpoints/             # Trained models
├── logs/                        # Training logs
├── src/                         # Source code
│   └── runner.py               # Main CLI entry point
├── llm/                        # LLM-based evaluation
│   ├── prompts.py              # WMT21-based prompt templates
│   ├── models.py               # LLM model implementations
│   └── evaluate_llm.py         # LLM evaluation script
└── scripts/                     # Utility scripts
```

## 📊 Data Format

**Input files**: `{lang}_majority_{split}.tsv`
```
ID	source	target	label
123	Hello world	Hallo Welt	NOT
456	Critical text	Bad translation	ERR
```

## 📈 Output Files

- **Models**: `results/checkpoints/best_model_*.pt`
- **Predictions**: `results/prediction_results.json` 
- **WMT Format**: `results/evaluation_{lang}_{method}.tsv`
- **Logs**: `logs/training_*.log`

## 🔧 Environment Variables

- `LANG=<pair>` - Language pair (en-de, en-ja, en-zh, en-cs)
- `MODEL=<path>` - Custom model path
- `SAMPLE_SIZE=<n>` - Limit dataset size
- `NO_EVAL_FORMAT=1` - Skip WMT evaluation format

## 🔍 Getting Help

```bash
# Show all commands
make help

# Show project status
make status
```

## 🔗 References

- [WMT21 Quality Estimation Task](https://www.statmt.org/wmt21/quality-estimation-task.html)
- [Critical Error Examples](https://statmt.org/wmt21/quality-estimation-task_critical-error-examples.html)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) 