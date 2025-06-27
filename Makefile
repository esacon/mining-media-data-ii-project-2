# WMT21 Task 3 - Critical Error Detection Makefile

# Project variables
PROJECT_NAME = wmt21-task3-critical-error-detection
PYTHON = python3
PIP = pip3
DATA_DIR = src/data
RESULTS_DIR = results
LOGS_DIR = logs
CONFIG_FILE = config.yaml

# Virtual environment
VENV_NAME = venv
VENV_DIR = $(VENV_NAME)
VENV_ACTIVATE = $(VENV_DIR)/bin/activate

# Default target
.PHONY: help
help:
	@echo "WMT21 Task 3 - Critical Error Detection"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@echo "  setup          - Set up the development environment"
	@echo "  install        - Install dependencies"
	@echo "  download-data  - Download WMT21 Task 3 data"
	@echo "  train          - Train the DistilBERT model"
	@echo "  evaluate       - Evaluate the trained model"
	@echo "  predict        - Generate predictions for test data"
	@echo "  clean          - Clean up generated files"
	@echo "  test           - Run tests"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black"
	@echo ""
	@echo "Training examples:"
	@echo "  make train DATA=train.tsv"
	@echo "  make train DATA=train.tsv LANG=en-de"
	@echo "  make evaluate MODEL=results/checkpoints/best_model.pt DATA=test.tsv"

# Development setup
.PHONY: setup
setup: $(VENV_ACTIVATE)
	@echo "Setting up development environment..."
	@$(MAKE) install
	@echo "Setup complete!"

$(VENV_ACTIVATE):
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

.PHONY: install
install: $(VENV_ACTIVATE)
	@echo "Installing dependencies..."
	. $(VENV_ACTIVATE) && $(PIP) install --upgrade pip
	. $(VENV_ACTIVATE) && $(PIP) install -r requirements.txt
	@echo "Dependencies installed!"

# Data management
.PHONY: download-data
download-data:
	@echo "Downloading WMT21 Task 3 data..."
	@mkdir -p $(DATA_DIR)/raw
	@echo "Please manually download data from:"
	@echo "https://github.com/WMT-QE-Task/wmt-qe-2021-data/tree/main/task3-critical-error-detection"
	@echo "and place files in $(DATA_DIR)/raw/"

.PHONY: check-data
check-data:
	@echo "Checking data structure..."
	@if [ ! -d "$(DATA_DIR)/raw" ]; then \
		echo "Error: Raw data directory not found. Run 'make download-data' first."; \
		exit 1; \
	fi
	@echo "Data check complete!"

# Training
.PHONY: train
train: check-data
	@echo "Training DistilBERT model..."
	@mkdir -p $(RESULTS_DIR)/checkpoints $(LOGS_DIR)
	. $(VENV_ACTIVATE) && $(PYTHON) train.py \
		--config $(CONFIG_FILE) \
		--data_path $(or $(DATA),$(DATA_DIR)/raw/train.tsv) \
		--output_dir $(RESULTS_DIR)/checkpoints \
		$(if $(LANG),--language_pair $(LANG)) \
		$(if $(DEBUG),--debug)

# Training with specific configurations
.PHONY: train-en-de
train-en-de:
	@$(MAKE) train LANG=en-de DATA=$(DATA_DIR)/raw/train_en-de.tsv

.PHONY: train-en-ja
train-en-ja:
	@$(MAKE) train LANG=en-ja DATA=$(DATA_DIR)/raw/train_en-ja.tsv

.PHONY: train-en-zh
train-en-zh:
	@$(MAKE) train LANG=en-zh DATA=$(DATA_DIR)/raw/train_en-zh.tsv

.PHONY: train-en-cs
train-en-cs:
	@$(MAKE) train LANG=en-cs DATA=$(DATA_DIR)/raw/train_en-cs.tsv

# Evaluation
.PHONY: evaluate
evaluate:
	@echo "Evaluating model..."
	@mkdir -p $(RESULTS_DIR)
	. $(VENV_ACTIVATE) && $(PYTHON) evaluate.py \
		--model_path $(or $(MODEL),$(RESULTS_DIR)/checkpoints/best_model.pt) \
		--data_path $(or $(DATA),$(DATA_DIR)/raw/test.tsv) \
		--config $(CONFIG_FILE) \
		--output_dir $(RESULTS_DIR) \
		$(if $(LANG),--language_pair $(LANG)) \
		$(if $(SUBMISSION),--submission_format)

# Generate predictions for submission
.PHONY: predict
predict:
	@$(MAKE) evaluate SUBMISSION=1 MODEL=$(MODEL) DATA=$(DATA)

# Testing and quality
.PHONY: test
test: $(VENV_ACTIVATE)
	@echo "Running tests..."
	. $(VENV_ACTIVATE) && $(PYTHON) -m pytest src/tests/ -v

.PHONY: lint
lint: $(VENV_ACTIVATE)
	@echo "Running linter..."
	. $(VENV_ACTIVATE) && flake8 src/ train.py evaluate.py --max-line-length=100

.PHONY: format
format: $(VENV_ACTIVATE)
	@echo "Formatting code..."
	. $(VENV_ACTIVATE) && black src/ train.py evaluate.py --line-length=100
	. $(VENV_ACTIVATE) && isort src/ train.py evaluate.py

# Analysis and visualization
.PHONY: analyze-data
analyze-data: $(VENV_ACTIVATE)
	@echo "Analyzing dataset..."
	. $(VENV_ACTIVATE) && $(PYTHON) -c "
from src.data.dataset import CriticalErrorDataset
from transformers import DistilBertTokenizer
import yaml

with open('$(CONFIG_FILE)') as f:
    config = yaml.safe_load(f)

tokenizer = DistilBertTokenizer.from_pretrained(config['model']['model_name'])
dataset = CriticalErrorDataset('$(DATA_DIR)/raw/train.tsv', tokenizer)

print('Dataset Statistics:')
print(f'Total samples: {len(dataset)}')
print('Label distribution:', dataset.get_label_distribution())
print('Language distribution:', dataset.get_language_distribution())
"

.PHONY: plot-training
plot-training: $(VENV_ACTIVATE)
	@echo "Plotting training history..."
	. $(VENV_ACTIVATE) && $(PYTHON) -c "
import json
import matplotlib.pyplot as plt

with open('$(RESULTS_DIR)/checkpoints/training_history.json') as f:
    history = json.load(f)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Loss
ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Validation')
ax1.set_title('Loss')
ax1.legend()

# MCC
ax2.plot(history['val_mcc'])
ax2.set_title('Validation MCC')

# Accuracy
ax3.plot(history['val_accuracy'])
ax3.set_title('Validation Accuracy')

# Learning Rate
ax4.plot(history['learning_rates'])
ax4.set_title('Learning Rate')

plt.tight_layout()
plt.savefig('$(RESULTS_DIR)/training_plots.png', dpi=300)
plt.show()
"

# Cleanup
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(LOGS_DIR)/*
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Cleanup complete!"

.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)

# Model export
.PHONY: export-model
export-model:
	@echo "Exporting model for deployment..."
	@mkdir -p $(RESULTS_DIR)/export
	. $(VENV_ACTIVATE) && $(PYTHON) -c "
import torch
from transformers import DistilBertTokenizer
from src.models.distilbert_classifier import DistilBERTClassifier

# Load best model
checkpoint = torch.load('$(RESULTS_DIR)/checkpoints/best_model.pt', map_location='cpu')
config = checkpoint['config']

# Load model and tokenizer
model = DistilBERTClassifier.from_pretrained(
    config['model']['model_name'],
    num_labels=config['model']['num_labels']
)
model.load_state_dict(checkpoint['model_state_dict'])
tokenizer = DistilBertTokenizer.from_pretrained(config['model']['model_name'])

# Save for deployment
model.save_pretrained('$(RESULTS_DIR)/export/model')
tokenizer.save_pretrained('$(RESULTS_DIR)/export/tokenizer')

print('Model exported to $(RESULTS_DIR)/export/')
"

# Quick start example
.PHONY: example
example:
	@echo "Running quick example..."
	@echo "This would run a small training example with sample data"
	@echo "Please ensure you have downloaded the WMT21 data first"

# Docker support (if needed)
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t $(PROJECT_NAME) .

.PHONY: docker-run
docker-run:
	@echo "Running in Docker..."
	docker run -it --rm -v $(PWD):/workspace $(PROJECT_NAME)

# Show project status
.PHONY: status
status:
	@echo "Project Status:"
	@echo "==============="
	@echo "Virtual environment: $(if $(wildcard $(VENV_ACTIVATE)),✓ Active,✗ Not found)"
	@echo "Configuration file: $(if $(wildcard $(CONFIG_FILE)),✓ Found,✗ Missing)"
	@echo "Data directory: $(if $(wildcard $(DATA_DIR)/raw),✓ Found,✗ Missing)"
	@echo "Results directory: $(if $(wildcard $(RESULTS_DIR)),✓ Found,✗ Missing)"
	@echo "Trained models: $(shell find $(RESULTS_DIR)/checkpoints -name "*.pt" 2>/dev/null | wc -l) found"
