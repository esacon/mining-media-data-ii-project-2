# WMT21 Task 3 - Critical Error Detection
# Simplified Makefile (assumes you're already in your virtual environment)

# Project variables
DATA_DIR = data
RESULTS_DIR = results
LOGS_DIR = logs
CONFIG_FILE = config.yaml

# Default target
.PHONY: help
help:
	@echo "WMT21 Task 3 - Critical Error Detection"
	@echo "======================================"
	@echo ""
	@echo "Core Commands:"
	@echo "  train          - Train model (use LANG=en-de for specific language)"
	@echo "  evaluate       - Evaluate model (use MODEL=path/to/model.pt)"
	@echo "  predict        - Generate predictions"
	@echo "  analyze        - Run data analysis"
	@echo "  experiment     - Run full train+evaluate experiment"
	@echo ""
	@echo "Development:"
	@echo "  install        - Install dependencies"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  clean          - Clean up generated files"
	@echo "  status         - Show project status"
	@echo ""
	@echo "Examples:"
	@echo "  make train LANG=en-de"
	@echo "  make evaluate MODEL=results/checkpoints/best_model.pt LANG=en-de"

# Installation
.PHONY: install
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Core pipeline commands
.PHONY: train
train: check-data
	@echo "Training model..."
	@mkdir -p $(RESULTS_DIR)/checkpoints $(LOGS_DIR)
	python scripts/run_pipeline.py train \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(CONFIG),--config $(CONFIG),--config $(CONFIG_FILE)) \
		$(if $(DEBUG),--debug)

.PHONY: evaluate
evaluate:
	@echo "Evaluating model..."
	@mkdir -p $(RESULTS_DIR)
	python scripts/run_pipeline.py evaluate \
		$(if $(MODEL),--model-path $(MODEL)) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(CONFIG),--config $(CONFIG),--config $(CONFIG_FILE))

.PHONY: predict
predict:
	@echo "Generating predictions..."
	python scripts/run_pipeline.py predict \
		$(if $(MODEL),--model-path $(MODEL)) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(CONFIG),--config $(CONFIG),--config $(CONFIG_FILE))

.PHONY: analyze
analyze:
	@echo "Running data analysis..."
	@mkdir -p $(RESULTS_DIR)
	python scripts/run_pipeline.py analyze \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(CONFIG),--config $(CONFIG),--config $(CONFIG_FILE))

.PHONY: experiment
experiment:
	@echo "Running full experiment..."
	python scripts/run_pipeline.py experiment \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(CONFIG),--config $(CONFIG),--config $(CONFIG_FILE))

# Quick training shortcuts
.PHONY: train-en-de train-en-ja train-en-zh train-en-cs
train-en-de:
	@$(MAKE) train LANG=en-de

train-en-ja:
	@$(MAKE) train LANG=en-ja

train-en-zh:
	@$(MAKE) train LANG=en-zh

train-en-cs:
	@$(MAKE) train LANG=en-cs

# Data validation
.PHONY: check-data
check-data:
	@echo "Checking data..."
	@if [ ! -d "$(DATA_DIR)/catastrophic_errors" ]; then \
		echo "Error: Data directory not found at $(DATA_DIR)/catastrophic_errors/"; \
		exit 1; \
	fi

# Development tools
.PHONY: test
test:
	@echo "Running tests..."
	python -m pytest src/ -v

.PHONY: lint
lint:
	@echo "Running linter..."
	flake8 src/ scripts/ pipeline/ --max-line-length=100 --ignore=E203,W503

.PHONY: format
format:
	@echo "Formatting code..."
	black src/ scripts/ pipeline/ --line-length=100
	isort src/ scripts/ pipeline/

# Project status and utilities
.PHONY: status
status:
	@echo "Project Status:"
	@echo "==============="
	@echo "Python: $(shell python --version)"
	@echo "Working directory: $(shell pwd)"
	@echo "Config file: $(if $(wildcard $(CONFIG_FILE)),✓ Found,✗ Missing)"
	@echo "Data directory: $(if $(wildcard $(DATA_DIR)/catastrophic_errors),✓ Found,✗ Missing)"
	@echo "Results directory: $(if $(wildcard $(RESULTS_DIR)),✓ Found,✗ Missing)"
	@echo "Trained models: $(shell find $(RESULTS_DIR)/checkpoints -name "*.pt" 2>/dev/null | wc -l | tr -d ' ') found"

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(LOGS_DIR)/* 2>/dev/null || true
	rm -rf __pycache__/ **/__pycache__/ 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

# Quick data overview
.PHONY: data-summary
data-summary:
	@echo "Data Summary:"
	@echo "============="
	@for lang in en-de en-ja en-zh en-cs; do \
		if [ -f "$(DATA_DIR)/catastrophic_errors/$${lang}_majority_train.tsv" ]; then \
			train_count=$$(wc -l < "$(DATA_DIR)/catastrophic_errors/$${lang}_majority_train.tsv"); \
			dev_count=$$(wc -l < "$(DATA_DIR)/catastrophic_errors/$${lang}_majority_dev.tsv" 2>/dev/null || echo "0"); \
			echo "$$lang: Train=$$train_count, Dev=$$dev_count"; \
		fi; \
	done
