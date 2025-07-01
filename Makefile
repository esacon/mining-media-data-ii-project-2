DATA_DIR = data/catastrophic_errors
RESULTS_DIR = results
LOGS_DIR = logs
CONFIG_FILE = config.yaml
PYTHON_SOURCE_DIRS = src scripts llm

define load_best_model
$(or $(shell find $(RESULTS_DIR)/checkpoints -name "best_model_*.pt" -type f | sort -r | head -1), \
     $(shell find $(RESULTS_DIR)/checkpoints -name "final_model_*.pt" -type f | sort -r | head -1), \
     $(shell find $(RESULTS_DIR)/checkpoints -name "*.pt" -type f | sort -r | head -1))
endef

DEFAULT_MODEL = $(call load_best_model)

BLUE=\033[34m
GREEN=\033[32m
YELLOW=\033[33m
RED=\033[31m
RESET=\033[0m
BOLD=\033[1m

.PHONY: help install train evaluate predict analyze debug-train debug-evaluate debug-predict debug-experiment status clean check-performance check-llm-performance check-all-performance format lint

help:
	@echo "$(BOLD)Critical Error Detection$(RESET)"
	@echo "$(BOLD)======================================"
	@echo ""
	@echo "$(BOLD)Core Commands:$(RESET)"
	@echo "  $(BLUE)train [LANG=en-de]     $(RESET)- Train model"
	@echo "  $(BLUE)evaluate [LANG=en-de]  $(RESET)- Evaluate model"
	@echo "  $(BLUE)predict [LANG=en-de]   $(RESET)- Generate predictions (JSON + WMT format)"
	@echo "  $(BLUE)analyze [LANG=en-de]   $(RESET)- Run data analysis"
	@echo ""
	@echo "$(BOLD)Debug Commands:$(RESET)"
	@echo "  $(BLUE)debug-train [LANG=en-de]    $(RESET)- Debug train (small sample, fast)"
	@echo "  $(BLUE)debug-evaluate [LANG=en-de] $(RESET)- Debug evaluate (small sample)"
	@echo "  $(BLUE)debug-predict [LANG=en-de]  $(RESET)- Debug predict (small sample)"
	@echo "  $(BLUE)debug-experiment [LANG=en-de]$(RESET)- Debug full experiment"
	@echo ""
	@echo "$(BOLD)Development & Quality:$(RESET)"
	@echo "  $(BLUE)install$(RESET)        - Install dependencies and pre-commit hooks"
	@echo "  $(BLUE)clean$(RESET)          - Clean up generated files and caches"
	@echo "  $(BLUE)status$(RESET)         - Show project status"
	@echo "  $(BLUE)check-performance$(RESET) - Analyze model performance and metrics"
	@echo "  $(BLUE)check-llm-performance$(RESET) - Analyze LLM evaluation results"
	@echo "  $(BLUE)check-all-performance$(RESET) - Analyze ALL performance (Traditional + LLM)"
	@echo "  $(BLUE)format$(RESET)         - Format code with black and isort"
	@echo "  $(BLUE)lint$(RESET)           - Run linting with flake8"
	@echo "  $(BLUE)all$(RESET)            - Run clean, format, and lint"
	@echo ""
	@echo "$(BOLD)LLM Commands:$(RESET)"
	@echo "  $(BLUE)evaluate-llm [LANG=en-de] $(RESET)- Evaluate LLM models"
	@echo "  $(BLUE)evaluate-llm-all         $(RESET)- Evaluate ALL models on ALL languages"
	@echo ""
	@echo "$(BOLD)Examples:$(RESET)"
	@echo "  make train LANG=en-de"
	@echo "  make debug-train LANG=en-de"
	@echo "  make train LANG=en-de SAMPLE_SIZE=200"
	@echo "  make evaluate LANG=en-de"
	@echo "  make predict LANG=en-de MODEL=path/to/model.pt"
	@echo "  make predict LANG=en-de NO_EVAL_FORMAT=1"
	@echo "  make evaluate-llm-all"
	@echo "  make evaluate-llm-all SAMPLE_SIZE=50"
	@echo "  make check-llm-performance"
	@echo "  make check-all-performance"
	@echo ""
	@echo "$(BOLD)Variables:$(RESET)"
	@echo "  LANG=<pair>       Language pair (en-de, en-ja, en-zh, en-cs)"
	@echo "  MODEL=<path>      Model path (default: $(DEFAULT_MODEL))"
	@echo "  SAMPLE_SIZE=<n>   Limit dataset to n samples (for testing)"
	@echo "  NO_EVAL_FORMAT=1  Skip creating WMT evaluation format files"

install:
	@echo "$(BOLD)Installing dependencies...$(RESET)"
	pip install --upgrade pip > /dev/null
	pip install -r requirements.txt > /dev/null
	pip install -e . > /dev/null
	@echo "$(GREEN)Dependencies installed successfully$(RESET)"

train:
	@echo "$(BOLD)Training model...$(RESET)"
	@mkdir -p $(RESULTS_DIR)/checkpoints $(LOGS_DIR)
	python -m src.runner --config $(CONFIG_FILE) train \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE))

evaluate:
	@echo "$(BOLD)Evaluating model...$(RESET)"
	@mkdir -p $(RESULTS_DIR)
	$(eval MODEL_PATH := $(or $(MODEL),$(DEFAULT_MODEL)))
	@if [ -z "$(MODEL_PATH)" ] || [ ! -f "$(MODEL_PATH)" ]; then \
		echo "$(RED)Error: No model specified or model not found$(RESET)"; \
		echo "Available models:"; \
		find $(RESULTS_DIR)/checkpoints -name "*.pt" -type f 2>/dev/null | sed 's/^/  /' || echo "  No models found - run 'make train' first"; \
		echo "Usage: make evaluate MODEL=path/to/model.pt"; \
		exit 1; \
	fi
	python -m src.runner --config $(CONFIG_FILE) evaluate \
		--model "$(MODEL_PATH)" \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE))

predict:
	@echo "$(BOLD)Generating predictions...$(RESET)"
	$(eval MODEL_PATH := $(or $(MODEL),$(DEFAULT_MODEL)))
	@if [ -z "$(MODEL_PATH)" ] || [ ! -f "$(MODEL_PATH)" ]; then \
		echo "$(RED)Error: No model specified or model not found$(RESET)"; \
		echo "Available models:"; \
		find $(RESULTS_DIR)/checkpoints -name "*.pt" -type f 2>/dev/null | sed 's/^/  /' || echo "  No models found - run 'make train' first"; \
		echo "Usage: make predict MODEL=path/to/model.pt"; \
		exit 1; \
	fi
	python -m src.runner --config $(CONFIG_FILE) predict \
		--model "$(MODEL_PATH)" \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE)) \
		$(if $(NO_EVAL_FORMAT),--no-evaluation-format)

analyze:
	@echo "$(BOLD)Running data analysis...$(RESET)"
	@mkdir -p $(RESULTS_DIR)
	python -m src.runner --config $(CONFIG_FILE) analyze \
		--data-dir $(DATA_DIR) \
		$(if $(LANG),--language-pairs $(LANG))

debug-train:
	@echo "$(BOLD)Debug training model (fast testing mode)...$(RESET)"
	@mkdir -p $(RESULTS_DIR)/checkpoints $(LOGS_DIR)
	python -m src.runner --config $(CONFIG_FILE) --debug-test train \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE))

debug-evaluate:
	@echo "$(BOLD)Debug evaluating model (fast testing mode)...$(RESET)"
	@mkdir -p $(RESULTS_DIR)
	$(eval MODEL_PATH := $(or $(MODEL),$(DEFAULT_MODEL)))
	@if [ -z "$(MODEL_PATH)" ] || [ ! -f "$(MODEL_PATH)" ]; then \
		echo "$(RED)Error: No model specified or model not found$(RESET)"; \
		echo "Available models:"; \
		find $(RESULTS_DIR)/checkpoints -name "*.pt" -type f 2>/dev/null | sed 's/^/  /' || echo "  No models found - run 'make train' first"; \
		echo "Usage: make debug-evaluate MODEL=path/to/model.pt"; \
		exit 1; \
	fi
	python -m src.runner --config $(CONFIG_FILE) --debug-test evaluate \
		--model "$(MODEL_PATH)" \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE))

debug-predict:
	@echo "$(BOLD)Debug generating predictions (fast testing mode)...$(RESET)"
	$(eval MODEL_PATH := $(or $(MODEL),$(DEFAULT_MODEL)))
	@if [ -z "$(MODEL_PATH)" ] || [ ! -f "$(MODEL_PATH)" ]; then \
		echo "$(RED)Error: No model specified or model not found$(RESET)"; \
		echo "Available models:"; \
		find $(RESULTS_DIR)/checkpoints -name "*.pt" -type f 2>/dev/null | sed 's/^/  /' || echo "  No models found - run 'make train' first"; \
		echo "Usage: make debug-predict MODEL=path/to/model.pt"; \
		exit 1; \
	fi
	python -m src.runner --config $(CONFIG_FILE) --debug-test predict \
		--model "$(MODEL_PATH)" \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE)) \
		$(if $(NO_EVAL_FORMAT),--no-evaluation-format)

debug-experiment:
	@echo "$(BOLD)Debug full experiment (fast testing mode)...$(RESET)"
	@mkdir -p $(RESULTS_DIR)/checkpoints $(LOGS_DIR)
	python -m src.runner --config $(CONFIG_FILE) --debug-test experiment \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG)) \
		$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE))

status:
	@echo "$(BOLD)Project Status:$(RESET)"
	@echo "$(BOLD}===============${RESET}"
	@echo "Python: $(shell python --version)"
	@echo "Config: $(if $(wildcard $(CONFIG_FILE)),$(GREEN)✓ Found$(RESET),$(RED)✗ Missing$(RESET))"
	@echo "Data: $(if $(wildcard $(DATA_DIR)),$(GREEN)✓ Found$(RESET),$(RED)✗ Missing$(RESET))"
	@echo "Default model: $(if $(wildcard $(DEFAULT_MODEL)),$(GREEN)✓ Found$(RESET),$(RED)✗ Missing$(RESET))"
	@echo ""
	@echo "$(BOLD)Available models:$(RESET)"
	@find $(RESULTS_DIR)/checkpoints -name "*.pt" -type f 2>/dev/null | sed 's/^/  /' || echo "  No models found"

clean:
	@echo "$(BOLD)Cleaning up...$(RESET)"
	rm -rf $(LOGS_DIR)/* 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed$(RESET)"

format:
	@echo "$(BOLD)Formatting code...$(RESET)"
	black --config pyproject.toml $(PYTHON_SOURCE_DIRS)
	isort --settings-path=pyproject.toml $(PYTHON_SOURCE_DIRS)
	@echo "$(GREEN)Code formatted successfully$(RESET)"

lint:
	@echo "$(BOLD)Running linting...$(RESET)"
	flake8 --max-line-length=88 --extend-ignore=E203,W503 $(PYTHON_SOURCE_DIRS)
	@echo "$(GREEN)Linting completed$(RESET)"

check-performance:
	@echo "$(BOLD)Analyzing model performance...$(RESET)"
	pipenv run python scripts/check_model_performance.py
	@echo "$(GREEN)Performance analysis completed$(RESET)"

check-llm-performance:
	@echo "$(BOLD)Analyzing LLM evaluation performance...$(RESET)"
	pipenv run python scripts/check_llm_performance.py
	@echo "$(GREEN)LLM performance analysis completed$(RESET)"

check-all-performance:
	@echo "$(BOLD)Analyzing ALL model performance (Traditional + LLM)...$(RESET)"
	@echo "$(BLUE)Running traditional model analysis...$(RESET)"
	pipenv run python scripts/check_model_performance.py
	@echo ""
	@echo "$(BLUE)Running LLM evaluation analysis...$(RESET)"
	pipenv run python scripts/check_llm_performance.py
	@echo "$(GREEN)All performance analyses completed$(RESET)"

evaluate-llm:
	@echo "$(BOLD)Evaluating LLM models...$(RESET)"
	@mkdir -p $(RESULTS_DIR)/llm_evaluation
	pipenv run python llm/evaluate_llm.py \
		$(if $(MODEL),--model $(MODEL),--model llama3) \
		$(if $(PROMPT),--prompt $(PROMPT),--prompt zero_shot) \
		$(if $(LANG),--language-pair $(LANG),--language-pair en-de) \
		$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE)) \
		$(if $(DEVICE),--device $(DEVICE)) \
		$(if $(LOAD_8BIT),--load-in-8bit)
	@echo "$(GREEN)LLM evaluation completed$(RESET)"

evaluate-llm-all:
	@echo "$(BOLD)Evaluating ALL LLM models on ALL language pairs...$(RESET)"
	@mkdir -p $(RESULTS_DIR)/llm_evaluation
	@echo "$(BLUE)Starting comprehensive LLM evaluation...$(RESET)"
	@for lang in en-cs en-de en-ja en-zh; do \
		echo "$(BOLD)Evaluating language pair: $$lang$(RESET)"; \
		pipenv run python llm/evaluate_llm.py \
			--model all \
			--prompt all \
			--language-pair $$lang \
			$(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE)) \
			$(if $(DEVICE),--device $(DEVICE)) \
			$(if $(LOAD_8BIT),--load-in-8bit) || \
		{ echo "$(RED)Failed to evaluate $$lang, continuing...$(RESET)"; }; \
	done
	@echo "$(GREEN)All LLM evaluations completed$(RESET)"

debug-llm:
	@echo "$(BOLD)Debug LLM evaluation (small sample)...$(RESET)"
	@mkdir -p $(RESULTS_DIR)/llm_evaluation
	pipenv run python llm/evaluate_llm.py \
		--model llama3 \
		--prompt zero_shot \
		$(if $(LANG),--language-pair $(LANG),--language-pair en-de) \
		--sample-size 10 \
		--device auto
	@echo "$(GREEN)Debug LLM evaluation completed$(RESET)"

all: clean format lint