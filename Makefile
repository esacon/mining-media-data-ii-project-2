DATA_DIR = data/catastrophic_errors
RESULTS_DIR = results
LOGS_DIR = logs
CONFIG_FILE = config.yaml
DEFAULT_MODEL = $(RESULTS_DIR)/checkpoints/best_model_epoch_3.pt
PYTHON_SOURCE_DIRS = src scripts

BLUE=\033[34m
GREEN=\033[32m
YELLOW=\033[33m
RED=\033[31m
RESET=\033[0m
BOLD=\033[1m

.PHONY: help format lint clean install check all typecheck upgrade autofix validate pre-commit quick fix ci train evaluate predict analyze train-en-de evaluate-en-de predict-en-de status

help:
	@echo "$(BOLD)WMT21 Task 3 - Critical Error Detection$(RESET)"
	@echo "$(BOLD)======================================"
	@echo ""
	@echo "$(BOLD)Core Commands:$(RESET)"
	@echo "  $(BLUE)train [LANG=en-de]     $(RESET)- Train model"
	@echo "  $(BLUE)evaluate [LANG=en-de]  $(RESET)- Evaluate model"
	@echo "  $(BLUE)predict [LANG=en-de]   $(RESET)- Generate predictions"
	@echo "  $(BLUE)analyze [LANG=en-de]   $(RESET)- Run data analysis"
	@echo ""
	@echo "$(BOLD)Development & Quality:$(RESET)"
	@echo "  $(BLUE)install$(RESET)        - Install dependencies and pre-commit hooks"
	@echo "  $(BLUE)format$(RESET)         - Format code with Black and isort"
	@echo "  $(BLUE)check$(RESET)          - Check code formatting without making changes"
	@echo "  $(BLUE)lint$(RESET)           - Run linter (flake8)"
	@echo "  $(BLUE)typecheck$(RESET)      - Run mypy type checking"
	@echo "  $(BLUE)upgrade$(RESET)        - Upgrade Python syntax with pyupgrade"
	@echo "  $(BLUE)autofix$(RESET)        - Remove unused imports and variables with autoflake"
	@echo "  $(BLUE)validate$(RESET)       - Validate file formats (YAML, TOML, JSON) and common issues"
	@echo "  $(BLUE)clean$(RESET)          - Clean up generated files and caches"
	@echo "  $(BLUE)status$(RESET)         - Show project status"
	@echo "  $(BLUE)all$(RESET)            - Run all quality checks (validate, upgrade, autofix, format, lint, typecheck)"
	@echo "  $(BLUE)pre-commit$(RESET)     - Run all pre-commit hooks directly"
	@echo "  $(BLUE)quick$(RESET)          - Quick checks (format + lint)"
	@echo "  $(BLUE)fix$(RESET)            - Apply all auto-fixes (upgrade, autofix, format)"
	@echo "  $(BLUE)ci$(RESET)             - Run CI-focused checks (check, lint, typecheck)"
	@echo ""
	@echo "$(BOLD)Examples:$(RESET)"
	@echo "  make train LANG=en-de"
	@echo "  make evaluate LANG=en-de"
	@echo "  make predict LANG=en-de MODEL=path/to/model.pt"
	@echo "  make format"
	@echo ""
	@echo "$(BOLD)Variables:$(RESET)"
	@echo "  LANG=<pair>    Language pair (en-de, en-ja, en-zh, en-cs)"
	@echo "  MODEL=<path>   Model path (default: $(DEFAULT_MODEL))"

install:
	@echo "$(BOLD)Installing dependencies...$(RESET)"
	pip install --upgrade pip > /dev/null
	pip install -r requirements.txt > /dev/null
	pip install -e . > /dev/null
	@if command -v pre-commit &>/dev/null; then \
		pre-commit install > /dev/null; \
		echo "$(GREEN)Pre-commit hooks installed$(RESET)"; \
	else \
		echo "$(YELLOW)pre-commit not found, skipping hook installation$(RESET)"; \
	fi
	@echo "$(GREEN)Dependencies installed successfully$(RESET)"

train:
	@echo "$(BOLD)Training model...$(RESET)"
	@mkdir -p $(RESULTS_DIR)/checkpoints $(LOGS_DIR)
	python scripts/run_pipeline.py --config $(CONFIG_FILE) train \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG))

evaluate:
	@echo "$(BOLD)Evaluating model...$(RESET)"
	@mkdir -p $(RESULTS_DIR)
	$(eval MODEL_PATH := $(or $(MODEL),$(DEFAULT_MODEL)))
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "$(RED)Error: Model not found at $(MODEL_PATH)$(RESET)"; \
		find $(RESULTS_DIR)/checkpoints -name "*.pt" -type f 2>/dev/null | sed 's/^/  /'; \
		exit 1; \
	fi
	python scripts/run_pipeline.py --config $(CONFIG_FILE) evaluate \
		--model "$(MODEL_PATH)" \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG))

predict:
	@echo "$(BOLD)Generating predictions...$(RESET)"
	$(eval MODEL_PATH := $(or $(MODEL),$(DEFAULT_MODEL)))
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "$(RED)Error: Model not found at $(MODEL_PATH)$(RESET)"; \
		exit 1; \
	fi
	python scripts/run_pipeline.py --config $(CONFIG_FILE) predict \
		--model "$(MODEL_PATH)" \
		--data $(DATA_DIR) \
		$(if $(LANG),--language-pair $(LANG))

analyze:
	@echo "$(BOLD)Running data analysis...$(RESET)"
	@mkdir -p $(RESULTS_DIR)
	python scripts/run_pipeline.py --config $(CONFIG_FILE) analyze \
		--data-dir $(DATA_DIR) \
		$(if $(LANG),--language-pairs $(LANG))

validate:
	@echo "$(BOLD)Validating files...$(RESET)"
	@echo "  • Trailing whitespace..."
	@find $(PYTHON_SOURCE_DIRS) -name "*.py" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" -o -name "*.toml" | grep -v ".txt" | xargs sed -i 's/[[:space:]]*$$//' 2>/dev/null || true
	@echo "  • End of file newlines..."
	@find $(PYTHON_SOURCE_DIRS) -name "*.py" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" -o -name "*.toml" | grep -v ".txt" | xargs -I {} sh -c 'tail -c1 "{}" | read -r _ || echo >> "{}"' 2>/dev/null || true
	@echo "  • YAML files..."
	@find . -name "*.yml" -o -name "*.yaml" | head -10 | xargs -I {} python -c "import yaml; yaml.safe_load(open('{}'))" 2>/dev/null || (echo "$(RED)YAML validation failed$(RESET)" && exit 1)
	@echo "  • TOML files..."
	@find . -name "*.toml" | head -10 | xargs -I {} python -c "import tomllib; tomllib.load(open('{}', 'rb'))" 2>/dev/null || (echo "$(RED)TOML validation failed$(RESET)" && exit 1)
	@echo "  • JSON files..."
	@find . -name "*.json" | head -10 | xargs -I {} python -c "import json; json.load(open('{}'))" 2>/dev/null || (echo "$(RED)JSON validation failed$(RESET)" && exit 1)
	@echo "  • Merge conflicts..."
	@! grep -r "<<<<<<< HEAD" $(PYTHON_SOURCE_DIRS) 2>/dev/null || (echo "$(RED)Merge conflict markers found$(RESET)" && exit 1)
	@echo "  • Debug statements..."
	@! grep -r "import pdb\|pdb.set_trace\|import ipdb\|ipdb.set_trace" $(PYTHON_SOURCE_DIRS) 2>/dev/null || (echo "$(RED)Debug statements found$(RESET)" && exit 1)
	@echo "$(GREEN)File validation completed$(RESET)"

upgrade:
	@echo "$(BOLD)Upgrading Python syntax...$(RESET)"
	@if python -c "import pyupgrade" &>/dev/null; then \
		pyupgrade --py39-plus $(PYTHON_SOURCE_DIRS) &>/dev/null; \
		echo "$(GREEN)Python syntax upgraded$(RESET)"; \
	else \
		echo "$(YELLOW)pyupgrade not available, skipping$(RESET)"; \
	fi

autofix:
	@echo "$(BOLD)Removing unused imports and variables...$(RESET)"
	@if python -c "import autoflake" &>/dev/null; then \
		autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --ignore-init-module-imports $(PYTHON_SOURCE_DIRS) -r &>/dev/null; \
		echo "$(GREEN)Unused imports and variables removed$(RESET)"; \
	else \
		echo "$(YELLOW)autoflake not available, skipping$(RESET)"; \
	fi

format:
	@echo "$(BOLD)Formatting code...$(RESET)"
	@echo "  • Running isort..."
	isort $(PYTHON_SOURCE_DIRS) &>/dev/null || true
	@echo "  • Running black..."
	black $(PYTHON_SOURCE_DIRS) &>/dev/null || true
	@echo "$(GREEN)Code formatting completed$(RESET)"

check:
	@echo "$(BOLD)Checking code formatting...$(RESET)"
	@echo "  • Checking isort..."
	isort $(PYTHON_SOURCE_DIRS) --check-only --diff &>/dev/null || (echo "$(RED)isort check failed$(RESET)" && exit 1)
	@echo "  • Checking black..."
	black $(PYTHON_SOURCE_DIRS) --check --diff &>/dev/null || (echo "$(RED)black check failed$(RESET)" && exit 1)
	@echo "$(GREEN)Code formatting check passed$(RESET)"

lint:
	@echo "$(BOLD)Running linter...$(RESET)"
	@if python -c "import flake8" &>/dev/null; then \
		flake8 $(PYTHON_SOURCE_DIRS) --max-line-length=88 --extend-ignore=E203,W503,I201 --exclude="__pycache__,.git,.tox,build,dist,*.egg-info" &>/dev/null || (echo "$(RED)Linting failed:$(RESET)" && flake8 $(PYTHON_SOURCE_DIRS) --max-line-length=88 --extend-ignore=E203,W503,I201 --exclude="__pycache__,.git,.tox,build,dist,*.egg-info" && exit 1); \
		echo "$(GREEN)Linting passed$(RESET)"; \
	else \
		echo "$(YELLOW)flake8 not available, skipping$(RESET)"; \
	fi

typecheck:
	@echo "$(BOLD)Running type checker...$(RESET)"
	@if python -c "import mypy" &>/dev/null; then \
		mypy $(PYTHON_SOURCE_DIRS) --ignore-missing-imports --no-strict-optional &>/dev/null || (echo "$(RED)Type checking failed$(RESET)" && exit 1); \
		echo "$(GREEN)Type checking passed$(RESET)"; \
	else \
		echo "$(YELLOW)mypy not available, skipping$(RESET)"; \
	fi

train-en-de:
	@$(MAKE) train LANG=en-de

evaluate-en-de:
	@$(MAKE) evaluate LANG=en-de

predict-en-de:
	@$(MAKE) predict LANG=en-de

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

all: validate upgrade autofix format lint typecheck
	@echo ""
	@echo "$(BOLD)$(GREEN)All quality checks completed successfully$(RESET)"

pre-commit:
	@echo "$(BOLD)Running pre-commit hooks...$(RESET)"
	@if command -v pre-commit &>/dev/null; then \
		pre-commit run --all-files || (echo "$(RED)Pre-commit checks failed$(RESET)" && exit 1); \
		echo "$(GREEN)Pre-commit checks passed$(RESET)"; \
	else \
		echo "$(YELLOW)pre-commit not found, skipping$(RESET)"; \
	fi

quick: format lint
	@echo "$(BOLD)$(GREEN)Quick checks completed$(RESET)"

fix: upgrade autofix format
	@echo "$(BOLD)$(GREEN)Auto-fixes applied$(RESET)"