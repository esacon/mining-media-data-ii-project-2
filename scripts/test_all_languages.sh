#!/bin/bash

# Test All Languages - Critical Error Detection
# Run this script to train and evaluate all language pairs

set -e  # Exit on any error

LANGUAGES=("en-de" "en-ja" "en-zh" "en-cs")
BLUE='\033[34m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'
BOLD='\033[1m'

echo -e "${BOLD}Critical Error Detection - All Languages Pipeline${RESET}"
echo -e "${BOLD}=================================================${RESET}"
echo ""

# Function to run command with language
run_for_lang() {
    local cmd=$1
    local lang=$2
    echo -e "${BLUE}Running: ${cmd} for ${lang}${RESET}"
    if make $cmd LANG=$lang; then
        echo -e "${GREEN}✓ ${cmd} completed for ${lang}${RESET}"
    else
        echo -e "${RED}✗ ${cmd} failed for ${lang}${RESET}"
        return 1
    fi
    echo ""
}

# Quick debug test for all languages
if [[ "$1" == "debug" ]]; then
    echo -e "${YELLOW}=== DEBUG MODE: Quick test all languages ===${RESET}"
    for lang in "${LANGUAGES[@]}"; do
        echo -e "${BOLD}Testing ${lang}...${RESET}"
        run_for_lang "debug-experiment" $lang
    done
    echo -e "${GREEN}All debug tests completed!${RESET}"
    exit 0
fi

# Full training pipeline
if [[ "$1" == "train-all" ]]; then
    echo -e "${YELLOW}=== TRAINING ALL LANGUAGES ===${RESET}"
    for lang in "${LANGUAGES[@]}"; do
        echo -e "${BOLD}Training ${lang}...${RESET}"
        run_for_lang "train" $lang
    done
    echo -e "${GREEN}All training completed!${RESET}"
    exit 0
fi

# Full evaluation pipeline
if [[ "$1" == "evaluate-all" ]]; then
    echo -e "${YELLOW}=== EVALUATING ALL LANGUAGES ===${RESET}"
    for lang in "${LANGUAGES[@]}"; do
        echo -e "${BOLD}Evaluating ${lang}...${RESET}"
        run_for_lang "evaluate" $lang
    done
    echo -e "${GREEN}All evaluations completed!${RESET}"
    exit 0
fi

# Generate predictions for all
if [[ "$1" == "predict-all" ]]; then
    echo -e "${YELLOW}=== GENERATING PREDICTIONS FOR ALL LANGUAGES ===${RESET}"
    for lang in "${LANGUAGES[@]}"; do
        echo -e "${BOLD}Predicting ${lang}...${RESET}"
        run_for_lang "predict" $lang
    done
    echo -e "${GREEN}All predictions completed!${RESET}"
    exit 0
fi

# Full pipeline - everything
if [[ "$1" == "full" ]]; then
    echo -e "${YELLOW}=== FULL PIPELINE: TRAIN + EVALUATE + PREDICT ===${RESET}"
    for lang in "${LANGUAGES[@]}"; do
        echo -e "${BOLD}Full pipeline for ${lang}...${RESET}"
        run_for_lang "train" $lang
        run_for_lang "evaluate" $lang
        run_for_lang "predict" $lang
        echo -e "${GREEN}✓ ${lang} pipeline completed${RESET}"
        echo "----------------------------------------"
    done
    echo -e "${GREEN}Full pipeline completed for all languages!${RESET}"
    exit 0
fi

# Show help
echo -e "${BOLD}Usage:${RESET}"
echo "  $0 debug        - Quick debug test all languages (fast)"
echo "  $0 train-all    - Train models for all languages"
echo "  $0 evaluate-all - Evaluate all trained models"
echo "  $0 predict-all  - Generate predictions for all languages"
echo "  $0 full         - Complete pipeline (train + evaluate + predict)"
echo ""
echo -e "${BOLD}Examples:${RESET}"
echo "  $0 debug                    # Test everything works (5-10 minutes)"
echo "  $0 train-all                # Train all models (several hours)"
echo "  $0 full                     # Complete end-to-end pipeline"
echo ""
echo -e "${BOLD}Individual commands:${RESET}"
echo "  make train LANG=en-de       # Train single language"
echo "  make status                 # Check project status"
echo "  make help                   # Show all available commands" 