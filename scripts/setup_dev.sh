#!/bin/bash
# This script sets up the development environment for the critical error detection project.

set -e

echo "🚀 Setting up development environment..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "📦 Creating venv..."
    python3 -m venv venv
fi

echo "🔧 Activating venv..."
source venv/bin/activate

echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "🪝 Installing pre-commit hooks..."
pre-commit install

echo "🎨 Running initial code formatting..."
make format

echo "🔍 Running linting checks..."
make lint

echo "✅ Development environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Available commands:"
echo "  make help     - Show all available commands"
echo "  make format   - Format code with black and isort"
echo "  make lint     - Run flake8 linter"
echo "  make all      - Run format, lint"
