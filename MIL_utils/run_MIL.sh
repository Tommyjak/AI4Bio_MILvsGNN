#!/bin/bash

set -e  # stop on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv_MIL"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"

echo "── Activating virtual environment ────────────────────────"
source "$VENV_DIR/bin/activate"

echo "── Installing requirements ────────────────────────────────"
pip install -r "$REQUIREMENTS"

echo "── Starting training pipeline ─────────────────────────────"
trap deactivate EXIT
python3 "$TRAIN_SCRIPT"

echo "── Training complete ──────────────────────────────────────"
echo "── Virtual environment deactivated ────────────────────────"