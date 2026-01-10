#!/bin/bash
# Sanity check script for the Binocular project training pipeline.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "================================================"
echo "🚀 Starting Sanity Checks for Training Pipeline"
echo "================================================"

# --- 0. Install the project in editable mode ---
echo ""
echo "--- Step 0: Installing the project in editable mode ---"
echo ""

uv pip install -e .

echo ""
echo "✅ Project installed."
echo ""

# --- 1. Overfit on a single batch ---
# This test verifies that the model can learn and achieve 100% accuracy on a
# small amount of data, which confirms that the training loop, loss function,
# and optimizer are working correctly.
echo ""
echo "--- Test 1: Overfitting a single batch ---"
echo "Expected outcome: Accuracy should reach > 0.99 within a few epochs."
echo ""

uv run python -m Binocular.scripts.train \
    --config Binocular/configs/m2_dev.yaml \
    --debug-overfit-batch

echo ""
echo "✅ Overfit test completed."
echo ""


# --- 2. Quick training test (3 epochs) ---
# This test runs the full training and validation pipeline for a few epochs to
# ensure all components work together without crashing. It checks data loading,
# training, validation, logging, and checkpointing.
echo "--- Test 2: Quick training run (3 epochs) ---"
echo "Expected outcome: The script should run to completion without errors."
echo ""

uv run python -m Binocular.scripts.train \
    --config Binocular/configs/m2_dev.yaml \
    --epochs 3

echo ""
echo "✅ Quick training run completed."
echo ""
echo "================================================"
echo "🎉 All sanity checks passed successfully!"
echo "================================================"
