#!/usr/bin/env bash
# run_feature_engineer_pipeline.sh
# This script runs the feature engineering pipeline in the following order:
# 1. feature_engineering.py
# 2. feature_refinement.py
# 3. dynamic_feature_selection.py
#
# Ensure you run this script from the project's root directory
# Usage: bash src/scripts/run_feature_engineer_pipeline.sh

echo "=== Starting feature engineering pipeline... ===" 

echo "Step 1: Running feature_engineering.py..."
python src/feature_engineering.py
if [ $? -ne 0 ]; then
    echo "Error encountered in feature_engineering.py. Exiting."
    exit 1
fi

echo "Step 2: Running feature_refinement.py..."
python src/feature_refinement.py
if [ $? -ne 0 ]; then
    echo "Error encountered in feature_refinement.py. Exiting."
    exit 1
fi

echo "Step 3: Running dynamic_feature_selection.py..."
python src/dynamic_feature_selection.py
if [ $? -ne 0 ]; then
    echo "Error encountered in dynamic_feature_selection.py. Exiting."
    exit 1
fi

echo "=== Feature engineering pipeline completed successfully! ==="
