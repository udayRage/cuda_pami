#!/bin/bash

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <algorithm_type> <algorithm_name>"
    echo "Example for multi-word type: bash scripts/create_experiment.sh 'periodic frequent' MyPeriodicMiner"
    echo "Example for single-word type: bash scripts/create_experiment.sh fuzzy MyFuzzyMiner"
    exit 1
fi

# The last argument is the algorithm name
ALGORITHM_NAME="${@: -1}"

# All arguments except the last one form the algorithm type
ALGORITHM_TYPE="${@:1:$#-1}"

# Define base paths
RESULTS_DIR="results"
SRC_DIR="src/algorithms"
NOTEBOOKS_DIR="notebooks"

# Create directories for the algorithm type
ALGO_TYPE_SRC_DIR="$SRC_DIR/$ALGORITHM_TYPE"
ALGO_TYPE_NOTEBOOKS_DIR="$NOTEBOOKS_DIR/$ALGORITHM_TYPE"
ALGO_TYPE_RESULTS_DIR="$RESULTS_DIR/$ALGORITHM_TYPE/$ALGORITHM_NAME"

if [ ! -d "$ALGO_TYPE_SRC_DIR" ]; then
    echo "Algorithm type '$ALGORITHM_TYPE' does not exist. Creating new type."
fi

echo "Creating experiment structure for algorithm '$ALGORITHM_NAME' of type '$ALGORITHM_TYPE'..."

mkdir -p "$ALGO_TYPE_RESULTS_DIR"
echo "Created results directory: $ALGO_TYPE_RESULTS_DIR"

mkdir -p "$ALGO_TYPE_SRC_DIR"
touch "$ALGO_TYPE_SRC_DIR/${ALGORITHM_NAME}.py"
echo "Created algorithm file: $ALGO_TYPE_SRC_DIR/${ALGORITHM_NAME}.py"

mkdir -p "$ALGO_TYPE_NOTEBOOKS_DIR"
touch "$ALGO_TYPE_NOTEBOOKS_DIR/${ALGORITHM_NAME}.ipynb"
echo "Created notebook: $ALGO_TYPE_NOTEBOOKS_DIR/${ALGORITHM_NAME}.ipynb"

echo "Experiment structure for '$ALGORITHM_NAME' created successfully."
