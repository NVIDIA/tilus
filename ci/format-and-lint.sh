#!/bin/bash

# Exit script on first error
set -e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the target directory (relative to script location)
TARGET_DIR=$(realpath "$SCRIPT_DIR/../python")
TEST_DIR=$(realpath "$SCRIPT_DIR/../tests")

# Ensure the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Run Ruff to format and fix issues
echo "Running ruff to clean code..."
ruff check --fix "$TARGET_DIR" "$TEST_DIR"
ruff format "$TARGET_DIR" "$TEST_DIR"

echo "Ruff completed successfully."

# Run Mypy for static type checking
echo "Running mypy for type checking..."
mypy "$TARGET_DIR" "$TEST_DIR"

echo "mypy completed successfully."

echo "Code cleaning process finished!"
