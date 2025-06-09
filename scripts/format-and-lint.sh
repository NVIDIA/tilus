#!/bin/bash

# Exit script on first error
set -e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the target directory (relative to script location)
TARGET_DIR=$(realpath "$SCRIPT_DIR/../python")
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/../examples")
TEST_DIR=$(realpath "$SCRIPT_DIR/../tests")
SCRIPTS_DIR=$(realpath "$SCRIPT_DIR")

# Run Ruff to format and fix issues
echo "Running ruff to clean code..."
ruff check --fix "$TARGET_DIR" "$TEST_DIR" "$EXAMPLE_DIR" "$SCRIPTS_DIR"
ruff format "$TARGET_DIR" "$TEST_DIR" "$EXAMPLE_DIR" "$SCRIPTS_DIR"

echo "Ruff completed successfully."

# Run Mypy for static type checking
echo "Running mypy for type checking..."
mypy "$TARGET_DIR"

echo "mypy completed successfully."

echo "Code cleaning process finished!"
