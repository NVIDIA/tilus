#!/bin/bash

# Exit script on first error
set -e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the target directory (relative to script location)
TARGET_DIR=$(realpath "$SCRIPT_DIR/../python")

# Ensure the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Run Ruff to format and fix issues
echo "Running ruff to clean code in $TARGET_DIR..."
ruff check --fix "$TARGET_DIR"
ruff format "$TARGET_DIR"

echo "Ruff completed successfully."

# Run Mypy for static type checking
echo "Running mypy for type checking in $TARGET_DIR..."
mypy "$TARGET_DIR"

echo "mypy completed successfully."

echo "Code cleaning process finished!"
