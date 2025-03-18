#!/bin/bash

# Exit script on first error
set -e

# Define color codes with bold formatting
BOLD_GREEN='\033[1;32m'
BOLD_RED='\033[1;31m'
NC='\033[0m' # No color/reset

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the target directory (relative to script location)
TARGET_DIR=$(realpath "$SCRIPT_DIR/../../../python")
TEST_DIR=$(realpath "$SCRIPT_DIR/../../../tests")

# Ensure the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${BOLD_RED}Error: Target directory '$TARGET_DIR' does not exist.${NC}"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo -e "${BOLD_RED}Error: Test directory '$TEST_DIR' does not exist.${NC}"
    exit 1
fi

# Run Ruff to check formatting and linting issues (without fixing)
echo "Running ruff to check linting..."
if ! ruff check "$TARGET_DIR" "$TEST_DIR"; then
    echo -e "${BOLD_RED}Ruff found issues. Please fix them.${NC}"
    exit 1
fi
echo -e "${BOLD_GREEN}✔ Ruff linting checks passed.${NC}"

echo "Running ruff to check formatting..."
if ! ruff format --check "$TARGET_DIR" "$TEST_DIR"; then
    echo -e "${BOLD_RED}Ruff formatting check failed. Please format the code properly.${NC}"
    exit 1
fi
echo -e "${BOLD_GREEN}✔ Ruff format checks passed.${NC}"

# Run Mypy for static type checking
echo "Running mypy for type checking..."
if ! mypy "$TARGET_DIR" "$TEST_DIR"; then
    echo -e "${BOLD_RED}mypy found type issues. Please fix them.${NC}"
    exit 1
fi
echo -e "${BOLD_GREEN}✔ mypy checks passed.${NC}"

echo -e "${BOLD_GREEN}🎉 All checks passed successfully.${NC}"
exit 0
