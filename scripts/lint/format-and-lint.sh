#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
