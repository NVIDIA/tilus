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

# Don't exit on error - we want to run all scripts even if some fail
set +e

# Initialize variables
failed_scripts=()
total_scripts=0
failed_count=0

# Function to run a Python script and capture its output
run_script() {
    local script="$1"
    local output

    # Run the script and capture both stdout and stderr
    if output=$(python "$script" 2>&1); then
        echo "✓ $script passed"
        return 0
    else
        local exit_code=$?
        echo "✗ $script failed with exit code $exit_code"
        echo "Error output:"
        echo "$output"
        echo "----------------------------------------"
        return 1
    fi
}

# Find and run all Python files in the examples directory
echo "Running all Python examples..."
echo "----------------------------------------"

# Find all Python files
pyfiles=$(find examples -type f -name "*.py" | sort)
echo "Found $(echo "$pyfiles" | wc -l) Python files"

# Loop through each file more traditionally
for script in $pyfiles; do
    ((total_scripts++))

    if ! run_script "$script"; then
        failed_scripts+=("$script")
        ((failed_count++))
        echo "Continuing to next script despite failure..."
    fi
    echo ""
done

# Print summary
echo "----------------------------------------"
echo "Summary:"
echo "Total scripts run: $total_scripts"
echo "Failed scripts: $failed_count"
echo "Passed scripts: $((total_scripts - failed_count))"

if [ $failed_count -gt 0 ]; then
    echo "Failed script details:"
    for script in "${failed_scripts[@]}"; do
        echo "- $script"
    done
    exit 1
else
    echo "All scripts passed successfully!"
    exit 0
fi
