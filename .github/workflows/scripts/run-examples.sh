#!/bin/bash

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
    echo "Running $script..."
    echo "Command: python \"$script\""
    
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

# Debug: Print current directory
echo "Current directory: $(pwd)"

# Debug: Print if examples directory exists
if [ -d "examples" ]; then
    echo "Examples directory exists"
else
    echo "Examples directory does not exist!"
    exit 1
fi

# Find all Python files
pyfiles=$(find examples -type f -name "*.py")
echo "Found $(echo "$pyfiles" | wc -l) Python files"

# Loop through each file more traditionally
for script in $pyfiles; do
    echo "Found: $script"
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



