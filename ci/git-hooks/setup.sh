#!/bin/bash
#
# This script sets up the git hooks for the project.
#

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# move the pre-commit and pre-push in current directory to the ../../.git/hooks folder
cp pre-commit ../../.git/hooks
cp pre-push ../../.git/hooks
