#!/bin/bash
# This script signs off all unsigned commits in the current branch that are not in the main branch,
# and shows the newly signed commits.

set -e

MAIN_BRANCH="main"

git fetch origin

BASE=$(git merge-base HEAD origin/$MAIN_BRANCH)

# List commits after the common ancestor missing "Signed-off-by"
UNSIGNED_COMMITS=$(git rev-list $BASE..HEAD | while read commit; do
    if ! git show --quiet --format=%B $commit | grep -q "Signed-off-by:"; then
        echo $commit
    fi
done)

if [ -z "$UNSIGNED_COMMITS" ]; then
    echo "No unsigned commits to sign off."
    exit 0
fi

# Rebase with signoff
git rebase --signoff $BASE

echo "Newly signed commits:"
for commit in $UNSIGNED_COMMITS; do
    git log --format="* %h %s" -n 1 $commit
done