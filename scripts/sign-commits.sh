#!/bin/bash
set -e

MAIN_BRANCH="main"

git fetch origin
BASE=$(git merge-base HEAD origin/$MAIN_BRANCH)

# Find unsigned commits
UNSIGNED_COMMITS=$(git rev-list $BASE..HEAD | while read commit; do
    if ! git verify-commit $commit &>/dev/null; then
        echo $commit
    fi
done)

if [ -z "$UNSIGNED_COMMITS" ]; then
    echo "No commits need GPG signing."
    exit 0
fi

# Sign each unsigned commit with GPG
echo "Signing the following commits with GPG:"
for commit in $UNSIGNED_COMMITS; do
    git rebase --onto $commit^ $commit --exec "git commit --amend -S --no-edit"
done

# Print signed commits
for commit in $UNSIGNED_COMMITS; do
    git log --format="* %h %s" -n 1 $commit
done
