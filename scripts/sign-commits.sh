#!/bin/bash
set -e

MAIN_BRANCH="main"

git fetch origin
BASE=$(git merge-base HEAD origin/$MAIN_BRANCH)

# Sign all unsigned commits in one rebase
git rebase $BASE --signoff
