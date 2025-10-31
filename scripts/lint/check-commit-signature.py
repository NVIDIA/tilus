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

import argparse
import subprocess
import sys
from typing import List, Tuple


def run_git_command(args: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a git command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(["git"] + args, capture_output=capture_output, text=True, check=False)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        print("Error: git command not found")
        sys.exit(1)


def get_merge_base(branch: str = "main") -> str:
    """Get the merge base between HEAD and origin/branch."""
    # Fetch from origin first
    print("Fetching from origin...")
    exit_code, _, stderr = run_git_command(["fetch", "origin"])
    if exit_code != 0:
        print(f"Error fetching from origin: {stderr}")
        sys.exit(1)

    # Get merge base
    exit_code, base_commit, stderr = run_git_command(["merge-base", "HEAD", f"origin/{branch}"])
    if exit_code != 0:
        print(f"Error finding merge base with origin/{branch}: {stderr}")
        sys.exit(1)

    return base_commit


def get_commit_range(base_commit: str) -> List[str]:
    """Get list of commit hashes between base_commit and HEAD."""
    exit_code, commit_list, stderr = run_git_command(["rev-list", f"{base_commit}..HEAD"])
    if exit_code != 0:
        print(f"Error getting commit range: {stderr}")
        sys.exit(1)

    return commit_list.split("\n") if commit_list else []


def check_commit_signature(commit_hash: str) -> bool:
    """Check if a commit has a Signed-off-by line."""
    exit_code, commit_message, stderr = run_git_command(["log", "--format=%B", "-n", "1", commit_hash])
    if exit_code != 0:
        print(f"Error getting commit message for {commit_hash}: {stderr}")
        return False

    # Check for "Signed-off-by:" line
    return "Signed-off-by:" in commit_message


def get_commit_info(commit_hash: str) -> str:
    """Get short commit info for display."""
    exit_code, info, stderr = run_git_command(["log", "--format=%h %s", "-n", "1", commit_hash])
    if exit_code != 0:
        return f"{commit_hash[:8]} (error getting info)"
    return info


def check_commits(base_commit: str) -> Tuple[List[str], List[str]]:
    """Check all commits between base and HEAD for signatures.

    Returns:
        Tuple of (signed_commits, unsigned_commits)
    """
    commits = get_commit_range(base_commit)

    if not commits:
        print("No commits found between base and HEAD")
        return [], []

    signed_commits = []
    unsigned_commits = []

    print(f"Checking {len(commits)} commits for signatures...")

    for commit in commits:
        if check_commit_signature(commit):
            signed_commits.append(commit)
        else:
            unsigned_commits.append(commit)

    return signed_commits, unsigned_commits


def fix_commit_signatures(base_commit: str) -> bool:
    """Use git rebase --signoff to sign all commits."""
    print(f"Signing all commits between {base_commit[:8]} and HEAD...")

    # Run git rebase with --signoff
    print("Running: git rebase --signoff")
    exit_code, stdout, stderr = run_git_command(["rebase", base_commit, "--signoff"], capture_output=False)

    if exit_code == 0:
        print("✅ Successfully signed all commits")
        return True
    else:
        print(f"❌ Error during rebase: {stderr}")
        print("You may need to resolve conflicts and continue the rebase manually")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check and fix commit signatures")
    parser.add_argument("--check", action="store_true", help="Only check if commits are signed, don't modify them")
    parser.add_argument("--fix", action="store_true", help="Sign all unsigned commits using rebase")
    parser.add_argument("--branch", default="main", help="Base branch to compare against (default: main)")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Don't prompt for confirmation when fixing (useful for CI)"
    )

    args = parser.parse_args()

    # Ensure only one of check or fix is specified
    if args.check and args.fix:
        parser.error("Only one of --check or --fix can be specified")

    # Default behavior is check if neither is specified
    if not args.check and not args.fix:
        args.check = True

    try:
        # Get the merge base
        base_commit = get_merge_base(args.branch)
        print(f"Base commit: {base_commit[:8]}")

        if args.check:
            # Check mode: report unsigned commits and exit with appropriate code
            signed_commits, unsigned_commits = check_commits(base_commit)

            if signed_commits:
                print(f"\n✅ Signed commits ({len(signed_commits)}):")
                for commit in signed_commits:
                    print(f"  {get_commit_info(commit)}")

            if unsigned_commits:
                print(f"\n❌ Unsigned commits ({len(unsigned_commits)}):")
                for commit in unsigned_commits:
                    print(f"  {get_commit_info(commit)}")
                print("\nRun with --fix to sign these commits:")
                print(f"  python {sys.argv[0]} --fix")
                sys.exit(1)
            else:
                print(f"\n✅ All {len(signed_commits)} commits are properly signed!")
                sys.exit(0)

        elif args.fix:
            # Fix mode: sign all commits
            signed_commits, unsigned_commits = check_commits(base_commit)

            if not unsigned_commits:
                print("✅ All commits are already signed!")
                sys.exit(0)

            print(f"\n📝 Found {len(unsigned_commits)} unsigned commits:")
            for commit in unsigned_commits:
                print(f"  {get_commit_info(commit)}")

            # Ask for confirmation unless non-interactive mode
            if not args.non_interactive:
                try:
                    response = input(f"\nSign these {len(unsigned_commits)} commits? [y/N]: ")
                    if response.lower() not in ["y", "yes"]:
                        print("Aborted by user")
                        sys.exit(1)
                except KeyboardInterrupt:
                    print("\nAborted by user")
                    sys.exit(1)
            else:
                print(f"\nRunning in non-interactive mode, proceeding to sign {len(unsigned_commits)} commits...")

            # Perform the fix
            if fix_commit_signatures(base_commit):
                sys.exit(0)
            else:
                sys.exit(1)

    except KeyboardInterrupt:
        print("\nAborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
