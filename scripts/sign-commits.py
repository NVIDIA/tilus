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
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CommitSignatureStatus:
    """Status of a commit's signatures."""

    hash: str
    has_signoff: bool
    has_gpg_signature: bool
    gpg_status: str  # "valid", "invalid", "missing", "error"
    gpg_details: str
    short_info: str


def run_git_command(args: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a git command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(["git"] + args, capture_output=capture_output, text=True, check=False)
        if capture_output:
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        else:
            return result.returncode, "", ""
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


def check_working_tree_clean() -> bool:
    """Check if the working tree is clean (no unstaged changes)."""
    exit_code, output, stderr = run_git_command(["status", "--porcelain"])
    if exit_code != 0:
        print(f"Error checking git status: {stderr}")
        return False

    return len(output.strip()) == 0


def prompt_for_clean_working_tree() -> None:
    """Check for unstaged changes and prompt user to clean them."""
    if check_working_tree_clean():
        return  # Working tree is clean, proceed

    print("\n⚠️  Working tree not clean")
    print("The following files have unstaged changes:")

    # Show the status
    exit_code, output, stderr = run_git_command(["status", "--porcelain"], capture_output=False)

    print("\n❌ Cannot proceed with rebase while there are unstaged changes.")
    print("\n🔧 TO FIX THIS, choose one of the following:")
    print("   1. Commit your changes:")
    print("      git add .")
    print('      git commit -m "Your commit message"')
    print("   2. Stash your changes:")
    print("      git stash")
    print("   3. Discard your changes (CAREFUL!):")
    print("      git checkout -- .")
    print("\n💡 Then run the script again.")

    sys.exit(1)


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


def check_gpg_signature(commit_hash: str) -> Tuple[bool, str, str]:
    """Check if a commit has a valid GPG signature.

    Returns:
        Tuple of (has_signature, status, details)
        status: "valid", "invalid", "missing", "error"
    """
    exit_code, output, stderr = run_git_command(["verify-commit", commit_hash])

    if exit_code == 0:
        return True, "valid", output
    elif "no signature found" in stderr.lower() or "bad signature" in output.lower():
        return False, "missing", stderr
    elif "bad signature" in stderr.lower() or "invalid" in stderr.lower():
        return True, "invalid", stderr
    else:
        return False, "error", f"Unknown error: {stderr}"


def get_comprehensive_commit_status(commit_hash: str) -> CommitSignatureStatus:
    """Get comprehensive signature status for a commit."""
    # Check sign-off
    has_signoff = check_commit_signature(commit_hash)

    # Check GPG signature
    has_gpg, gpg_status, gpg_details = check_gpg_signature(commit_hash)

    # Get commit info
    short_info = get_commit_info(commit_hash)

    return CommitSignatureStatus(
        hash=commit_hash,
        has_signoff=has_signoff,
        has_gpg_signature=has_gpg,
        gpg_status=gpg_status,
        gpg_details=gpg_details,
        short_info=short_info,
    )


def get_commit_info(commit_hash: str) -> str:
    """Get short commit info for display."""
    exit_code, info, stderr = run_git_command(["log", "--format=%h %s", "-n", "1", commit_hash])
    if exit_code != 0:
        return f"{commit_hash[:8]} (error getting info)"
    return info


def check_commits(base_commit: str) -> Dict[str, List[CommitSignatureStatus]]:
    """Check all commits between base and HEAD for signatures.

    Returns:
        Dict with keys: "all_good", "missing_signoff", "missing_gpg", "invalid_gpg", "missing_both"
    """
    commits = get_commit_range(base_commit)

    if not commits:
        print("No commits found between base and HEAD")
        return {"all_good": [], "missing_signoff": [], "missing_gpg": [], "invalid_gpg": [], "missing_both": []}

    print(f"Checking {len(commits)} commits for signatures...")

    # Categorize commits
    all_good = []
    missing_signoff = []
    missing_gpg = []
    invalid_gpg = []
    missing_both = []

    for commit in commits:
        status = get_comprehensive_commit_status(commit)

        # Categorize based on status
        if status.has_signoff and status.gpg_status == "valid":
            all_good.append(status)
        elif not status.has_signoff and status.gpg_status in ["missing", "error"]:
            missing_both.append(status)
        elif not status.has_signoff:
            missing_signoff.append(status)
        elif status.gpg_status in ["missing", "error"]:
            missing_gpg.append(status)
        elif status.gpg_status == "invalid":
            invalid_gpg.append(status)
        else:
            # Edge case - has signoff but something else is wrong with GPG
            if status.has_signoff:
                missing_gpg.append(status)
            else:
                missing_signoff.append(status)

    return {
        "all_good": all_good,
        "missing_signoff": missing_signoff,
        "missing_gpg": missing_gpg,
        "invalid_gpg": invalid_gpg,
        "missing_both": missing_both,
    }


def find_first_unsigned_commit(base_commit: str) -> str:
    """Find the first commit that needs signing, starting from the merge base.

    Returns the commit hash of the first commit that needs DCO sign-off,
    or the base_commit if all commits are properly signed.
    """
    commits = get_commit_range(base_commit)

    if not commits:
        return base_commit

    # Check commits in reverse order (oldest first) to find the first unsigned one
    commits.reverse()

    for commit in commits:
        if not check_commit_signature(commit):
            # Found the first commit that needs signing
            # Return the parent of this commit so we can rebase from there
            exit_code, parent, stderr = run_git_command(["rev-parse", f"{commit}^"])
            if exit_code == 0:
                return parent
            else:
                # If we can't get the parent (e.g., this is the first commit),
                # fall back to the base commit
                return base_commit

    # All commits are properly signed
    return base_commit


def print_signature_report(results: Dict[str, List[CommitSignatureStatus]]) -> None:
    """Print a comprehensive human-readable report of signature status."""
    total_commits = sum(len(commits) for commits in results.values())

    print(f"\n{'=' * 60}")
    print("Commit signature analysis report")
    print(f"{'=' * 60}")
    print(f"Total commits analyzed: {total_commits}")

    # Print good commits
    if results["all_good"]:
        print(f"\n✅ FULLY SIGNED COMMITS ({len(results['all_good'])}):")
        for status in results["all_good"]:
            print(f"   {status.short_info}")

    # Print issues with clear explanations and todos

    if results["missing_both"]:
        print(f"\n❌ MISSING BOTH SIGNATURES ({len(results['missing_both'])}):")
        for status in results["missing_both"]:
            print(f"   {status.short_info}")

    if results["missing_signoff"]:
        print(f"\n📝 MISSING DCO SIGN-OFF ({len(results['missing_signoff'])}):")
        for status in results["missing_signoff"]:
            print(f"   {status.short_info}")

    if results["missing_gpg"]:
        print(f"\n🔐 MISSING GPG SIGNATURES ({len(results['missing_gpg'])}):")
        for status in results["missing_gpg"]:
            print(f"   {status.short_info}")

    if results["invalid_gpg"]:
        print(f"\n🚫 INVALID GPG SIGNATURES ({len(results['invalid_gpg'])}):")
        print("   ⚠️  SECURITY WARNING: These commits have invalid or corrupted signatures")
        print("   📋 TODO: Investigate and re-sign these commits")
        for status in results["invalid_gpg"]:
            print(f"   {status.short_info}")
            print(f"      Error: {status.gpg_details}")
        print("\n   🔧 HOW TO FIX:")
        print("      1. Verify your GPG key is valid: gpg --list-secret-keys")
        print("      2. Re-sign the commits: git rebase --gpg-sign <base_commit>")
        print("      3. If key is compromised, revoke and create new key")

    # # Summary and recommendations
    # print(f"\n{'=' * 60}")
    # if not issues_found:
    #     print("🎉 All commits are properly signed with both DCO and GPG signatures.")
    # else:
    #     print("📊 Summary of issues:")
    #     if results["missing_both"]:
    #         print(f"   • {len(results['missing_both'])} commits missing BOTH signatures")
    #     if results["missing_signoff"]:
    #         print(f"   • {len(results['missing_signoff'])} commits missing DCO sign-off")
    #     if results["missing_gpg"]:
    #         print(f"   • {len(results['missing_gpg'])} commits missing GPG signatures")
    #     if results["invalid_gpg"]:
    #         print(f"   • {len(results['invalid_gpg'])} commits with INVALID GPG signatures")

    #     print("")
    #     print("   Run 'python scripts/sign-commits.py --fix' to fix.")

    print(f"{'=' * 60}")


def get_fix_summary(results: Dict[str, List[CommitSignatureStatus]]) -> str:
    """Generate a summary of what the fix operation will do."""
    commits_needing_signoff = len(results["missing_signoff"]) + len(results["missing_both"])

    if commits_needing_signoff == 0:
        return "✅ All commits already have DCO sign-off!"

    summary = f"📝 Will add DCO sign-off to {commits_needing_signoff} commits:\n"

    for status in results["missing_signoff"] + results["missing_both"]:
        summary += f"   • {status.short_info}\n"

    if results["missing_gpg"] or results["invalid_gpg"]:
        gpg_count = len(results["missing_gpg"]) + len(results["invalid_gpg"])
        summary += f"\n🔐 Note: {gpg_count} commits will still need GPG signatures after this fix.\n"
        summary += "    Use 'git rebase --gpg-sign <base_commit>' to add GPG signatures.\n"

    return summary


def fix_commit_signatures(base_commit: str) -> bool:
    """Use git rebase --signoff to sign all commits from the first unsigned commit."""
    # Find the optimal rebase point (first unsigned commit)
    optimal_base = find_first_unsigned_commit(base_commit)

    if optimal_base == base_commit:
        # Check if there are any commits that need signing
        commits = get_commit_range(base_commit)
        if commits:
            unsigned_count = sum(1 for commit in commits if not check_commit_signature(commit))
            if unsigned_count == 0:
                print("✅ All commits are already properly signed")
                return True

        print(f"Signing all commits between {base_commit[:8]} and HEAD...")
        rebase_target = base_commit
    else:
        # Get commit count for both ranges for comparison
        original_commits = get_commit_range(base_commit)
        optimized_commits = get_commit_range(optimal_base)

        print(f"🚀 Optimization: Instead of rebasing {len(original_commits)} commits from {base_commit[:8]},")
        print(f"   rebasing only {len(optimized_commits)} commits from {optimal_base[:8]} (first unsigned commit)")

        rebase_target = optimal_base

    # Run git rebase with --signoff
    print(f"Running: git rebase --signoff {rebase_target[:8]}")
    exit_code, stdout, stderr = run_git_command(["rebase", rebase_target, "--signoff"], capture_output=False)

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
            # Check mode: report signature status and exit with appropriate code
            results = check_commits(base_commit)

            # Print comprehensive report
            print_signature_report(results)

            # Exit with error if any issues found
            issues_found = any(len(commits) > 0 for key, commits in results.items() if key != "all_good")

            if issues_found:
                # print("\n❌ Issues found with commit signatures.")
                print()
                print("💡 Run the following command to fix:")
                print(f"      python {sys.argv[0]} --fix")
                sys.exit(1)
            else:
                print("\n✅ All commits have proper signatures!")
                sys.exit(0)

        elif args.fix:
            # Fix mode: only fix DCO sign-off (GPG signatures require separate handling)

            # Check for unstaged changes before proceeding
            print("Checking working tree status...")
            prompt_for_clean_working_tree()

            results = check_commits(base_commit)

            commits_needing_signoff = len(results["missing_signoff"]) + len(results["missing_both"])

            if commits_needing_signoff == 0:
                print("✅ All commits already have DCO sign-off!")

                # Check if only GPG issues remain
                gpg_issues = len(results["missing_gpg"]) + len(results["invalid_gpg"])
                if gpg_issues > 0:
                    print(f"\n🔐 Note: {gpg_issues} commits still need GPG signatures.")
                    print("    Use 'git rebase --gpg-sign <base_commit>' to add GPG signatures.")

                sys.exit(0)

            # Show what will be fixed
            print(get_fix_summary(results))

            # Show optimization info
            optimal_base = find_first_unsigned_commit(base_commit)
            if optimal_base != base_commit:
                original_commits = get_commit_range(base_commit)
                optimized_commits = get_commit_range(optimal_base)
                print("\n🚀 Optimization detected:")
                print(f"   • Original plan: rebase {len(original_commits)} commits from {base_commit[:8]}")
                print(f"   • Optimized plan: rebase {len(optimized_commits)} commits from {optimal_base[:8]}")
                print(f"   • Saving {len(original_commits) - len(optimized_commits)} unnecessary rebases!")

            # Ask for confirmation unless non-interactive mode
            if not args.non_interactive:
                try:
                    response = input(f"\nAdd DCO sign-off to {commits_needing_signoff} commits? [Y/n]: ")
                    if response.lower().strip() not in ["y", "yes", ""]:
                        print("Aborted by user")
                        sys.exit(1)
                except KeyboardInterrupt:
                    print("\nAborted by user")
                    sys.exit(1)
            else:
                print(f"\nRunning in non-interactive mode, proceeding to sign {commits_needing_signoff} commits...")

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
