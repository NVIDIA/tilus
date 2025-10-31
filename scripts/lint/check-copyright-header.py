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
import os
import sys
from datetime import datetime

current_year = datetime.now().year

LICENSE_HEADER = """
# SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
""".format(year=current_year)

SHORT_LICENSE_HEADER = """
# SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
""".format(year=current_year)

LICENSE_LINE = "# SPDX-License-Identifier: Apache-2.0\n"
TARGET_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "python"),
    os.path.join(os.path.dirname(__file__), "..", "..", "tests"),
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts"),
]
SHORT_LICENSE_TARGET_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "examples"),
]


def add_license_to_file(filepath, filetype, use_short_header=False):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Choose which header to use
    header_to_use = SHORT_LICENSE_HEADER if use_short_header else LICENSE_HEADER

    # Find where to insert (after shebang if present, else at top)
    insert_idx = 1 if lines and lines[0].startswith("#!") else 0

    # Check first 15 lines for any existing license header
    search_lines = lines[:15]

    # Check if any header already exists (with SPDX-FileCopyrightText)
    has_any_header = any("SPDX-FileCopyrightText" in line for line in search_lines)
    if has_any_header:
        return False  # Header already exists, don't modify

    # Check if only the short license line exists
    has_license_line = any(LICENSE_LINE.strip() in line for line in search_lines)

    if has_license_line:
        # Replace the short license line with the appropriate header
        # Find and remove the existing short license line
        for i, line in enumerate(lines):
            if LICENSE_LINE.strip() in line:
                lines.pop(i)
                break
        # Insert the appropriate header at the right position
        header_lines = header_to_use.split("\n")
        for i, header_line in enumerate(reversed(header_lines)):
            if header_line:  # Skip empty lines at the end
                lines.insert(insert_idx, header_line + "\n")
    else:
        # No license header exists, add the appropriate header
        header_lines = header_to_use.split("\n")
        for i, header_line in enumerate(reversed(header_lines)):
            if header_line:  # Skip empty lines at the end
                lines.insert(insert_idx, header_line + "\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True


def process_directory(target_dirs, use_short_header=False, check_only=False):
    total_files = 0
    need_update = 0
    updated = 0
    files_needing_update = []

    for target_dir in target_dirs:
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(".py") or file.endswith(".sh"):
                    total_files += 1
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # Check first 15 lines for any existing license header
                    search_lines = lines[:15]
                    has_any_header = any("SPDX-FileCopyrightText" in line for line in search_lines)

                    # File needs update if it has no header
                    if not has_any_header:
                        need_update += 1
                        files_needing_update.append(filepath)
                        if not check_only:
                            if add_license_to_file(
                                filepath, filetype=file.split(".")[-1], use_short_header=use_short_header
                            ):
                                updated += 1

    return total_files, need_update, updated, files_needing_update


def main():
    parser = argparse.ArgumentParser(description="Check and fix copyright headers in source files")
    parser.add_argument("--check", action="store_true",
                       help="Only check if files have headers, don't modify them")
    parser.add_argument("--fix", action="store_true",
                       help="Automatically add missing headers to files")

    args = parser.parse_args()

    # Default behavior is fix if neither check nor fix is specified
    if not args.check and not args.fix:
        args.fix = True

    check_only = args.check    # Process directories with full license header
    full_total, full_need_update, full_updated, full_files_needing_update = process_directory(
        TARGET_DIRS, use_short_header=False, check_only=check_only)

    # Process directories with short license header
    short_total, short_need_update, short_updated, short_files_needing_update = process_directory(
        SHORT_LICENSE_TARGET_DIRS, use_short_header=True, check_only=check_only)

    # Combined totals
    total_files = full_total + short_total
    need_update = full_need_update + short_need_update
    updated = full_updated + short_updated
    all_files_needing_update = full_files_needing_update + short_files_needing_update

    if check_only:
        if need_update > 0:
            print(f"Found {need_update} files missing copyright headers:")
            for filepath in all_files_needing_update:
                print(f"  {filepath}")
            sys.exit(1)
        else:
            print(f"All {total_files} source files have copyright headers.")
            sys.exit(0)
    else:
        print(f"Total source files found (.py/.sh): {total_files}")
        print(f"Files with full header: {full_total} (need update: {full_need_update}, updated: {full_updated})")
        print(f"Files with short header: {short_total} (need update: {short_need_update}, updated: {short_updated})")
        print(f"Total files needing update: {need_update}")
        print(f"Total files updated: {updated}")

        if need_update > 0 and updated == 0:
            sys.exit(1)
        sys.exit(0)
if __name__ == "__main__":
    main()
