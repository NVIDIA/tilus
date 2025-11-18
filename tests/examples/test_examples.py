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
"""
Tests for examples in the examples/ folder.

This test suite ensures that all examples run successfully and that no example
scripts are added without being explicitly listed for testing.
"""

from pathlib import Path
from typing import Optional

import pytest
from tilus.target import Target, get_current_target, nvgpu_sm80, nvgpu_sm100a

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


# Explicitly defined examples with their target requirements
# Format: (folder_name, script_name, required_target)
EXAMPLES = [
    # matmul examples (SM 8.0+)
    ("matmul", "matmul_v0.py", None),
    ("matmul", "matmul_v1.py", None),
    ("matmul", "matmul_v2.py", None),
    ("matmul", "matmul_v3.py", None),
    ("matmul", "matmul_v4.py", None),
    ("matmul", "matmul_v5.py", None),
    # attention examples (SM 8.0+)
    ("attention", "flash_attention_v1.py", nvgpu_sm80),
    ("attention", "flash_attention_v2.py", nvgpu_sm80),
    ("attention", "flash_attention_v3.py", nvgpu_sm80),
    # attention with kvcache examples (SM 8.0+)
    ("attention_with_kvcache", "attention_v1.py", nvgpu_sm80),
    # blackwell matmul examples (SM 10.0a)
    ("blackwell_matmul", "matmul_v0.py", nvgpu_sm100a),
    ("blackwell_matmul", "matmul_v1.py", nvgpu_sm100a),
    ("blackwell_matmul", "matmul_v2.py", nvgpu_sm100a),
    ("blackwell_matmul", "matmul_v3.py", nvgpu_sm100a),
    ("blackwell_matmul", "matmul_v4.py", nvgpu_sm100a),
    ("blackwell_matmul", "matmul_v5.py", nvgpu_sm100a),
    # quantization examples (SM 8.0+)
    ("quantization", "matmul_a16wx.py", nvgpu_sm80),
    # flash attention decode examples (SM 8.0+)
    ("flash_attention_decode", "main.py", nvgpu_sm80),
]

# Scripts that should be ignored (baseline implementations, utilities, etc.)
IGNORED_SCRIPTS = [
    # Internal implementations, not example entrypoints
    ("flash_attention_decode", "torch_kernel.py"),
    ("flash_attention_decode", "triton_kernel.py"),
    ("flash_attention_decode", "tilus_kernel.py"),
]


def should_skip_example(required_target: Optional[Target]) -> tuple[bool, str]:
    """
    Determine if an example should be skipped based on the required target.

    Args:
        required_target: The target required to run the example, or None if any target is supported.

    Returns:
        A tuple of (should_skip, reason), where should_skip is True if the example should be skipped.
    """
    if required_target is None:
        return False, ""

    try:
        current_target = get_current_target()
        if not current_target.supports(required_target):
            current_capability = current_target.properties.compute_capability
            return True, (
                f"Example requires architecture {required_target}, but current GPU capability is {current_capability}"
            )
        return False, ""
    except Exception as e:
        return True, f"Cannot determine current GPU capability: {e}"


@pytest.mark.parametrize("folder,script,required_target", EXAMPLES)
def test_example(folder: str, script: str, required_target: Optional[Target]):
    """Test that an example script runs successfully."""
    should_skip, skip_reason = should_skip_example(required_target)
    if should_skip:
        pytest.skip(skip_reason)

    script_path = PROJECT_ROOT / "examples" / folder / script

    # Verify the script exists
    assert script_path.exists(), f"Example script not found: {script_path}"

    # Run the script using subprocess to ensure clean environment
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    # Check if the script succeeded
    if result.returncode != 0:
        error_msg = f"Example {folder}/{script} failed with exit code {result.returncode}\n"
        error_msg += f"STDOUT:\n{result.stdout}\n"
        error_msg += f"STDERR:\n{result.stderr}"
        pytest.fail(error_msg)


def test_all_examples_are_listed():
    """
    Ensure that all Python scripts in the examples directory are either
    explicitly listed in EXAMPLES or IGNORED_SCRIPTS.

    This test prevents new examples from being added without being properly
    configured in the test suite.
    """
    examples_dir = PROJECT_ROOT / "examples"

    # Build sets of (folder, script) tuples for quick lookup
    listed_examples = {(folder, script) for folder, script, _ in EXAMPLES}
    ignored_scripts = set(IGNORED_SCRIPTS)

    # Find all Python files in examples directory
    found_scripts = []
    for folder_path in sorted(examples_dir.iterdir()):
        if not folder_path.is_dir():
            continue
        if folder_path.name in [".ruff_cache", "__pycache__"]:
            continue

        folder_name = folder_path.name
        for script_path in sorted(folder_path.glob("*.py")):
            script_name = script_path.name
            found_scripts.append((folder_name, script_name))

    # Check for unlisted scripts
    unlisted_scripts = []
    for folder, script in found_scripts:
        if (folder, script) not in listed_examples and (folder, script) not in ignored_scripts:
            unlisted_scripts.append(f"{folder}/{script}")

    # Report error if unlisted scripts are found
    if unlisted_scripts:
        error_msg = (
            "Found Python scripts in examples/ that are not listed in test_examples.py:\n"
            + "\n".join(f"  - {script}" for script in unlisted_scripts)
            + "\n\nPlease add them to either EXAMPLES or IGNORED_SCRIPTS in tests/examples/test_examples.py"
        )
        pytest.fail(error_msg)


def test_no_missing_examples():
    """
    Ensure that all examples listed in EXAMPLES actually exist.

    This test catches typos or outdated entries in the EXAMPLES list.
    """
    examples_dir = PROJECT_ROOT / "examples"
    missing_examples = []

    for folder, script, _ in EXAMPLES:
        script_path = examples_dir / folder / script
        if not script_path.exists():
            missing_examples.append(f"{folder}/{script}")

    if missing_examples:
        error_msg = (
            "Listed examples in test_examples.py that do not exist:\n"
            + "\n".join(f"  - {script}" for script in missing_examples)
            + "\n\nPlease remove them from EXAMPLES or create the missing files."
        )
        pytest.fail(error_msg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
