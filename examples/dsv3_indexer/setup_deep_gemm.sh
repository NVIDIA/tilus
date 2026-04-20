#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Clone and install DeepGEMM into a chosen Python environment.
# Needed to enable the `vllm` column in benchmark.py (provides
# fp8_paged_mqa_logits + get_paged_mqa_logits_metadata behind
# vllm.utils.deep_gemm).
#
# Usage:
#   # Either activate your venv first so `python` points at it, e.g.
#   source /home/yaoyaod/repos/tilus/.venv/bin/activate
#   ./setup_deep_gemm.sh
#
#   # ...or point the script at the interpreter explicitly:
#   PYTHON=/home/yaoyaod/repos/tilus/.venv/bin/python ./setup_deep_gemm.sh
#
# What it does:
#   1. Clones (or updates) DeepGEMM into third_party/deep_gemm next to this
#      script, including its git submodules (cutlass, fmt).
#   2. Replicates DeepGEMM's install.sh but uses `$PYTHON -m pip` so the
#      wheel goes into the same env as the interpreter — avoids the PEP 668
#      "externally managed" block that hits when bare `pip` resolves to the
#      system pip.
#   3. Verifies from a neutral working directory so the source tree doesn't
#      shadow the installed wheel (otherwise `from . import _C` blows up).
#
# The third_party/deep_gemm checkout is not committed — add
# examples/dsv3_indexer/third_party/ to .gitignore before committing.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEEP_GEMM_DIR="$SCRIPT_DIR/third_party/deep_gemm"
REPO_URL="https://github.com/deepseek-ai/DeepGEMM.git"

PYTHON="${PYTHON:-python}"

if ! command -v "$PYTHON" &> /dev/null; then
    echo "error: '$PYTHON' not found on PATH. Activate a venv or pass PYTHON=..." >&2
    exit 1
fi

PYTHON_BIN="$(command -v "$PYTHON")"
echo "Using python: $PYTHON_BIN"
"$PYTHON" --version

# Must be a venv / conda env — otherwise we'd hit PEP 668.
IS_VENV=$("$PYTHON" -c '
import sys, os
in_venv = (sys.prefix != getattr(sys, "base_prefix", sys.prefix)) or bool(os.environ.get("CONDA_PREFIX"))
print("yes" if in_venv else "no")
')
if [ "$IS_VENV" != "yes" ]; then
    echo "error: '$PYTHON_BIN' is not a virtual environment." >&2
    echo "       Activate your venv (e.g. 'source .venv/bin/activate') or re-run with" >&2
    echo "       PYTHON=/path/to/venv/bin/python ./setup_deep_gemm.sh" >&2
    exit 1
fi

# Clone (recursive for cutlass + fmt submodules) or update in place.
if [ ! -d "$DEEP_GEMM_DIR/.git" ]; then
    echo "Cloning DeepGEMM into $DEEP_GEMM_DIR ..."
    mkdir -p "$(dirname "$DEEP_GEMM_DIR")"
    git clone --recursive "$REPO_URL" "$DEEP_GEMM_DIR"
else
    echo "Updating existing DeepGEMM checkout at $DEEP_GEMM_DIR ..."
    git -C "$DEEP_GEMM_DIR" fetch --tags
    git -C "$DEEP_GEMM_DIR" pull --ff-only
    git -C "$DEEP_GEMM_DIR" submodule update --init --recursive
fi

# Ensure pip is available in this interpreter. uv-created venvs skip pip by
# default; bootstrap it with ensurepip. If that also fails, fall back to
# `uv pip install` targeting the interpreter.
if ! "$PYTHON" -m pip --version > /dev/null 2>&1; then
    echo "pip missing in $PYTHON_BIN — bootstrapping via ensurepip ..."
    if ! "$PYTHON" -m ensurepip --default-pip > /dev/null 2>&1; then
        if command -v uv &> /dev/null; then
            echo "ensurepip unavailable; will use 'uv pip install --python $PYTHON_BIN' instead."
            USE_UV_PIP=1
        else
            echo "error: no pip and no uv found. Install one of them and retry." >&2
            exit 1
        fi
    fi
fi

# Build wheel + install. Replicates DeepGEMM's ./install.sh but uses
# `$PYTHON -m pip` so we hit the right interpreter's site-packages.
echo "Building wheel ..."
(
    cd "$DEEP_GEMM_DIR"
    rm -rf build dist ./*.egg-info
    "$PYTHON" setup.py bdist_wheel
    if [ "${USE_UV_PIP:-0}" = "1" ]; then
        uv pip install --python "$PYTHON_BIN" --force-reinstall --no-deps dist/*.whl
    else
        "$PYTHON" -m pip install --force-reinstall --no-deps dist/*.whl
    fi
)

# Verify from / so the source tree in $DEEP_GEMM_DIR doesn't shadow the
# installed package (Python prepends cwd to sys.path for `-c`).
echo ""
echo "Verifying installation:"
(
    cd /
    "$PYTHON" -c "import deep_gemm; print('deep_gemm:', deep_gemm.__file__)"
    if "$PYTHON" -c "import vllm" 2> /dev/null; then
        "$PYTHON" -c "from vllm.utils.deep_gemm import has_deep_gemm; print('vllm has_deep_gemm():', has_deep_gemm())"
    fi
)

echo "Done."
