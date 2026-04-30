# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""IR-level validation tests for ``ScanInst``.

Exercises the ``ScanInst.create(...)`` validators directly; no GPU / CUDA
compilation required.
"""

import pytest
from tilus.hidet.ir.dtypes import float32, int32
from tilus.ir.inst import InstructionError
from tilus.ir.instructions.generic import SCAN_BITWISE_OPS, SCAN_OPS, ScanInst
from tilus.ir.tensor import RegisterTensor


def _reg(shape=(8,), dtype=int32) -> RegisterTensor:
    return RegisterTensor.create(dtype=dtype, shape=shape)


# ---------------------------------------------------------------------------
# Accepts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", SCAN_OPS)
def test_scan_accepts_all_ops_on_int_dtype(op):
    x = _reg(shape=(8,), dtype=int32)
    y = _reg(shape=(8,), dtype=int32)
    inst = ScanInst.create(x=x, output=y, dim=0, op=op, exclusive=False)
    assert inst.op == op
    assert inst.dim == 0
    assert inst.exclusive is False
    assert inst.inputs == (x,)


def test_scan_accepts_float_for_non_bitwise_ops():
    x = _reg(shape=(8,), dtype=float32)
    y = _reg(shape=(8,), dtype=float32)
    for op in ("add", "mul", "max", "min"):
        ScanInst.create(x=x, output=y, dim=0, op=op, exclusive=False)


def test_scan_accepts_exclusive_flag():
    x = _reg(shape=(8,))
    y = _reg(shape=(8,))
    inst = ScanInst.create(x=x, output=y, dim=0, op="add", exclusive=True)
    assert inst.exclusive is True


# ---------------------------------------------------------------------------
# Rejects
# ---------------------------------------------------------------------------


def test_scan_rejects_unknown_op():
    x = _reg()
    y = _reg()
    with pytest.raises(InstructionError, match="scan op must be one of"):
        ScanInst.create(x=x, output=y, dim=0, op="bogus", exclusive=False)


def test_scan_rejects_out_of_range_dim():
    x = _reg(shape=(8,))
    y = _reg(shape=(8,))
    with pytest.raises(InstructionError, match="scan dim 1 out of range"):
        ScanInst.create(x=x, output=y, dim=1, op="add", exclusive=False)
    with pytest.raises(InstructionError, match="scan dim -1 out of range"):
        ScanInst.create(x=x, output=y, dim=-1, op="add", exclusive=False)


def test_scan_rejects_shape_mismatch():
    x = _reg(shape=(8,))
    y = _reg(shape=(16,))
    with pytest.raises(InstructionError, match="shape"):
        ScanInst.create(x=x, output=y, dim=0, op="add", exclusive=False)


def test_scan_rejects_dtype_mismatch():
    x = _reg(shape=(8,), dtype=int32)
    y = _reg(shape=(8,), dtype=float32)
    with pytest.raises(InstructionError, match="dtype"):
        ScanInst.create(x=x, output=y, dim=0, op="add", exclusive=False)


@pytest.mark.parametrize("op", SCAN_BITWISE_OPS)
def test_scan_rejects_bitwise_on_float(op):
    x = _reg(shape=(8,), dtype=float32)
    y = _reg(shape=(8,), dtype=float32)
    with pytest.raises(InstructionError, match="integer dtype"):
        ScanInst.create(x=x, output=y, dim=0, op=op, exclusive=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
