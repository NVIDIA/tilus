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
"""IR-level validation tests for atomic and scatter instructions.

These tests exercise the ``*.create(...)`` validators directly; they do not
require a GPU or compile any CUDA code.
"""

import pytest
from tilus.hidet.ir.dtypes import float32, int32
from tilus.ir.inst import InstructionError
from tilus.ir.instructions.cuda.atomic import (
    AtomicGlobalInst,
    AtomicScatterGlobalInst,
    AtomicScatterSharedInst,
    AtomicSharedInst,
)
from tilus.ir.instructions.generic import StoreGlobalScatterInst, StoreSharedScatterInst
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor


def _reg(shape=(8,), dtype=int32) -> RegisterTensor:
    return RegisterTensor.create(dtype=dtype, shape=shape)


def _smem(shape=(8,), dtype=int32) -> SharedTensor:
    return SharedTensor.create(dtype=dtype, shape=shape)


def _gmem(shape=(8,), dtype=int32) -> GlobalTensor:
    from tilus.ir.layout import global_row_major

    return GlobalTensor.create(dtype=dtype, layout=global_row_major(*shape))


# ---------------------------------------------------------------------------
# Element-wise atomic: validation
# ---------------------------------------------------------------------------


def test_atomic_shared_add_accepts_matched_shape_dtype():
    dst = _smem(shape=(8,))
    values = _reg(shape=(8,))
    inst = AtomicSharedInst.create(dst=dst, values=values, op="add")
    assert inst.op == "add"
    assert inst.sem == "relaxed"
    assert inst.scope == "cta"
    assert inst.inputs == (dst, values)


def test_atomic_shared_rejects_unknown_op():
    dst = _smem()
    values = _reg()
    with pytest.raises(InstructionError, match="atomic op must be one of"):
        AtomicSharedInst.create(dst=dst, values=values, op="bogus")


def test_atomic_shared_rejects_unknown_sem_and_scope():
    dst = _smem()
    values = _reg()
    with pytest.raises(InstructionError, match="atomic sem must be one of"):
        AtomicSharedInst.create(dst=dst, values=values, op="add", sem="strong")
    with pytest.raises(InstructionError, match="atomic scope must be one of"):
        AtomicSharedInst.create(dst=dst, values=values, op="add", scope="block")


def test_atomic_shared_rejects_shape_mismatch():
    dst = _smem(shape=(8,))
    values = _reg(shape=(16,))
    with pytest.raises(InstructionError, match="dst shape"):
        AtomicSharedInst.create(dst=dst, values=values, op="add")


def test_atomic_shared_rejects_dtype_mismatch():
    dst = _smem(dtype=int32)
    values = _reg(dtype=float32)
    with pytest.raises(InstructionError, match="dtype"):
        AtomicSharedInst.create(dst=dst, values=values, op="add")


def test_atomic_shared_cas_requires_compare():
    dst = _smem()
    values = _reg()
    with pytest.raises(InstructionError, match="requires a compare"):
        AtomicSharedInst.create(dst=dst, values=values, op="cas")


def test_atomic_shared_compare_rejected_for_non_cas():
    dst = _smem()
    values = _reg()
    compare = _reg()
    with pytest.raises(InstructionError, match="compare is only valid"):
        AtomicSharedInst.create(dst=dst, values=values, op="add", compare=compare)


def test_atomic_shared_cas_validates_compare_shape_dtype():
    dst = _smem(shape=(8,), dtype=int32)
    values = _reg(shape=(8,), dtype=int32)
    compare_bad_shape = _reg(shape=(4,), dtype=int32)
    with pytest.raises(InstructionError, match="compare must match"):
        AtomicSharedInst.create(dst=dst, values=values, op="cas", compare=compare_bad_shape)
    compare_bad_dtype = _reg(shape=(8,), dtype=float32)
    with pytest.raises(InstructionError, match="compare must match"):
        AtomicSharedInst.create(dst=dst, values=values, op="cas", compare=compare_bad_dtype)


def test_atomic_global_mirrors_shared_rules():
    dst = _gmem(shape=(16,))
    values = _reg(shape=(16,))
    inst = AtomicGlobalInst.create(dst=dst, values=values, op="max")
    assert inst.scope == "gpu"  # default differs from shared
    with pytest.raises(InstructionError, match="shape"):
        AtomicGlobalInst.create(dst=dst, values=_reg(shape=(8,)), op="add")


# ---------------------------------------------------------------------------
# Scatter atomic: validation
# ---------------------------------------------------------------------------


def test_atomic_shared_scatter_accepts_matched():
    dst = _smem(shape=(16, 8))
    indices = _reg(shape=(4, 8))
    values = _reg(shape=(4, 8))
    inst = AtomicScatterSharedInst.create(dst=dst, indices=indices, values=values, dim=0, op="add")
    assert inst.dim == 0


def test_atomic_shared_scatter_rejects_cas_exch():
    dst = _smem(shape=(16, 8))
    indices = _reg(shape=(4, 8))
    values = _reg(shape=(4, 8))
    with pytest.raises(InstructionError, match="atomic_shared_scatter op must be one of"):
        AtomicScatterSharedInst.create(dst=dst, indices=indices, values=values, dim=0, op="cas")
    with pytest.raises(InstructionError, match="atomic_shared_scatter op must be one of"):
        AtomicScatterSharedInst.create(dst=dst, indices=indices, values=values, dim=0, op="exch")


def test_atomic_shared_scatter_shape_rules():
    dst = _smem(shape=(16, 8))
    # non-dim axis must match exactly
    with pytest.raises(InstructionError, match="non-scatter axis"):
        AtomicScatterSharedInst.create(dst=dst, indices=_reg(shape=(4, 4)), values=_reg(shape=(4, 4)), dim=0, op="add")
    # indices.shape must equal values.shape
    with pytest.raises(InstructionError, match="indices.shape"):
        AtomicScatterSharedInst.create(dst=dst, indices=_reg(shape=(4, 8)), values=_reg(shape=(8, 8)), dim=0, op="add")
    # rank mismatch
    with pytest.raises(InstructionError, match="dst rank"):
        AtomicScatterSharedInst.create(dst=dst, indices=_reg(shape=(4,)), values=_reg(shape=(4,)), dim=0, op="add")
    # dim out of range
    with pytest.raises(InstructionError, match="dim 3 out of range"):
        AtomicScatterSharedInst.create(dst=dst, indices=_reg(shape=(4, 8)), values=_reg(shape=(4, 8)), dim=3, op="add")


def test_atomic_global_scatter_mirrors_rules():
    dst = _gmem(shape=(16, 8))
    indices = _reg(shape=(4, 8))
    values = _reg(shape=(4, 8))
    inst = AtomicScatterGlobalInst.create(dst=dst, indices=indices, values=values, dim=0, op="min")
    assert inst.op == "min"
    with pytest.raises(InstructionError, match="atomic_global_scatter op must be one of"):
        AtomicScatterGlobalInst.create(dst=dst, indices=indices, values=values, dim=0, op="cas")


# ---------------------------------------------------------------------------
# Non-atomic scatter stores: validation
# ---------------------------------------------------------------------------


def test_store_shared_scatter_shape_rules():
    dst = _smem(shape=(16, 8))
    inst = StoreSharedScatterInst.create(dst=dst, indices=_reg(shape=(4, 8)), values=_reg(shape=(4, 8)), dim=0)
    assert inst.dim == 0
    assert inst.output is None


def test_store_global_scatter_shape_rules():
    dst = _gmem(shape=(16, 8))
    inst = StoreGlobalScatterInst.create(dst=dst, indices=_reg(shape=(4, 8)), values=_reg(shape=(4, 8)), dim=0)
    assert inst.dim == 0
    with pytest.raises(InstructionError, match="indices.shape"):
        StoreGlobalScatterInst.create(dst=dst, indices=_reg(shape=(4, 8)), values=_reg(shape=(2, 8)), dim=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
