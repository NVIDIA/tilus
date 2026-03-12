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
"""Tests for the dead code elimination pass."""

import pytest
from hidet.ir.dtypes import float16, float32
from hidet.ir.expr import Var, as_expr
from hidet.ir.primitives.cuda.vars import blockIdx
from hidet.ir.type import PointerType
from tilus.ir.func import Function, Metadata
from tilus.ir.instructions.generic import (
    AddInst,
    AllocateRegisterInst,
    CastInst,
    MulInst,
    StoreGlobalGenericInst,
    SyncThreadsInst,
)
from tilus.ir.prog import Program
from tilus.ir.stmt import InstStmt, SeqStmt
from tilus.ir.tensor import RegisterTensor
from tilus.ir.tools.instruction_collector import collect_instructions
from tilus.transforms.dead_code_elimination import dead_code_elimination_pass


def _make_function(insts) -> Function:
    """Create a minimal Function wrapping a sequence of instructions."""
    body = SeqStmt(tuple(InstStmt(inst) for inst in insts))
    p = Var("p", PointerType(float32))
    return Function.create(
        name="test",
        params=[p],
        body=body,
        metadata=Metadata.create(
            grid_blocks=[as_expr(1), as_expr(1), as_expr(1)],
            cluster_blocks=[1, 1, 1],
            block_indices=[blockIdx.x, blockIdx.y, blockIdx.z],  # type: ignore[attr-defined]
            num_warps=1,
        ),
    )


def _make_program(insts) -> Program:
    func = _make_function(insts)
    return Program.create({func.name: func})


def _count_insts(program: Program, inst_type) -> int:
    func = list(program.functions.values())[0]
    insts = collect_instructions(func)
    return sum(1 for inst in insts if isinstance(inst, inst_type))


def _alloc(shape=(4,), dtype=float32) -> tuple[AllocateRegisterInst, RegisterTensor]:
    """Create an AllocateRegisterInst and return (inst, output_tensor)."""
    output = RegisterTensor.create(dtype=dtype, shape=shape)
    inst = AllocateRegisterInst.create(output=output, f_init=lambda _: dtype.zero)
    return inst, output


def _store(src: RegisterTensor) -> StoreGlobalGenericInst:
    """Create a StoreGlobalGenericInst that consumes the given register tensor."""
    p = Var("p", PointerType(float32))
    return StoreGlobalGenericInst.create(x=src, ptr=p, f_offset=lambda _: 0)


def test_eliminate_unused_add():
    """An AddInst whose output is never consumed should be eliminated."""
    alloc_a, a = _alloc()
    alloc_b, b = _alloc()
    out_add = RegisterTensor.create(dtype=float32, shape=(4,))
    add_inst = AddInst.create(a, b, out_add)
    store = _store(a)  # only 'a' is consumed, not add output

    prog = _make_program([alloc_a, alloc_b, add_inst, store])
    assert _count_insts(prog, AddInst) == 1

    opt_prog = dead_code_elimination_pass()(prog)
    assert _count_insts(opt_prog, AddInst) == 0
    assert _count_insts(opt_prog, StoreGlobalGenericInst) == 1


def test_preserve_used_add():
    """An AddInst whose output is consumed by a store should NOT be eliminated."""
    alloc_a, a = _alloc()
    alloc_b, b = _alloc()
    out_add = RegisterTensor.create(dtype=float32, shape=(4,))
    add_inst = AddInst.create(a, b, out_add)
    store = _store(out_add)

    prog = _make_program([alloc_a, alloc_b, add_inst, store])
    opt_prog = dead_code_elimination_pass()(prog)

    assert _count_insts(opt_prog, AddInst) == 1
    assert _count_insts(opt_prog, StoreGlobalGenericInst) == 1


def test_chain_elimination():
    """A chain of functional instructions where the final result is unused should all be eliminated."""
    alloc_a, a = _alloc()
    alloc_b, b = _alloc()
    out_add = RegisterTensor.create(dtype=float32, shape=(4,))
    add_inst = AddInst.create(a, b, out_add)
    out_mul = RegisterTensor.create(dtype=float32, shape=(4,))
    mul_inst = MulInst.create(out_add, a, out_mul)
    store = _store(a)  # only 'a' is consumed

    prog = _make_program([alloc_a, alloc_b, add_inst, mul_inst, store])
    assert _count_insts(prog, AddInst) == 1
    assert _count_insts(prog, MulInst) == 1

    opt_prog = dead_code_elimination_pass()(prog)
    assert _count_insts(opt_prog, AddInst) == 0
    assert _count_insts(opt_prog, MulInst) == 0
    assert _count_insts(opt_prog, StoreGlobalGenericInst) == 1


def test_partial_chain_elimination():
    """Only the unused tail of a chain should be eliminated; live prefix remains."""
    alloc_a, a = _alloc()
    alloc_b, b = _alloc()
    out_add = RegisterTensor.create(dtype=float32, shape=(4,))
    add_inst = AddInst.create(a, b, out_add)
    out_mul = RegisterTensor.create(dtype=float32, shape=(4,))
    mul_inst = MulInst.create(out_add, a, out_mul)  # dead: mul output not consumed
    store = _store(out_add)  # add output is live

    prog = _make_program([alloc_a, alloc_b, add_inst, mul_inst, store])
    opt_prog = dead_code_elimination_pass()(prog)

    assert _count_insts(opt_prog, AddInst) == 1
    assert _count_insts(opt_prog, MulInst) == 0
    assert _count_insts(opt_prog, StoreGlobalGenericInst) == 1


def test_no_change_when_all_live():
    """When all instructions are live, the function should be returned by identity."""
    alloc_a, a = _alloc()
    alloc_b, b = _alloc()
    out_add = RegisterTensor.create(dtype=float32, shape=(4,))
    add_inst = AddInst.create(a, b, out_add)
    store = _store(out_add)

    prog = _make_program([alloc_a, alloc_b, add_inst, store])
    opt_prog = dead_code_elimination_pass()(prog)

    func = list(prog.functions.values())[0]
    opt_func = list(opt_prog.functions.values())[0]
    assert func is opt_func


def test_side_effecting_never_eliminated():
    """Side-effecting instructions are never eliminated."""
    alloc_a, a = _alloc()
    store = _store(a)
    sync = SyncThreadsInst.create()

    prog = _make_program([alloc_a, store, sync])
    opt_prog = dead_code_elimination_pass()(prog)

    assert _count_insts(opt_prog, StoreGlobalGenericInst) == 1
    assert _count_insts(opt_prog, SyncThreadsInst) == 1


def test_cast_elimination():
    """An unused CastInst should be eliminated."""
    alloc_a, a = _alloc()
    out_cast = RegisterTensor.create(dtype=float16, shape=(4,))
    cast_inst = CastInst.create(a, out_cast)
    store = _store(a)  # store 'a', not the cast output

    prog = _make_program([alloc_a, cast_inst, store])
    assert _count_insts(prog, CastInst) == 1

    opt_prog = dead_code_elimination_pass()(prog)
    assert _count_insts(opt_prog, CastInst) == 0


def test_unused_allocate_register_eliminated():
    """An AllocateRegisterInst whose output is never consumed should be eliminated."""
    alloc_a, a = _alloc()
    alloc_b, _b = _alloc()  # b is never consumed
    store = _store(a)

    prog = _make_program([alloc_a, alloc_b, store])
    assert _count_insts(prog, AllocateRegisterInst) == 2

    opt_prog = dead_code_elimination_pass()(prog)
    assert _count_insts(opt_prog, AllocateRegisterInst) == 1


def test_diamond_dependency():
    """Both branches of a diamond are live if the final consumer is live."""
    alloc_a, a = _alloc()
    alloc_b, b = _alloc()
    out_add = RegisterTensor.create(dtype=float32, shape=(4,))
    add_inst = AddInst.create(a, b, out_add)
    out_mul = RegisterTensor.create(dtype=float32, shape=(4,))
    mul_inst = MulInst.create(a, b, out_mul)
    # Final add consuming both add_out and mul_out
    out_final = RegisterTensor.create(dtype=float32, shape=(4,))
    final_add = AddInst.create(out_add, out_mul, out_final)
    store = _store(out_final)

    prog = _make_program([alloc_a, alloc_b, add_inst, mul_inst, final_add, store])
    opt_prog = dead_code_elimination_pass()(prog)

    assert _count_insts(opt_prog, AddInst) == 2
    assert _count_insts(opt_prog, MulInst) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
