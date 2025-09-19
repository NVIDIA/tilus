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
from enum import Enum
from typing import Sequence, no_type_check

from hidet.ir.dtypes import int32, uint32
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func
from hidet.ir.stmt import asm
from hidet.utils import initialize

from tilus.extensions.hidet.ir.primitives.utils import register_primitive_function_decorator


class Tcgen05LoadStoreShapeKind(Enum):
    R16x64B = ".16x64b"
    R16x128B = ".16x128b"
    R16x256B = ".16x256b"
    R32x32B = ".32x32b"

    def rows(self) -> int:
        if self == Tcgen05LoadStoreShapeKind.R16x64B:
            return 16
        elif self == Tcgen05LoadStoreShapeKind.R16x128B:
            return 16
        elif self == Tcgen05LoadStoreShapeKind.R16x256B:
            return 16
        elif self == Tcgen05LoadStoreShapeKind.R32x32B:
            return 32
        else:
            raise ValueError(f"Unsupported shape: {self}")

    def columns_bits(self) -> int:
        if self == Tcgen05LoadStoreShapeKind.R16x64B:
            return 64
        elif self == Tcgen05LoadStoreShapeKind.R16x128B:
            return 128
        elif self == Tcgen05LoadStoreShapeKind.R16x256B:
            return 256
        elif self == Tcgen05LoadStoreShapeKind.R32x32B:
            return 32
        else:
            raise ValueError(f"Unsupported shape: {self}")

    def regs_per_thread(self) -> int:
        if self == Tcgen05LoadStoreShapeKind.R16x64B:
            return 1
        elif self == Tcgen05LoadStoreShapeKind.R32x32B:
            return 1
        elif self == Tcgen05LoadStoreShapeKind.R16x128B:
            return 2
        elif self == Tcgen05LoadStoreShapeKind.R16x256B:
            return 4
        else:
            raise ValueError(f"Unsupported shape: {self}")


class Tcgen05LoadStoreNumKind(Enum):
    X1 = ".x1"
    X2 = ".x2"
    X4 = ".x4"
    X8 = ".x8"
    X16 = ".x16"
    X32 = ".x32"
    X64 = ".x64"
    X128 = ".x128"

    def __int__(self) -> int:
        return int(self.value[2:])

    @staticmethod
    def from_int(num: int) -> "Tcgen05LoadStoreNumKind":
        if num == 1:
            return Tcgen05LoadStoreNumKind.X1
        elif num == 2:
            return Tcgen05LoadStoreNumKind.X2
        elif num == 4:
            return Tcgen05LoadStoreNumKind.X4
        elif num == 8:
            return Tcgen05LoadStoreNumKind.X8
        elif num == 16:
            return Tcgen05LoadStoreNumKind.X16
        elif num == 32:
            return Tcgen05LoadStoreNumKind.X32
        elif num == 64:
            return Tcgen05LoadStoreNumKind.X64
        elif num == 128:
            return Tcgen05LoadStoreNumKind.X128
        else:
            raise ValueError(f"Unsupported num: {num}")


class Tcgen05LoadStorePackKind(Enum):
    NONE = ""
    PACK_16B = ".pack::16b"


def get_num_reg32(
    shape: Tcgen05LoadStoreShapeKind, num: Tcgen05LoadStoreNumKind, pack: Tcgen05LoadStorePackKind
) -> int:
    base = {
        Tcgen05LoadStoreShapeKind.R16x64B: 1,
        Tcgen05LoadStoreShapeKind.R32x32B: 1,
        Tcgen05LoadStoreShapeKind.R16x128B: 2,
        Tcgen05LoadStoreShapeKind.R16x256B: 4,
    }
    multiplier = {
        Tcgen05LoadStoreNumKind.X1: 1,
        Tcgen05LoadStoreNumKind.X2: 2,
        Tcgen05LoadStoreNumKind.X4: 4,
        Tcgen05LoadStoreNumKind.X8: 8,
        Tcgen05LoadStoreNumKind.X16: 16,
        Tcgen05LoadStoreNumKind.X32: 32,
        Tcgen05LoadStoreNumKind.X64: 64,
        Tcgen05LoadStoreNumKind.X128: 128,
    }
    num_reg32 = base[shape] * multiplier[num]
    if pack == Tcgen05LoadStorePackKind.PACK_16B:
        num_reg32 = num_reg32 // 2
    return num_reg32


def resolve_tcgen05_relinquish_alloc_permit(cta_group: int) -> str:
    assert cta_group in (1, 2)
    return "cuda_tcgen05_relinquish_alloc_permit_cta_group_" + str(cta_group)


def resolve_tcgen05_alloc(cta_group: int) -> str:
    assert cta_group in (1, 2)
    return "cuda_tcgen05_alloc_cta_group_" + str(cta_group)


def resolve_tcgen05_dealloc(cta_group: int) -> str:
    assert cta_group in (1, 2)
    return "cuda_tcgen05_dealloc_cta_group_" + str(cta_group)


def resolve_tcgen05_load(
    pack: Tcgen05LoadStorePackKind, num: Tcgen05LoadStoreNumKind, shape: Tcgen05LoadStoreShapeKind
) -> str:
    ret = "cuda_tcgen05_load_" + pack.value + num.value + shape.value
    ret = ret.replace(".", "_").replace("::", "_")
    return ret


def resolve_tcgen05_store(
    pack: Tcgen05LoadStorePackKind, num: Tcgen05LoadStoreNumKind, shape: Tcgen05LoadStoreShapeKind
) -> str:
    ret = "cuda_tcgen05_store_" + pack.value + num.value + shape.value
    ret = ret.replace(".", "_").replace("::", "_")
    return ret


@initialize()
def register_tcgen05_instructions():
    from hidet.lang import attrs, meta

    from tilus.extensions.hidet.lang import script

    for cta_group in [1, 2]:

        @register_primitive_function_decorator
        @no_type_check
        @script
        def tcgen05_relinquish_alloc_permit_():
            attrs.func_name = resolve_tcgen05_relinquish_alloc_permit(cta_group)
            attrs.func_kind = "cuda_internal"
            asm("tcgen05.relinquish_alloc_permit.cta_group::{}.sync.aligned;".format(cta_group), is_volatile=True)

        @register_primitive_function_decorator
        @no_type_check
        @script
        def tcgen05_alloc_(dst: uint32, num_columns: uint32):
            attrs.func_name = resolve_tcgen05_alloc(cta_group)
            attrs.func_kind = "cuda_internal"
            asm(
                "tcgen05.alloc.cta_group::{}.sync.aligned.shared::cta.b32 [%0], %1;".format(cta_group),
                inputs=[dst, num_columns],
                is_volatile=True,
            )

        @register_primitive_function_decorator
        @no_type_check
        @script
        def tcgen05_dealloc_(taddr: int32, num_columns: uint32):
            attrs.func_name = resolve_tcgen05_dealloc(cta_group)
            attrs.func_kind = "cuda_internal"
            asm(
                "tcgen05.dealloc.cta_group::{}.sync.aligned.b32 %0, %1;".format(cta_group),
                inputs=[taddr, num_columns],
                is_volatile=True,
            )

    for pack in [Tcgen05LoadStorePackKind.NONE, Tcgen05LoadStorePackKind.PACK_16B]:
        for num in [
            Tcgen05LoadStoreNumKind.X1,
            Tcgen05LoadStoreNumKind.X2,
            Tcgen05LoadStoreNumKind.X4,
            Tcgen05LoadStoreNumKind.X8,
            Tcgen05LoadStoreNumKind.X16,
            Tcgen05LoadStoreNumKind.X32,
            Tcgen05LoadStoreNumKind.X64,
            Tcgen05LoadStoreNumKind.X128,
        ]:
            for shape in [
                Tcgen05LoadStoreShapeKind.R16x64B,
                Tcgen05LoadStoreShapeKind.R16x128B,
                Tcgen05LoadStoreShapeKind.R16x256B,
                Tcgen05LoadStoreShapeKind.R32x32B,
            ]:
                num_regs = get_num_reg32(shape, num, pack)
                regs = ", ".join([f"%{i + 1}" for i in range(num_regs)])
                regs_type = meta.types(arg_types=[~uint32 for _ in range(num_regs)])

                load_template = f"tcgen05.ld.sync.aligned{shape.value}{num.value}{pack.value}.b32 {{{regs}}}, [%0];"
                store_template = f"tcgen05.st.sync.aligned{shape.value}{num.value}{pack.value}.b32 [%0], {{{regs}}};"

                @register_primitive_function_decorator
                @no_type_check
                @script
                def tcgen05_load_(taddr: int32, regs: regs_type):
                    attrs.func_name = resolve_tcgen05_load(pack, num, shape)
                    attrs.func_kind = "cuda_internal"
                    asm(
                        load_template,
                        inputs=[taddr, *[reg[0] for reg in regs]],
                        is_volatile=True,
                    )

                @register_primitive_function_decorator
                @no_type_check
                @script
                def tcgen05_store_(taddr: int32, regs: regs_type):
                    attrs.func_name = resolve_tcgen05_store(pack, num, shape)
                    attrs.func_kind = "cuda_internal"
                    asm(
                        store_template,
                        inputs=[taddr, *[reg[0] for reg in regs]],
                        is_volatile=True,
                    )

    @register_primitive_function_decorator
    @no_type_check
    @script
    def tcgen05_wait_load_():
        attrs.func_name = "cuda_tcgen05_wait_load"
        attrs.func_kind = "cuda_internal"
        asm("tcgen05.wait::ld.sync.aligned;", is_volatile=True)

    @register_primitive_function_decorator
    @no_type_check
    @script
    def tcgen05_wait_store_():
        attrs.func_name = "cuda_tcgen05_wait_store"
        attrs.func_kind = "cuda_internal"
        asm("tcgen05.wait::st.sync.aligned;", is_volatile=True)


def tcgen05_relinquish_alloc_permit(cta_group: int) -> Expr:
    func_name = resolve_tcgen05_relinquish_alloc_permit(cta_group)
    return call_primitive_func(func_name, [])


def tcgen05_alloc(dst: Expr, num_columns: Expr, cta_group: int) -> Expr:
    func_name = resolve_tcgen05_alloc(cta_group)
    return call_primitive_func(func_name, [dst, num_columns])


def tcgen05_dealloc(taddr: Expr, num_columns: Expr, cta_group: int) -> Expr:
    func_name = resolve_tcgen05_dealloc(cta_group)
    return call_primitive_func(func_name, [taddr, num_columns])


def tcgen05_load(
    taddr: Expr,
    regs: Sequence[Expr],
    pack: Tcgen05LoadStorePackKind,
    num: Tcgen05LoadStoreNumKind,
    shape: Tcgen05LoadStoreShapeKind,
) -> Expr:
    func_name = resolve_tcgen05_load(pack, num, shape)
    return call_primitive_func(func_name, [taddr, *regs])


def tcgen05_store(
    taddr: Expr,
    regs: Sequence[Expr],
    pack: Tcgen05LoadStorePackKind,
    num: Tcgen05LoadStoreNumKind,
    shape: Tcgen05LoadStoreShapeKind,
) -> Expr:
    func_name = resolve_tcgen05_store(pack, num, shape)
    return call_primitive_func(func_name, [taddr, *regs])


def tcgen05_wait_load() -> Expr:
    func_name = "cuda_tcgen05_wait_load"
    return call_primitive_func(func_name, [])


def tcgen05_wait_store() -> Expr:
    func_name = "cuda_tcgen05_wait_store"
    return call_primitive_func(func_name, [])
