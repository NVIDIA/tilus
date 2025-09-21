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
from typing import Optional, Sequence, no_type_check

from hidet.ir.dtypes import int32, uint8, uint32, uint64
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func
from hidet.ir.stmt import asm
from hidet.utils import initialize

from tilus.extensions.hidet.ir.primitives.utils import register_primitive_function_decorator


class Tcgen05CtaGroupKind(Enum):
    CTA_1 = ".cta_group::1"
    CTA_2 = ".cta_group::2"

    @staticmethod
    def from_int(cta_group: int) -> "Tcgen05CtaGroupKind":
        assert cta_group in (1, 2)
        if cta_group == 1:
            return Tcgen05CtaGroupKind.CTA_1
        elif cta_group == 2:
            return Tcgen05CtaGroupKind.CTA_2
        else:
            raise ValueError(f"Unsupported cta_group: {cta_group}")


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


class Tcgen05CopyShapeKind(Enum):
    R128x256B = ".128x256b"
    R128x128B = ".128x128b"
    R64x128B = ".64x128b"
    R32x128B = ".32x128b"
    R4x128B = ".4x128b"

    def as_int_tuple(self) -> tuple[int, int]:
        table = {
            Tcgen05CopyShapeKind.R128x256B: (128, 256),
            Tcgen05CopyShapeKind.R128x128B: (128, 128),
            Tcgen05CopyShapeKind.R64x128B: (64, 128),
            Tcgen05CopyShapeKind.R32x128B: (32, 128),
            Tcgen05CopyShapeKind.R4x128B: (4, 128),
        }
        return table[self]


class Tcgen05CopyMulticastKind(Enum):
    NONE = ""
    WARP_X2_02_13 = ".warpx2_02_13"
    WARP_X2_01_23 = ".warpx2_01_23"
    WARP_X4 = ".warpx4"


class Tcgen05CommitMulticastKind(Enum):
    NONE = ""
    CLUSTER = ".multicast::cluster"


def get_num_reg32(
    shape: Tcgen05LoadStoreShapeKind, num: Tcgen05LoadStoreNumKind, pack: Tcgen05LoadStorePackKind
) -> int:
    """
    Get the number of 32-bit registers needed for the tcgen05 load/store instruction given the shape, num, and pack.

    See the tables:
    - https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-num-shapes-ld
    - https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-num-shapes-st
    """
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


def resolve_tcgen05_relinquish_alloc_permit(cta_group: Tcgen05CtaGroupKind) -> str:
    ret = "cuda_tcgen05_relinquish_alloc_permit_cta_group_" + cta_group.value
    return ret.replace(".", "_").replace("::", "_")


def resolve_tcgen05_alloc(cta_group: Tcgen05CtaGroupKind) -> str:
    ret = "cuda_tcgen05_alloc_cta_group_" + cta_group.value
    return ret.replace(".", "_").replace("::", "_")


def resolve_tcgen05_dealloc(cta_group: Tcgen05CtaGroupKind) -> str:
    ret = "cuda_tcgen05_dealloc_cta_group_" + cta_group.value
    return ret.replace(".", "_").replace("::", "_")


def resolve_tcgen05_load(
    pack: Tcgen05LoadStorePackKind, num: Tcgen05LoadStoreNumKind, shape: Tcgen05LoadStoreShapeKind
) -> str:
    ret = "cuda_tcgen05_load" + pack.value + num.value + shape.value
    ret = ret.replace(".", "_").replace("::", "_")
    return ret


def resolve_tcgen05_store(
    pack: Tcgen05LoadStorePackKind, num: Tcgen05LoadStoreNumKind, shape: Tcgen05LoadStoreShapeKind
) -> str:
    ret = "cuda_tcgen05_store" + pack.value + num.value + shape.value
    ret = ret.replace(".", "_").replace("::", "_")
    return ret


def resolve_tcgen05_copy(
    cta_group: Tcgen05CtaGroupKind, shape: Tcgen05CopyShapeKind, multicast: Tcgen05CopyMulticastKind
) -> str:
    ret = "cuda_tcgen05_copy_cta_group" + cta_group.value + shape.value + multicast.value
    ret = ret.replace(".", "_").replace("::", "_")
    return ret


def resolve_tcgen05_commit(
    cta_group: Tcgen05CtaGroupKind, multicast: Tcgen05CommitMulticastKind, has_mask: bool
) -> str:
    ret = "cuda_tcgen05_commit_cta_group" + cta_group.value + multicast.value + ("_mask" if has_mask else "")
    ret = ret.replace(".", "_").replace("::", "_")
    return ret


@initialize()
def register_tcgen05_instructions():
    from hidet.lang import attrs, meta

    from tilus.extensions.hidet.lang import script

    # alloc, dealloc, relinquish_alloc_permit
    for cta_group in [Tcgen05CtaGroupKind.CTA_1, Tcgen05CtaGroupKind.CTA_2]:

        @register_primitive_function_decorator
        @no_type_check
        @script
        def tcgen05_relinquish_alloc_permit_():
            attrs.func_name = resolve_tcgen05_relinquish_alloc_permit(cta_group)
            attrs.func_kind = "cuda_internal"
            asm("tcgen05.relinquish_alloc_permit{}.sync.aligned;".format(cta_group.value), is_volatile=True)

        @register_primitive_function_decorator
        @no_type_check
        @script
        def tcgen05_alloc_(dst: uint32, num_columns: uint32):
            attrs.func_name = resolve_tcgen05_alloc(cta_group)
            attrs.func_kind = "cuda_internal"
            asm(
                "tcgen05.alloc{}.sync.aligned.shared::cta.b32 [%0], %1;".format(cta_group.value),
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
                "tcgen05.dealloc{}.sync.aligned.b32 %0, %1;".format(cta_group.value),
                inputs=[taddr, num_columns],
                is_volatile=True,
            )

    # load, store
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
                regs_type = meta.types(arg_types=[~uint32 for _ in range(num_regs)])

                load_regs = ", ".join([f"%{i}" for i in range(num_regs)])
                load_template = (
                    f"tcgen05.ld.sync.aligned{shape.value}{num.value}{pack.value}.b32 {{{load_regs}}}, [%{num_regs}];"
                )
                store_regs = ", ".join([f"%{i + 1}" for i in range(num_regs)])
                store_template = (
                    f"tcgen05.st.sync.aligned{shape.value}{num.value}{pack.value}.b32 [%0], {{{store_regs}}};"
                )

                @register_primitive_function_decorator
                @no_type_check
                @script
                def tcgen05_load_(taddr: int32, regs: regs_type):
                    attrs.func_name = resolve_tcgen05_load(pack, num, shape)
                    attrs.func_kind = "cuda_internal"
                    asm(
                        load_template,
                        outputs=[reg[0] for reg in regs],
                        inputs=[taddr],
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

    # wait_load, wait_store
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

    # copy
    for cta_group in [Tcgen05CtaGroupKind.CTA_1, Tcgen05CtaGroupKind.CTA_2]:
        for shape_kind in [
            Tcgen05CopyShapeKind.R128x256B,
            Tcgen05CopyShapeKind.R128x128B,
            Tcgen05CopyShapeKind.R64x128B,
            Tcgen05CopyShapeKind.R32x128B,
            Tcgen05CopyShapeKind.R4x128B,
        ]:
            for multicast in [
                Tcgen05CopyMulticastKind.NONE,
                Tcgen05CopyMulticastKind.WARP_X2_02_13,
                Tcgen05CopyMulticastKind.WARP_X2_01_23,
                Tcgen05CopyMulticastKind.WARP_X4,
            ]:
                template = f"tcgen05.cp{cta_group.value}{shape_kind.value}{multicast.value} [%0], %1;"

                @register_primitive_function_decorator
                @no_type_check
                @script
                def tcgen05_copy_(taddr: int32, sdesc: uint64):
                    attrs.func_name = resolve_tcgen05_copy(cta_group, shape_kind, multicast)
                    attrs.func_kind = "cuda_internal"
                    asm(template, inputs=[taddr, sdesc], is_volatile=True)

    # commit
    for cta_group in [Tcgen05CtaGroupKind.CTA_1, Tcgen05CtaGroupKind.CTA_2]:
        for multicast in [  # type: ignore[assignment]
            Tcgen05CommitMulticastKind.NONE,
            Tcgen05CommitMulticastKind.CLUSTER,
        ]:
            for has_mask in [False, True]:
                template = f"tcgen05.commit{cta_group.value}.mbarrier::arrive::one.shared::cluster{multicast.value}.b64 [%0]{', %1' if has_mask else ''};"
                cta_mask_type = meta.types(arg_types=[uint32]) if has_mask else meta.types(arg_types=[])

                @register_primitive_function_decorator
                @no_type_check
                @script
                def tcgen05_commit_(mbarrier: int32, cta_mask: cta_mask_type):
                    attrs.func_name = resolve_tcgen05_commit(cta_group, multicast, has_mask)
                    attrs.func_kind = "cuda_internal"
                    asm(template, inputs=[mbarrier, *cta_mask], is_volatile=True)

    # encode_smem_descriptor
    @register_primitive_function_decorator
    @no_type_check
    @script
    def tcgen05_encode_smem_descriptor(
        smem_addr: uint32,  # 14 bits
        lbo: uint32,  # 14 bits
        sbo: uint32,  # 14 bits
        mbo: uint8,  # 3 bits
        stride_mode: uint8,  # 1 bit
        swizzle_mode: uint8,  # 3 bits
    ) -> uint64:
        attrs.func_name = "cuda_tcgen05_encode_smem_descriptor"
        attrs.func_kind = "cuda_internal"
        desc: uint64 = uint64(0)
        desc = desc | uint64(lbo & uint32(0x3FFF)) << 16
        desc = desc | uint64(sbo & uint32(0x3FFF)) << 32
        desc = desc | uint64(0b001) << 46
        desc = desc | uint64(mbo & uint8(0b111)) << 49
        desc = desc | uint64(stride_mode & uint8(0b1)) << 52
        desc = desc | uint64(swizzle_mode & uint8(0b111)) << 61
        desc = desc | uint64(smem_addr & uint32(0x3FFF))
        return desc


def tcgen05_relinquish_alloc_permit(cta_group: Tcgen05CtaGroupKind) -> Expr:
    func_name = resolve_tcgen05_relinquish_alloc_permit(cta_group)
    return call_primitive_func(func_name, [])


def tcgen05_alloc(dst: Expr, num_columns: Expr, cta_group: Tcgen05CtaGroupKind) -> Expr:
    func_name = resolve_tcgen05_alloc(cta_group)
    return call_primitive_func(func_name, [dst, num_columns])


def tcgen05_dealloc(taddr: Expr, num_columns: Expr, cta_group: Tcgen05CtaGroupKind) -> Expr:
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


def tcgen05_encode_smem_descriptor(
    smem_addr: Expr,
    lbo: Expr | int,
    sbo: Expr | int,
    mbo: Expr | int,
    stride_mode: Expr | int,
    swizzle_mode: Expr | int,
) -> Expr:
    func_name = "cuda_tcgen05_encode_smem_descriptor"
    return call_primitive_func(
        func_name, [smem_addr, uint32(lbo), uint32(sbo), uint8(mbo), uint8(stride_mode), uint8(swizzle_mode)]
    )


def tcgen05_copy(
    taddr: Expr,
    sdesc: Expr,
    cta_group: Tcgen05CtaGroupKind,
    shape: Tcgen05CopyShapeKind,
    multicast: Tcgen05CopyMulticastKind,
) -> Expr:
    func_name = resolve_tcgen05_copy(cta_group, shape, multicast)
    return call_primitive_func(func_name, [taddr, sdesc])


def tcgen05_commit(
    mbarrier: Expr,
    cta_mask: Optional[int],
    cta_group: Tcgen05CtaGroupKind,
    multicast: Tcgen05CommitMulticastKind,
) -> Expr:
    func_name = resolve_tcgen05_commit(cta_group, multicast, cta_mask is not None)
    if cta_mask is None:
        args = [mbarrier]
    else:
        args = [mbarrier, uint32(cta_mask)]
    return call_primitive_func(func_name, args)
