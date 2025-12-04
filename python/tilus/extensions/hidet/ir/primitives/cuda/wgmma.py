from enum import Enum
from typing import Optional, Sequence, no_type_check

from hidet.ir.dtypes import int32, uint8, uint32, uint64
from hidet.ir.expr import Expr, as_expr
from hidet.ir.primitives.func import call_primitive_func
from hidet.ir.stmt import asm
from hidet.utils import initialize

from tilus.extensions.hidet.ir.primitives.utils import register_primitive_function_decorator

@initialize()
def register_wgmma_instructions():
    from hidet.lang import attrs, meta

    from tilus.extensions.hidet.lang import script

    @register_primitive_function_decorator
    @no_type_check
    @script
    def wgmma_encode_smem_descriptor(
        smem_addr: uint32,  # 14 bits
        lbo: uint32,  # 14 bits
        sbo: uint32,  # 14 bits
        mbo: uint8,  # 3 bits
        swizzle_mode: uint8,  # 2 bits
    ) -> uint64:
        attrs.func_name = "cuda_wgmma_encode_smem_descriptor"
        attrs.func_kind = "cuda_internal"
        desc: uint64 = uint64(0)
        desc = desc | uint64(lbo & uint32(0x3FFF)) << 16
        desc = desc | uint64(sbo & uint32(0x3FFF)) << 32
        desc = desc | uint64(mbo & uint8(0b111)) << 49
        desc = desc | uint64(swizzle_mode & uint8(0b11)) << 62
        desc = desc | uint64(smem_addr & uint32(0x3FFF))
        return desc


def wgmma_encode_smem_descriptor(
    smem_addr: Expr | int,
    lbo: Expr | int,
    sbo: Expr | int,
    mbo: Expr | int,
    swizzle_mode: Expr | int,
) -> Expr:
    func_name = "cuda_wgmma_encode_smem_descriptor"
    return call_primitive_func(
        func_name, [uint32(smem_addr), uint32(lbo), uint32(sbo), uint8(mbo), uint8(swizzle_mode)]
    )