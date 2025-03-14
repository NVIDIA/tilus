from typing import List

from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_configs
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.stmt import asm
from hidet.utils import initialize
from tilus.extensions.hidet.ir.expr import deref


@initialize()
def register_mma_instructions():
    for config in mma_configs.values():
        inst_name = config.inst_name()

        a_regs, b_regs, c_regs = config.a_regs, config.b_regs, config.c_regs
        template_sub_strings = [
            inst_name,
            "{{{}}},".format(", ".join([f"%{i}" for i in range(c_regs)])),
            "{{{}}},".format(", ".join([f"%{i}" for i in range(c_regs, c_regs + a_regs)])),
            "{{{}}},".format(", ".join([f"%{i}" for i in range(c_regs + a_regs, c_regs + a_regs + b_regs)])),
            "{{{}}};".format(", ".join([f"%{i}" for i in range(c_regs)])),
        ]
        template_string = " ".join(template_sub_strings)

        # v1
        func_name = "cuda_" + inst_name.replace(".", "_")

        # v2
        from hidet.lang import attrs, script, meta
        from hidet.lang.types import void_p, uint32

        a_reg_p_type = meta.types([void_p for _ in range(config.a_regs)])
        b_reg_p_type = meta.types([void_p for _ in range(config.b_regs)])
        c_reg_p_type = meta.types([void_p for _ in range(config.c_regs)])

        @script
        def mma_sync_v2_primitive(a_reg_p: a_reg_p_type, b_reg_p: b_reg_p_type, c_reg_p: c_reg_p_type):
            attrs.func_name = func_name + "_v2"
            attrs.func_kind = "cuda_internal"

            asm(
                template_string,
                output_inputs=[deref(c_reg_p[i], uint32) for i in range(c_regs)],
                inputs=[deref(a_reg_p[i], uint32) for i in range(a_regs)]
                + [deref(b_reg_p[i], uint32) for i in range(b_regs)],
            )

        register_primitive_function(mma_sync_v2_primitive.name, mma_sync_v2_primitive)


def mma_sync_v2(config: MmaConfig, a_reg_p: List[Expr], b_reg_p: List[Expr], c_reg_p: List[Expr]):
    name = config.inst_name().replace(".", "_") + "_v2"
    return call_cuda(func_name=name, args=[*a_reg_p, *b_reg_p, *c_reg_p])
