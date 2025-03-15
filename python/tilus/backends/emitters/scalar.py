from typing import Callable
from hidet.ir.expr import var, Expr
from hidet.ir.primitives.cuda.atomic import atomic_add, atomic_sub, atomic_min, atomic_max
from tilus.extensions.hidet.ir.primitives.cuda.ldst import load, store
from hidet.ir.tools import infer_type
from hidet.ir.type import DataType, PointerType

from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.instructions import (
    AllocateScalarInst,
    AssignScalarInst,
    LoadScalarInst,
    StoreScalarInst,
    AtomicScalarInst,
)
from tilus.target import nvgpu_any, gpgpu_any


@register_inst_emitter(AllocateScalarInst, target=gpgpu_any)
class AllocateScalarInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateScalarInst):
        self.declare(inst.var, init=inst.init)


@register_inst_emitter(AssignScalarInst, target=gpgpu_any)
class AssignScalarInstEmitter(BaseInstEmitter):
    def emit(self, inst: AssignScalarInst):
        self.assign(inst.var, inst.scalar_expr)


@register_inst_emitter(LoadScalarInst, target=nvgpu_any)
class LoadScalarInstEmitter(BaseInstEmitter):
    def emit(self, inst: LoadScalarInst):
        dtype = inst.var.type
        assert isinstance(dtype, DataType)
        self.declare(inst.var)
        self.append(
            load(
                dtype=dtype,
                addr=inst.ptr,
                dst_addrs=[~inst.var],
                space="generic" if inst.sync is not None else "global",
                sync=inst.sync if inst.sync != "weak" else None,
            )
        )


@register_inst_emitter(StoreScalarInst, target=nvgpu_any)
class StoreScalarInstEmitter(BaseInstEmitter):
    def emit(self, inst: StoreScalarInst):
        ptr_type = infer_type(inst.ptr)
        if isinstance(ptr_type, PointerType):
            dtype = ptr_type.base_type
        else:
            raise NotImplementedError(ptr_type)
        assert isinstance(dtype, DataType)
        value = self.declare(var("value", dtype), init=inst.value)
        self.append(
            store(
                dtype=dtype,
                addr=inst.ptr,
                src_addrs=[~value],
                space="generic" if inst.sync is not None else "global",
                sync=inst.sync if inst.sync != "weak" else None,
            )
        )


@register_inst_emitter(AtomicScalarInst, target=nvgpu_any)
class AtomicScalarInstEmitter(BaseInstEmitter):
    def emit(self, inst: AtomicScalarInst):
        op2primitive: dict[str, Callable[[Expr, Expr], Expr]] = {
            "add": atomic_add,
            "sub": atomic_sub,
            "min": atomic_min,
            "max": atomic_max,
        }
        if inst.op not in op2primitive:
            raise NotImplementedError(inst.op)
        prim_func = op2primitive[inst.op]
        self.append(prim_func(inst.ptr, inst.value))
