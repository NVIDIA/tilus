from hidet.ir.dtypes import int32
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.type import tensor_pointer_type
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.inst import AllocateSharedInst, FreeSharedInst
from tilus.ir.tensor import SharedTensor
from tilus.target import gpgpu_any


@register_inst_emitter(AllocateSharedInst, target=gpgpu_any)
class AllocateSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateSharedInst) -> None:
        value: SharedTensor = inst.shared_output

        allocator_addr = self.codegen.allocate_shared_value(value, nbytes=value.nbytes)
        self.tensor2var[value] = self.declare_var(
            name="shared",
            tp=tensor_pointer_type(dtype=value.dtype, shape=[value.size]),
            init=dynamic_shared_memory(byte_offset=allocator_addr, dtype=value.dtype),
        )
        shared_space_addr = cvta_generic_to_shared(self.tensor2var[value])
        self.shared_value_shared_space_addr[value] = self.declare_var(
            name="shared_addr", tp=int32, init=shared_space_addr
        )


@register_inst_emitter(FreeSharedInst, target=gpgpu_any)
class FreeSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: FreeSharedInst) -> None:
        value: SharedTensor = inst.inputs[0].as_shared_tensor()
        self.codegen.free_shared_value(value)

        del self.tensor2var[value]
        del self.shared_value_shared_space_addr[value]
