from hidet.ir.dtypes import int32
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.type import tensor_pointer_type
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.inst import Instruction
from tilus.ir.instructions import AllocateSharedInst, FreeSharedInst, SharedSliceInst
from tilus.ir.tensor import SharedTensor


@register_emitter(AllocateSharedInst)
class AllocateSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateSharedInst) -> None:
        tensor: SharedTensor = inst.shared_output

        allocator_addr = self.codegen.allocate_shared_tensor(tensor, nbytes=tensor.nbytes)
        self.tensor2var[tensor] = self.declare_var(
            name="shared",
            tp=tensor_pointer_type(dtype=tensor.dtype, shape=[tensor.size]),
            init=dynamic_shared_memory(byte_offset=allocator_addr, dtype=tensor.dtype),
        )
        shared_space_addr = cvta_generic_to_shared(self.tensor2var[tensor])
        self.shared_tensor_shared_space_addr[tensor] = self.declare_var(
            name="shared_addr", tp=int32, init=shared_space_addr
        )


@register_emitter(FreeSharedInst)
class FreeSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: FreeSharedInst) -> None:
        tensor: SharedTensor = inst.inputs[0].as_shared_tensor()
        self.codegen.free_shared_value(tensor)

        del self.tensor2var[tensor]
        del self.shared_tensor_shared_space_addr[tensor]


@register_emitter(SharedSliceInst)
class SharedSliceInstEmitter(BaseInstEmitter):
    def emit(self, inst: Instruction) -> None:
        shared_input: SharedTensor = inst.shared_input
        shared_output: SharedTensor = inst.shared_output
        self.tensor2var[shared_output] = self.tensor2var[shared_input]
        self.shared_tensor_shared_space_addr[shared_output] = self.shared_tensor_shared_space_addr[shared_input]
