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
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Type

from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dtypes import int32, uint8
from hidet.ir.expr import Constant, Expr, SymbolVar, Var, cast, tensor_pointer_var, tensor_var
from hidet.ir.func import Function as HidetFunction
from hidet.ir.module import IRModule
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.primitives.cuda.vars import threadIdx
from hidet.ir.stmt import DeclareScope
from hidet.ir.type import void_p
from hidet.utils.doc import Doc, Text

from tilus.extensions.hidet.ir.module import merge_ir_modules
from tilus.extensions.hidet.ir.tools import rewrite
from tilus.extensions.hidet.ir.tools.verifier import verify as verify_ir_module
from tilus.ir.func import Function
from tilus.ir.functors import IRFunctor
from tilus.ir.inst import Instruction
from tilus.ir.instructions import FormatPrintInst, PrintTensorInst
from tilus.ir.layout import shared_row_major
from tilus.ir.prog import Program
from tilus.ir.stmt import (
    AssignStmt,
    DeclareStmt,
    ForStmt,
    ForThreadGroupStmt,
    IfStmt,
    InstStmt,
    LetStmt,
    ReturnStmt,
    SeqStmt,
    TensorElemPtrStmt,
    TensorElemValueStmt,
    WhileStmt,
)
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor
from tilus.ir.tools import IRPrinter
from tilus.ir.tools.instruction_collector import collect_instructions
from tilus.target import Target, get_current_target, gpgpu_any, match_target


class InvalidInstruction(Exception):
    def __init__(self, inst):
        self.inst = inst


class CodeGenerationFailed(Exception):
    pass


def is_nvgpu():
    return get_current_target().is_nvgpu()


def is_amdgpu():
    return get_current_target().is_amdgpu()


class BaseInstEmitter(StmtBuilder):
    # inst -> emitter
    REGISTRY: Dict[Type[Instruction], Dict[Target, Type["BaseInstEmitter"]]] = {}

    def __init__(self, codegen: Codegen) -> None:
        super().__init__()
        # todo: currently, the instruction emitters (that inherit from BaseInstEmitter) directly access the codegen
        #       object to access some data in the codegen object. This is not a good design. We should refactor this
        #       to use the methods of the BaseInstEmitter class to access the data in the codegen object.
        self.codegen: Codegen = codegen

    def sync(self):
        from hidet.ir.primitives.cuda import syncthreads

        if self.codegen.thread_groups.num_levels() == 1:  # all threads in the cta
            self.append(syncthreads())
        else:
            if get_current_target().is_nvgpu():
                from hidet.ir.primitives.cuda.barrier import barrier_sync

                barrier = self.codegen.thread_groups.num_levels() - 1
                count = self.codegen.thread_groups.group_size[-1]
                self.append(barrier_sync(barrier=barrier, count=count))
            else:
                raise NotImplementedError()

    def sync_reduce(self, value: Expr, op: str) -> Expr:
        if get_current_target().is_nvgpu():
            from hidet.ir.primitives.cuda.barrier import barrier_sync
            from hidet.ir.primitives.cuda.sync import syncthreads_and, syncthreads_or

            op2sync = {"and": syncthreads_and, "or": syncthreads_or}
            syncthreads_op = op2sync[op]

            if self.codegen.thread_groups.num_levels() == 1:  # all threads in the cta
                return syncthreads_op(value)
            else:
                barrier = self.codegen.thread_groups.num_levels() - 1
                count = self.codegen.thread_groups.group_size[-1]
                self.append(barrier_sync(barrier=barrier, count=count))
                raise NotImplementedError("barrier_sync_reduce")
        else:
            raise NotImplementedError()

    def get_or_allocate_var(self, tensor: Tensor, name: Optional[str] = None) -> Var:
        if tensor in self.tensor2var:
            return self.tensor2var[tensor]
        else:
            if isinstance(tensor, RegisterTensor):
                name = name if name else "regs"
                var = self.declare(
                    tensor_var(name, shape=[tensor.local_size], dtype=tensor.dtype), scope=DeclareScope.Register
                )
            elif isinstance(tensor, SharedTensor):
                name = name if name else "smem"
                var = self.declare(tensor_pointer_var(name, shape=[tensor.size], dtype=tensor.dtype))
            elif isinstance(tensor, GlobalTensor):
                name = name if name else "gmem"
                var = self.declare(tensor_pointer_var(name, shape=[tensor.size], dtype=tensor.dtype))
            else:
                raise NotImplementedError()
            self.tensor2var[tensor] = var
            return var

    @property
    def current_worker(self) -> Expr:
        return self.codegen.thread_groups.current_worker[-1]

    @property
    def current_num_workers(self) -> int:
        return self.codegen.thread_groups.group_size[-1]

    @property
    def thread_groups(self):
        return self.codegen.thread_groups

    @property
    def tensor2var(self) -> Dict[Tensor, Var]:
        return self.codegen.tensor2var

    @property
    def shared_tensor_shared_space_addr(self):
        return self.codegen.shared_tensor_shared_space_addr

    @property
    def num_warps(self) -> int:
        return self.codegen.function.metadata.num_warps

    def request_shared_workspace(self, inst: Instruction) -> int:
        return 0

    def emit(self, inst: Instruction) -> None:
        raise NotImplementedError()


def register_emitter(
    inst_cls: Type[Instruction], *, target: Optional[Target] = None
) -> Callable[[Type[BaseInstEmitter]], Type[BaseInstEmitter]]:
    assert issubclass(inst_cls, Instruction)
    if target is None:
        target = gpgpu_any

    def decorator(emitter_cls: Type[BaseInstEmitter]) -> Type[BaseInstEmitter]:
        assert issubclass(emitter_cls, BaseInstEmitter)

        if inst_cls not in BaseInstEmitter.REGISTRY:
            BaseInstEmitter.REGISTRY[inst_cls] = {}

        if target in BaseInstEmitter.REGISTRY[inst_cls]:
            raise ValueError(f"Emitter for instruction {inst_cls} and target {target} already exists")

        BaseInstEmitter.REGISTRY[inst_cls][target] = emitter_cls
        return emitter_cls

    return decorator


def resolve_inst_emitter(inst_cls: Type[Instruction]) -> Optional[Type[BaseInstEmitter]]:
    target = get_current_target()
    emitter_classes = {}
    for registry_inst_cls, registry_emitter_classes in BaseInstEmitter.REGISTRY.items():
        if issubclass(inst_cls, registry_inst_cls):
            emitter_classes.update(registry_emitter_classes)
            break

    matched_target = match_target(target, list(emitter_classes))
    if matched_target is None:
        return None
    return emitter_classes[matched_target]


class SharedMemoryAllocator:
    def __init__(self) -> None:
        self.free_slots: List[Tuple[int, int]] = [(0, (1 << 32) - 1)]
        self.addr2nbytes: Dict[int, int] = {}
        self.allocated: int = 0
        self.maximum_allocated: int = 0

    def allocate(self, nbytes: int) -> int:
        # align the nbytes to 128 bytes aligned
        nbytes = (nbytes + 127) // 128 * 128

        # find the first slot that can fit the request
        i = min(i for i, (start, end) in enumerate(self.free_slots) if end - start >= nbytes)
        addr = self.free_slots[i][0]
        if self.free_slots[i][1] - self.free_slots[i][0] == nbytes:
            # remove the slot
            del self.free_slots[i]
        else:
            # shrink the slot
            self.free_slots[i] = (addr + nbytes, self.free_slots[i][1])
        self.addr2nbytes[addr] = nbytes
        self.maximum_allocated = max(self.maximum_allocated, addr + nbytes)
        self.allocated += nbytes
        return addr

    def free(self, addr: int) -> None:
        # find the slot that is right before the address
        before = [i for i, slot in enumerate(self.free_slots) if slot[1] <= addr]
        after = [i for i, slot in enumerate(self.free_slots) if slot[0] > addr]
        assert len(before) + len(after) == len(self.free_slots)
        nbytes = self.addr2nbytes[addr]
        if (
            before
            and after
            and self.free_slots[before[-1]][1] == addr
            and self.free_slots[after[0]][0] == addr + nbytes
        ):
            # merge three slots
            self.free_slots[before[-1]] = (self.free_slots[before[-1]][0], self.free_slots[after[0]][1])
        elif before and self.free_slots[before[-1]][1] == addr:
            # merge with previous slot
            self.free_slots[before[-1]] = (self.free_slots[before[-1]][0], addr + nbytes)
        elif after and self.free_slots[after[0]][0] == addr + nbytes:
            # merge with next slot
            self.free_slots[after[0]] = (addr, self.free_slots[after[0]][1])
        else:
            # add a new slot
            self.free_slots.append((addr, addr + nbytes))
            self.free_slots = list(sorted(self.free_slots, key=lambda x: x[0]))
        self.allocated -= nbytes
        del self.addr2nbytes[addr]


class CommentInlinedIRPrinter(IRPrinter):
    def add_key_comment(self, key_hint: str, comment: str | Doc) -> Doc:
        return Text(comment) if isinstance(comment, str) else comment


class Codegen(IRFunctor):
    GMEM_WORKSPACE_NAME = "__gmem_workspace"
    GMEM_CLEAN_WORKSPACE_NAME = "__gmem_clean_workspace"

    @dataclass
    class ThreadGroups:
        current_worker: List[Expr]
        num_groups: List[int]
        group_size: List[int]

        def num_levels(self):
            return len(self.num_groups)

    def __init__(self) -> None:
        super().__init__()
        self._builder: Optional[FunctionBuilder] = None
        self._function: Optional[Function] = None
        self.printer: IRPrinter = CommentInlinedIRPrinter()

        # value mapping
        self.tensor2var: Dict[Tensor, Var] = {}

        # global memory management
        self.gmem_base_ptr: Var = SymbolVar(dtype=~uint8, name=self.GMEM_WORKSPACE_NAME)  # type: ignore
        self.gmem_allocated: Expr = int32.zero
        self.gmem_maximum_allocated: Expr = int32.zero
        self.gmem_clean_base_ptr: Var = SymbolVar(dtype=~uint8, name=self.GMEM_CLEAN_WORKSPACE_NAME)  # type: ignore
        self.gmem_clean_allocated: Expr = int32.zero
        self.gmem_clean_maximum_allocated: Expr = int32.zero

        # shared memory allocator
        self.smem_allocator: SharedMemoryAllocator = SharedMemoryAllocator()
        # mapping from shared value to the address in shared memory allocator for all allocated shared values
        self.shared_value_allocator_addr: Dict[SharedTensor, int] = {}
        # mapping from shared value to the address in shared memory space (e.g., returned by cvta ptx instruction)
        self.shared_tensor_shared_space_addr: Dict[SharedTensor, Var] = {}

        # shared memory workspace
        self.smem_workspace: Optional[SharedTensor] = None

        # stacks of for_thread_groups
        self.thread_groups = Codegen.ThreadGroups([], [], [])

    def __call__(self, prog: Function) -> IRModule:
        return self.visit(prog)

    @property
    def function(self) -> Function:
        assert self._function is not None
        return self._function

    @property
    def builder(self) -> FunctionBuilder:
        assert self._builder is not None
        return self._builder

    def sync(self) -> None:
        from tilus.ir.instructions import SyncThreadsInst

        self.visit(SyncThreadsInst.create())

    def allocate_shared_tensor(self, value: SharedTensor, nbytes: int) -> int:
        addr: int = self.smem_allocator.allocate(nbytes)
        assert value not in self.shared_value_allocator_addr
        self.shared_value_allocator_addr[value] = addr
        return addr

    def free_shared_value(self, value: SharedTensor) -> None:
        assert value in self.shared_value_allocator_addr
        self.smem_allocator.free(addr=self.shared_value_allocator_addr[value])
        del self.shared_value_allocator_addr[value]

    def allocate_global_memory(self, nbytes: Expr | int, clean: bool) -> Expr:
        nbytes = (nbytes + 127) // 128 * 128  # align to 128 bytes
        if clean:
            ret = self.gmem_clean_base_ptr + self.gmem_clean_allocated
            self.gmem_clean_allocated = self.gmem_clean_allocated + nbytes
            self.gmem_clean_maximum_allocated = self.gmem_clean_allocated
        else:
            ret = self.gmem_base_ptr + self.gmem_allocated
            self.gmem_allocated = self.gmem_allocated + nbytes
            self.gmem_maximum_allocated = self.gmem_allocated
        return ret

    def check_emitter_existence(self) -> None:
        failed_instructions: Set[str] = set()
        for inst in collect_instructions(self.function):
            if resolve_inst_emitter(inst.__class__) is None:
                failed_instructions.add(inst.__class__.__name__)

        if failed_instructions:
            raise CodeGenerationFailed(
                "Failed to find emitter for the following instructions: \n{}".format("\n".join(failed_instructions))
            )

    def init_smem_workspace(self, program: Function) -> None:
        smem_workspace_nbytes: int = 0
        for inst in collect_instructions(program):  # todo: add this to emiter
            # smem_workspace_nbytes = max(smem_workspace_nbytes, inst.request_shared_workspace())
            emitter = resolve_inst_emitter(inst.__class__)(self)
            smem_workspace_nbytes = max(smem_workspace_nbytes, emitter.request_shared_workspace(inst))
        if smem_workspace_nbytes > 0:
            smem_workspace = SharedTensor.create(dtype=uint8, optional_layout=shared_row_major(smem_workspace_nbytes))
            self.allocate_shared_tensor(smem_workspace, nbytes=smem_workspace_nbytes)
            self.tensor2var[smem_workspace] = self.builder.declare(
                v=Var("temp_smem", type=void_p),
                init=dynamic_shared_memory(byte_offset=self.shared_value_allocator_addr[smem_workspace], dtype=uint8),
            )
            self.smem_workspace = smem_workspace

    def generate_launch_function(self, ir_module: IRModule, kernel_func: HidetFunction) -> IRModule:
        from hidet.ir.primitives.runtime import set_symbol_value_ptr
        from hidet.ir.stmt import SeqStmt
        from hidet.transforms.generate_launch_func import add_launch_func
        from hidet.transforms.instantiate_symbols import InstantiateSymbolsRewriter

        # add the launch function for the kernel function
        add_launch_func(ir_module, kernel_func=kernel_func)

        # instantiate the symbols in the kernel function, like __gmem_workspace, etc.
        instantiate_rewriter = InstantiateSymbolsRewriter()
        ir_module = instantiate_rewriter(ir_module)

        launch_func = ir_module.functions["launch"]
        launch_func = HidetFunction(
            name=kernel_func.name.removesuffix("_kernel"),
            params=launch_func.params,
            body=launch_func.body,
            ret_type=launch_func.ret_type,
            kind=launch_func.kind,
            attrs=launch_func.attrs,
        )

        if is_nvgpu():
            from hidet.ir.primitives.runtime import request_cuda_workspace

            request_workspace = request_cuda_workspace
        elif is_amdgpu():
            from hidet.ir.primitives.runtime import request_hip_workspace

            request_workspace = request_hip_workspace
        else:
            assert False

        # set the workspace
        sb = StmtBuilder()
        remap = {prog_param: launch_param for prog_param, launch_param in zip(self.function.params, launch_func.params)}
        if not (isinstance(self.gmem_allocated, Constant) and int(self.gmem_allocated) == 0):
            sb += set_symbol_value_ptr(
                self.GMEM_WORKSPACE_NAME,
                cast(
                    request_workspace(nbytes=rewrite(self.gmem_maximum_allocated, remap), require_clean=False), ~uint8
                ),
            )
        if not (isinstance(self.gmem_clean_allocated, Constant) and int(self.gmem_clean_allocated) == 0):
            sb += set_symbol_value_ptr(
                self.GMEM_CLEAN_WORKSPACE_NAME,
                cast(
                    request_workspace(nbytes=rewrite(self.gmem_clean_maximum_allocated, remap), require_clean=True),
                    ~uint8,
                ),
            )

        launch_func.body = SeqStmt([sb.finish(), launch_func.body])
        updated_ir_module = IRModule(
            functions={
                launch_func.name: launch_func,
                kernel_func.name: ir_module.functions[kernel_func.name],
            },
        )
        return updated_ir_module

    def visit_Function(self, func: Function) -> IRModule:
        assert func.metadata.analysis is not None, "Function analysis is required for code generation"
        # warmup printer
        self.printer(func)

        self._function = func

        self.check_emitter_existence()

        self._builder = FunctionBuilder(
            name=func.name + "_kernel",
            kind="cuda_kernel" if is_nvgpu() else "hip_kernel",
            label="",
            grid_dim=self._function.metadata.grid_blocks,
            cluster_dim=self._function.metadata.cluster_blocks
            if self._function.metadata.cluster_blocks != (1, 1, 1)
            else None,
            block_dim=func.metadata.num_warps * 32,
            dynamic_smem_bytes=None,
            min_blocks=None,
        )
        self.builder.extend_params(list(func.params))

        # init for_thread_group stack
        self.thread_groups.num_groups = [1]
        self.thread_groups.group_size = [func.metadata.num_warps * 32]
        self.thread_groups.current_worker = [threadIdx.x]

        # init pre-defined variables
        self.init_smem_workspace(func)

        # emit body
        self.visit(func.body)

        # check shared memory allocation and set dynamic shared memory size
        if self.smem_workspace:
            self.free_shared_value(self.smem_workspace)
            self.smem_workspace = None
        # if self.smem_allocator.allocated != 0:
        #     raise ValueError("Shared memory is not properly allocated/freed")
        if self.smem_allocator.maximum_allocated > get_current_target().properties.shared_memory_per_block:
            raise CodeGenerationFailed(
                "Request shared memory {} bytes, but the device only allows {} bytes.".format(
                    self.smem_allocator.maximum_allocated, get_current_target().properties.shared_memory_per_block
                )
            )
        if is_nvgpu():
            self.builder.attrs["cuda.dynamic_smem_bytes"] = self.smem_allocator.maximum_allocated
        elif is_amdgpu():
            self.builder.attrs["hip.dynamic_smem_bytes"] = self.smem_allocator.maximum_allocated
        else:
            assert False

        self.builder.finish_func()
        kernel_function = self.builder.get()
        ir_module = IRModule(functions={kernel_function.name: kernel_function})
        ir_module = self.generate_launch_function(ir_module, kernel_func=kernel_function)
        return ir_module

    def visit_SeqStmt(self, stmt: SeqStmt) -> None:
        for sub_stmt in stmt.seq:
            self.visit(sub_stmt)

    def visit_IfStmt(self, stmt: IfStmt) -> None:
        with self.builder.if_then(stmt.cond):
            self.visit(stmt.then_body)
        if stmt.else_body is not None:
            with self.builder.otherwise():
                self.visit(stmt.else_body)

    def visit_WhileStmt(self, stmt: WhileStmt) -> None:
        with self.builder.while_loop(stmt.cond):
            self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt) -> None:
        if stmt.unroll_factor is None:
            attr = "."
        elif stmt.unroll_factor == -1:
            attr = "u"
        else:
            attr = "u{}".format(stmt.unroll_factor)  # no unroll
        with self.builder.for_loop(stmt.iter_var, stmt.extent, attr=attr):
            self.visit(stmt.body)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt) -> None:
        prev_group_size = self.thread_groups.group_size[-1]
        group_size = prev_group_size // stmt.num_groups

        self.builder.declare(v=stmt.iter_var, init=threadIdx.x % prev_group_size // group_size)
        with self.builder.for_range(stmt.num_groups) as i:
            self.thread_groups.num_groups.append(stmt.num_groups)
            self.thread_groups.group_size.append(group_size)
            self.thread_groups.current_worker.append(threadIdx.x % group_size)
            with self.builder.if_then(stmt.iter_var == i):
                self.visit(stmt.body)
            self.thread_groups.group_size.pop()
            self.thread_groups.num_groups.pop()
            self.thread_groups.current_worker.pop()

            self.sync()

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        self.builder.declare(stmt.var, init=stmt.init)

    def visit_LetStmt(self, stmt: LetStmt) -> None:
        with self.builder.lets(bind_vars=stmt.bind_vars, values=stmt.bind_values):
            self.visit(stmt.body)

    def visit_AssignStmt(self, stmt: AssignStmt) -> None:
        self.builder.assign(stmt.var, value=stmt.value)

    def visit_TensorElemPtrStmt(self, stmt: TensorElemPtrStmt) -> None:
        if stmt.space in ["generic", "global"]:
            if stmt.space == "generic":
                assert isinstance(stmt.tensor, (GlobalTensor, SharedTensor))
            else:
                assert isinstance(stmt.tensor, GlobalTensor)
            ptr = self.tensor2var[stmt.tensor]
            if stmt.indices is not None:
                ptr = ptr + stmt.tensor.layout(*stmt.indices)
            self.builder.declare(stmt.ptr_var, ptr)
        elif stmt.space == "local":
            raise NotImplementedError("Local tensor pointer is not supported yet.")
        elif stmt.space == "shared":
            if not isinstance(stmt.tensor, SharedTensor):
                raise ValueError("Expected a SharedTensor for shared tensor pointer, got: {}".format(stmt.tensor))
            shared_tensor: SharedTensor = stmt.tensor
            addr = self.shared_tensor_shared_space_addr[shared_tensor]
            if stmt.indices is not None:
                addr = addr + shared_tensor.layout(*stmt.indices) * shared_tensor.dtype.nbytes
            self.builder.declare(stmt.ptr_var, addr)
        else:
            raise ValueError("Unknown tensor pointer space: {}".format(stmt.space))

    def visit_TensorElemValueStmt(self, stmt: TensorElemValueStmt) -> None:
        assert isinstance(stmt.tensor, (GlobalTensor, SharedTensor))
        self.builder.declare(stmt.var, init=self.tensor2var[stmt.tensor][stmt.tensor.layout(*stmt.indices)])

    def visit_ReturnStmt(self, stmt: ReturnStmt) -> None:
        self.builder.ret()

    def visit_InstStmt(self, stmt: InstStmt) -> None:
        self.visit(stmt.inst)

    def visit_Instruction(self, inst: Instruction) -> None:
        # insert a comment statement
        skip_comment_instructions = (PrintTensorInst, FormatPrintInst)
        if not isinstance(inst, skip_comment_instructions):
            self.builder.comment(str(self.printer(inst)), style="/*")

        # implement the vm instruction
        emitter_cls = resolve_inst_emitter(inst.__class__)
        if emitter_cls is None:
            raise RuntimeError("Can not resolve the emitter for instruction: {}".format(inst.__class__.__name__))
        emitter = emitter_cls(self)
        emitter.emit(inst)
        if inst.output is not None and inst.output not in self.tensor2var:
            raise RuntimeError(
                "The emitter for instruction {} does not set the mapping for its output tensor.".format(
                    inst.__class__.__name__
                )
            )
        self.builder.append(emitter.finish())


class ProgramCodegen(IRFunctor):
    def __call__(self, prog: Program) -> IRModule:
        return self.visit(prog)

    def visit_Program(self, prog: Program) -> IRModule:
        ir_module = IRModule()
        for name, func in prog.functions.items():
            func_codegen = Codegen()
            sub_ir_module = func_codegen(func)
            ir_module = merge_ir_modules([ir_module, sub_ir_module])

        # if there is only one public function, we copy it and generate a function named 'launch', which is used as the
        # entry point of the module
        public_functions = [func for func in ir_module.functions.values() if func.kind == "public"]

        if len(public_functions) == 1 and "launch" not in ir_module.functions:
            public_func: HidetFunction = public_functions[0]
            ir_module.add_function(
                name="launch",
                func=HidetFunction(
                    name="launch",
                    params=public_func.params,
                    body=public_func.body,
                    ret_type=public_func.ret_type,
                    kind=public_func.kind,
                    attrs=public_func.attrs,
                ),
            )
        return ir_module


def generate_ir_module(prog: Program) -> IRModule:
    """
    Generate an IRModule from a Program by compiling the statements and instructions to lower-level Hidet IR.

    Parameters
    ----------
    prog: Program
        The program to be compiled.

    Returns
    -------
    ir_module: IRModule
        The lower-level Hidet IR module.
    """
    codegen = ProgramCodegen()
    ir_module: IRModule = codegen(prog)

    # verify the IR module
    verify_ir_module(ir_module)

    return ir_module
