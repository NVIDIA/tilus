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

from typing import Dict, Optional, Set, Type

from hidet.ir import FuncType
from hidet.ir.builders import FunctionBuilder
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from hidet.ir.func import Function as HidetFunction
from hidet.ir.module import IRModule
from hidet.ir.primitives import set_kernel_max_dynamic_smem_bytes
from hidet.ir.primitives.cuda.vars import threadIdx
from hidet.ir.stmt import LaunchKernelStmt
from hidet.utils import prod
from hidet.utils.doc import Doc, Text

from tilus.backends.emitter import BaseInstEmitter
from tilus.extensions.hidet.ir.module import merge_ir_modules
from tilus.extensions.hidet.ir.tools.verifier import verify as verify_ir_module
from tilus.ir.func import Function
from tilus.ir.functors import IRFunctor
from tilus.ir.inst import Instruction
from tilus.ir.instructions import FormatPrintInst, PrintTensorInst
from tilus.ir.prog import Program
from tilus.ir.stmt import (
    AssignStmt,
    DeclareStmt,
    ForStmt,
    IfStmt,
    InstStmt,
    LetStmt,
    ReturnStmt,
    SeqStmt,
    TensorItemPtrStmt,
    TensorItemValueStmt,
    ThreadGroupStmt,
    WhileStmt,
)
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor
from tilus.ir.tools import IRPrinter
from tilus.ir.tools.instruction_collector import collect_instructions
from tilus.ir.utils.normalize import normalize_dim3
from tilus.ir.utils.thread_group_stack import ThreadGroupStack
from tilus.target import get_current_target, match_target


class InvalidInstruction(Exception):
    def __init__(self, inst):
        self.inst = inst


class CodeGenerationFailed(Exception):
    pass


class CommentInlinedIRPrinter(IRPrinter):
    def add_key_comment(self, key_hint: str, comment: str | Doc) -> Doc:
        return Text(comment) if isinstance(comment, str) else comment


class FunctionCodegen(IRFunctor):
    def __init__(self) -> None:
        super().__init__()
        from tilus.backends.contexts.contexts import EmitContexts

        self._function: Optional[Function] = None
        self._builder: Optional[FunctionBuilder] = None
        self._host_builder: Optional[FunctionBuilder] = None
        self.printer: IRPrinter = CommentInlinedIRPrinter()

        # extra parameters that computed in host function and passed to device kernel
        self.extra_params: list[Var] = []

        # tensor mapping
        self.tensor2var: Dict[Tensor, Var] = {}
        self.shared_tensor_addr: dict[SharedTensor, Var] = {}  # shared tensor to uint32 addr in shared space

        # codegen contexts
        self.contexts: EmitContexts = EmitContexts(self)

        # stacks of for_thread_groups
        self._current_thread: Optional[Var] = None
        self.thread_group_stack = ThreadGroupStack()

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

    @property
    def host_builder(self) -> FunctionBuilder:
        assert self._host_builder is not None
        return self._host_builder

    @property
    def current_thread(self):
        assert self._current_thread is not None
        return self._current_thread

    def resolve_inst_emitter(self, inst_cls: Type[Instruction]) -> Optional[Type[BaseInstEmitter]]:
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

    def check_emitter_existence(self) -> None:
        failed_instructions: Set[str] = set()
        for inst in collect_instructions(self.function):
            if self.resolve_inst_emitter(inst.__class__) is None:
                failed_instructions.add(inst.__class__.__name__)

        if failed_instructions:
            raise CodeGenerationFailed(
                "Failed to find emitter for the following instructions: \n{}".format("\n".join(failed_instructions))
            )

    def launch_kernel(self, kernel_func: HidetFunction) -> None:
        """Generate the host code to launch the kernel function."""
        if kernel_func.kind == "cuda_kernel":
            func_var = Var(hint=None, type=FuncType.from_func(kernel_func), name=kernel_func.name)
            dynamic_shared_bytes = kernel_func.get_attr("cuda.dynamic_smem_bytes", int32(0))
            assert isinstance(dynamic_shared_bytes, Expr)

            # set max dynamic shared memory bytes if needed
            with self.host_builder.if_then(dynamic_shared_bytes > 48 * 1024):
                self.host_builder.append(set_kernel_max_dynamic_smem_bytes(func_var, dynamic_shared_bytes))

            # launch the kernel
            kernel_args = list(self.host_builder.params) + list(self.extra_params)
            self.host_builder.append(
                LaunchKernelStmt(
                    func_var=func_var,
                    args=kernel_args,
                    grid_dim=normalize_dim3(kernel_func.get_attr("cuda.grid_dim")),  # type: ignore
                    cluster_dim=normalize_dim3(kernel_func.get_attr("cuda.cluster_dim", default=1)),  # type: ignore
                    block_dim=normalize_dim3(kernel_func.get_attr("cuda.block_dim")),  # type: ignore
                    shared_mem=dynamic_shared_bytes,
                    target="cuda",
                )
            )
        else:
            raise NotImplementedError("Only cuda kernel launch is supported now.")

    def visit_Function(self, func: Function) -> IRModule:
        if func.metadata.analysis is None:
            raise RuntimeError("Function analysis is required for code generation")
        self._function = func

        # create function builders for both device and host side
        self._builder = FunctionBuilder(
            name=func.name + "_kernel",
            kind="cuda_kernel" if get_current_target().is_nvgpu() else "hip_kernel",
            label="",
            grid_dim=self._function.metadata.grid_blocks,
            cluster_dim=self._function.metadata.cluster_blocks
            if self._function.metadata.cluster_blocks != (1, 1, 1)
            else None,
            block_dim=func.metadata.num_warps * 32,
            dynamic_smem_bytes=None,
            min_blocks=None,
        )
        self._host_builder = FunctionBuilder(
            name=func.name,
            kind="public",
            label="",
        )
        self.builder.extend_params(list(func.params))
        self.host_builder.extend_params(list(func.params))

        # warmup printer
        self.printer(func)

        # make sure all instructions have matched emitters
        self.check_emitter_existence()

        # initialize for_thread_group stack
        self._current_thread = threadIdx.x
        self.thread_group_stack.push(group_index=0, group_size=func.metadata.num_warps * 32)

        # initialize all contexts
        self.contexts.initialize()

        # emit body
        self.visit(func.body)

        # finalize all contexts
        self.contexts.finalize()

        # create the kernel function
        self.builder.extend_params(self.extra_params)
        self.builder.finish_func()
        kernel_function: HidetFunction = self.builder.get()

        # launch the kernel function on the host side
        self.launch_kernel(kernel_function)

        # create the host function
        self.host_builder.finish_func()
        host_function: HidetFunction = self.host_builder.get()

        # create the IR module contains both host and device functions
        ir_module = IRModule(functions={kernel_function.name: kernel_function, host_function.name: host_function})

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

    def visit_ThreadGroupStmt(self, stmt: ThreadGroupStmt) -> None:
        # check the validity of the thread group
        parent_group_size = self.thread_group_stack.group_size[-1]
        if parent_group_size % stmt.group_size != 0:
            raise ValueError("group_size must be a divisor of the parent group_size")
        num_groups = parent_group_size // stmt.group_size
        if stmt.group_index < 0 or stmt.group_index >= num_groups:
            raise ValueError(
                "group_index must be in [0, num_groups), got group_index={}, num_groups={}".format(
                    stmt.group_index, num_groups
                )
            )

        self.builder.comment(
            "ThreadGroup(group_index={}, group_size={})".format(stmt.group_index, stmt.group_size),
            style="/*",
        )
        with self.builder.if_then(cond=self.current_thread // stmt.group_size == stmt.group_index):
            tid = self.builder.declare_var("tid", tp=int32, init=self.current_thread % stmt.group_size)
            old_thread = self._current_thread
            self._current_thread = tid
            self.thread_group_stack.push(
                group_index=stmt.group_index,
                group_size=stmt.group_size,
            )
            self.visit(stmt.body)
            self._current_thread = old_thread
            self.thread_group_stack.pop()

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        self.builder.declare(stmt.var, init=stmt.init)

    def visit_LetStmt(self, stmt: LetStmt) -> None:
        with self.builder.lets(bind_vars=stmt.bind_vars, values=stmt.bind_values):
            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                ctx = self.contexts.invariant_ctx
                ctx.bind(bind_var, bind_value)
            self.visit(stmt.body)

    def visit_AssignStmt(self, stmt: AssignStmt) -> None:
        self.builder.assign(stmt.var, value=stmt.value)

    def visit_TensorItemPtrStmt(self, stmt: TensorItemPtrStmt) -> None:
        if stmt.space in ["generic", "global"]:
            if stmt.space == "generic":
                assert isinstance(stmt.tensor, (GlobalTensor, SharedTensor))
            else:
                assert isinstance(stmt.tensor, GlobalTensor)
            ptr = self.tensor2var[stmt.tensor]
            indices = [int32.zero for _ in range(len(stmt.tensor.shape))]
            ptr = ptr + stmt.tensor.layout(*indices)
            self.builder.declare(stmt.ptr_var, ptr)
        elif stmt.space == "local":
            raise NotImplementedError("Local tensor pointer is not supported yet.")
        elif stmt.space == "shared":
            if not isinstance(stmt.tensor, SharedTensor):
                raise ValueError("Expected a SharedTensor for shared tensor pointer, got: {}".format(stmt.tensor))
            shared_tensor: SharedTensor = stmt.tensor
            addr = self.shared_tensor_addr[shared_tensor]
            indices = [int32.zero for _ in range(len(shared_tensor.shape))]
            addr = addr + shared_tensor.layout(*indices) * shared_tensor.dtype.nbytes
            self.builder.declare(stmt.ptr_var, addr)
        else:
            raise ValueError("Unknown tensor pointer space: {}".format(stmt.space))

    def visit_TensorItemValueStmt(self, stmt: TensorItemValueStmt) -> None:
        if isinstance(stmt.tensor, (GlobalTensor, SharedTensor)):
            indices = [int32.zero for _ in range(len(stmt.tensor.shape))]
            value = self.tensor2var[stmt.tensor][stmt.tensor.layout(*indices)]
        elif isinstance(stmt.tensor, RegisterTensor):
            if not prod(stmt.tensor.shape) == 1:
                raise ValueError(
                    "Indexing into the a non-scalar register tensor is not supported yet: {}".format(stmt.tensor)
                )
            value = self.tensor2var[stmt.tensor][0]
        else:
            raise ValueError(
                "Only global tensor, shared tensor, and scalar register tensor are supported for indexing, got {}".format(
                    stmt.tensor
                )
            )
        self.builder.declare(stmt.var, init=value)

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
        emitter_cls = self.resolve_inst_emitter(inst.__class__)
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
            func_codegen = FunctionCodegen()
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
