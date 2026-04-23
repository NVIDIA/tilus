# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tilus.backends.emitter import BaseInstEmitter
from tilus.hidet.ir import FuncType
from tilus.hidet.ir.builders import FunctionBuilder
from tilus.hidet.ir.dtypes import int32, uint32
from tilus.hidet.ir.expr import Constant, Var, logical_and
from tilus.hidet.ir.func import Function as HidetFunction
from tilus.hidet.ir.module import IRModule, merge_ir_modules
from tilus.hidet.ir.primitives import set_kernel_max_dynamic_smem_bytes
from tilus.hidet.ir.primitives.cuda.elect import elect_one_sync, shfl_sync_i32
from tilus.hidet.ir.primitives.cuda.vars import threadIdx
from tilus.hidet.ir.stmt import LaunchKernelStmt
from tilus.hidet.ir.tools.verifier import verify as verify_ir_module
from tilus.hidet.utils import prod
from tilus.hidet.utils.doc import Doc, Text
from tilus.ir.func import Function
from tilus.ir.functors import IRFunctor
from tilus.ir.inst import Instruction
from tilus.ir.instructions import FormatPrintInst, PrintTensorInst
from tilus.ir.prog import Program
from tilus.ir.stmt import (
    AssignStmt,
    DeclareStmt,
    EvaluateStmt,
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
from tilus.target import get_current_target, match_target, nvgpu_sm90
from tilus.utils import is_power_of_two


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
        failed_instructions: Set[Type[Instruction]] = set()
        for inst in collect_instructions(self.function):
            if self.resolve_inst_emitter(inst.__class__) is None:
                failed_instructions.add(inst.__class__)

        if failed_instructions:
            rows = [f"Failed to find emitter for the following instructions (target: {get_current_target()}):"]
            required_targets: list[str] = []
            for inst_cls in failed_instructions:
                for registry_inst_cls, emitter_classes in BaseInstEmitter.REGISTRY.items():
                    if issubclass(inst_cls, registry_inst_cls):
                        required_targets.extend([str(target) for target in emitter_classes.keys()])
                        break
                if not required_targets:
                    rows.append(f"  - {inst_cls.__name__} (no registered emitters)")
                else:
                    rows.append(f"  - {inst_cls.__name__} (registered targets: {', '.join(required_targets)})")
            raise CodeGenerationFailed("\n".join(rows))

    def launch_kernel(self, kernel_func: HidetFunction) -> None:
        """Generate the host code to launch the kernel function."""
        if kernel_func.kind == "cuda_kernel":
            func_var = Var(name=kernel_func.name, type=FuncType.from_func(kernel_func))
            dynamic_shared_bytes = (
                kernel_func.attrs.dynamic_smem_bytes if kernel_func.attrs.dynamic_smem_bytes is not None else int32(0)
            )
            assert isinstance(dynamic_shared_bytes, Constant | int), (
                "dynamic shared memory bytes must be a constant integer"
            )
            if int(dynamic_shared_bytes) > get_current_target().properties.shared_memory_per_block:
                raise RuntimeError(
                    f"Requested dynamic shared memory bytes {int(dynamic_shared_bytes)} exceed the device limit "
                    f"{get_current_target().properties.shared_memory_per_block} for target {get_current_target()}."
                )

            # set max dynamic shared memory bytes if needed
            with self.host_builder.if_then(dynamic_shared_bytes > 48 * 1024):
                self.host_builder.append(set_kernel_max_dynamic_smem_bytes(func_var, dynamic_shared_bytes))

            # launch the kernel
            kernel_args = list(self.host_builder.params) + list(self.extra_params)
            cluster_dim = kernel_func.attrs.cluster_dim if kernel_func.attrs.cluster_dim is not None else 1
            from tilus.hidet.ir.expr import convert as _convert_expr  # noqa: PLC0415

            def _to_expr_triple(triple):
                return tuple(_convert_expr(v) for v in triple)

            self.host_builder.append(
                LaunchKernelStmt.create(
                    func_var=func_var,
                    args=[_convert_expr(a) for a in kernel_args],
                    grid_dim=_to_expr_triple(normalize_dim3(kernel_func.attrs.grid_dim)),  # type: ignore
                    cluster_dim=_to_expr_triple(normalize_dim3(cluster_dim)),  # type: ignore
                    block_dim=_to_expr_triple(normalize_dim3(kernel_func.attrs.block_dim)),  # type: ignore
                    shared_mem_bytes=int32(dynamic_shared_bytes),
                    target="cuda",
                )
            )
        else:
            raise NotImplementedError("Only cuda kernel launch is supported now.")

    def visit_Function(self, func: Function) -> IRModule:
        if func.metadata.analysis is None:
            raise RuntimeError("Function analysis is required for code generation")
        self._function = func

        current_target = get_current_target()
        if current_target.supports(nvgpu_sm90):
            cluster_blocks = self._function.metadata.cluster_blocks
        else:
            if self._function.metadata.cluster_blocks != (1, 1, 1):
                raise RuntimeError(
                    f"Target {current_target} does not support cluster blocks, but function {func.name} has cluster blocks {self._function.metadata.cluster_blocks}"
                )
            cluster_blocks = None

        # create function builders for both device and host side
        self._builder = FunctionBuilder(
            name=func.name + "_kernel",
            kind="cuda_kernel" if get_current_target().is_nvgpu() else "hip_kernel",
            grid_dim=self._function.metadata.grid_blocks,
            cluster_dim=cluster_blocks,
            block_dim=func.metadata.num_warps * 32,
            dynamic_smem_bytes=None,
            min_blocks=None,
        )
        self._host_builder = FunctionBuilder(
            name=func.name,
            kind="public",
        )
        self.builder.extend_params(list(func.params))
        self.host_builder.extend_params(list(func.params))

        # warmup printer
        self.printer(func)

        # make sure all instructions have matched emitters
        self.check_emitter_existence()

        # initialize for_thread_group stack
        self._current_thread = threadIdx.x
        self.thread_group_stack.push(thread_begin=0, num_threads=func.metadata.num_warps * 32)

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

    def _elect_any_cond(self, num_threads: int, parent_num_threads: int) -> tuple:
        """Build the condition and tid_value for an elect-any ThreadGroupStmt.

        Returns (cond, tid_value) where cond is None if all threads should execute.
        """
        if num_threads == parent_num_threads:
            # All threads — no condition needed
            return None, self.current_thread
        elif num_threads == 1:
            if parent_num_threads % 32 != 0 or not get_current_target().supports(nvgpu_sm90):
                # Non-warp-aligned parent or pre-sm_90: fallback to thread 0
                return self.current_thread == 0, Constant(0, int32)
            elif parent_num_threads == 32:
                # Single warp: use elect.sync directly
                return elect_one_sync(), Constant(0, int32)
            else:
                # Multi-warp (parent % 32 == 0): pick warp 0, then elect one thread within it.
                # Use shfl_sync to move warp_id into a uniform register — threadIdx / 32 is
                # mathematically uniform within a warp but the compiler may not recognize it.
                # shfl_sync forces the value through the warp shuffle unit, producing a
                # guaranteed uniform value.
                warp_id = self.current_thread // 32
                uniform_warp_id = shfl_sync_i32(Constant(0xFFFFFFFF, uint32), warp_id, Constant(0, int32))
                return (uniform_warp_id == 0) & elect_one_sync(), Constant(0, int32)
        elif num_threads % 32 == 0 and parent_num_threads % 32 == 0:
            # Warp-aligned multi-thread: use shfl_sync for uniform warp-level predicate
            group_id = self.current_thread // num_threads
            elected_group = shfl_sync_i32(Constant(0xFFFFFFFF, uint32), group_id, Constant(0, int32))
            return group_id == elected_group, self.current_thread % num_threads
        else:
            # Sub-warp or non-aligned: fallback to picking the first group
            return self.current_thread < num_threads, self.current_thread % num_threads

    def visit_ThreadGroupStmt(self, stmt: ThreadGroupStmt) -> None:
        parent_thread_begin = self.thread_group_stack.thread_begin[-1]
        parent_num_threads = self.thread_group_stack.num_threads[-1]

        if stmt.thread_begin == -1:
            # Elect-any mode
            assert stmt.num_threads <= parent_num_threads
            self.builder.comment(
                "ThreadGroup(thread_begin=elect_any, num_threads={})".format(stmt.num_threads),
                style="/*",
            )
            cond, tid_value = self._elect_any_cond(stmt.num_threads, parent_num_threads)
            effective_thread_begin = 0
        else:
            # Fixed thread_begin mode
            assert 0 <= stmt.thread_begin and stmt.thread_begin + stmt.num_threads <= parent_num_threads
            self.builder.comment(
                "ThreadGroup(thread_begin={}, thread_end={}, num_threads={})".format(
                    stmt.thread_begin, stmt.thread_begin + stmt.num_threads, stmt.num_threads
                ),
                style="/*",
            )
            is_warp_aligned = (
                stmt.num_threads >= 32
                and stmt.num_threads % 32 == 0
                and stmt.thread_begin % 32 == 0
                and parent_thread_begin % 32 == 0
                and is_power_of_two(stmt.num_threads)
                and is_power_of_two(parent_num_threads)
            )
            if is_warp_aligned:
                # Warp-aligned: use shfl_sync for uniform predicate
                target_group = stmt.thread_begin // stmt.num_threads
                group_id = self.current_thread // stmt.num_threads
                elected_group = shfl_sync_i32(Constant(0xFFFFFFFF, uint32), group_id, Constant(0, int32))
                cond = elected_group == target_group
                tid_value = self.current_thread % stmt.num_threads
            elif (
                is_power_of_two(stmt.num_threads)
                and is_power_of_two(parent_num_threads)
                and stmt.thread_begin % stmt.num_threads == 0
            ):
                cond = (self.current_thread // stmt.num_threads) == (stmt.thread_begin // stmt.num_threads)
                tid_value = self.current_thread % stmt.num_threads
            else:
                cond = logical_and(
                    stmt.thread_begin <= self.current_thread,
                    self.current_thread < stmt.thread_begin + stmt.num_threads,
                )
                tid_value = self.current_thread - stmt.thread_begin
            effective_thread_begin = stmt.thread_begin

        # Emit the body
        if cond is None:
            # All threads — no condition
            tid = self.builder.declare_var("tid", tp=int32, init=tid_value)
            old_thread = self._current_thread
            self._current_thread = tid
            self.thread_group_stack.push(thread_begin=effective_thread_begin, num_threads=stmt.num_threads)
            self.visit(stmt.body)
            self._current_thread = old_thread
            self.thread_group_stack.pop()
        else:
            with self.builder.if_then(cond=cond):
                tid = self.builder.declare_var("tid", tp=int32, init=tid_value)
                old_thread = self._current_thread
                self._current_thread = tid
                self.thread_group_stack.push(thread_begin=effective_thread_begin, num_threads=stmt.num_threads)
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

    def visit_EvaluateStmt(self, stmt: EvaluateStmt) -> None:
        if stmt.pred is not None:
            with self.builder.if_then(stmt.pred):
                self.builder.append(stmt.expr)
        else:
            self.builder.append(stmt.expr)

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

    def visit_BreakStmt(self, stmt):
        self.builder.brk()

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

        # The 'launch' entry point is handled by GenerateLaunchFuncPass which runs later
        # in the optimization pipeline.
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
    import os
    if not os.environ.get("TILUS_SKIP_VERIFY"):
        verify_ir_module(ir_module)

    return ir_module
