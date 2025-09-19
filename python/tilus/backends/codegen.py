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

from typing import Any, Callable, Dict, Optional, Sequence, Set, Type

from hidet.ir import FuncType
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var, tensor_pointer_var, tensor_var
from hidet.ir.func import Function as HidetFunction
from hidet.ir.module import IRModule
from hidet.ir.primitives import set_kernel_max_dynamic_smem_bytes
from hidet.ir.primitives.cuda.cluster import this_cluster
from hidet.ir.primitives.cuda.vars import blockIdx, dim3, threadIdx
from hidet.ir.stmt import DeclareScope, LaunchKernelStmt
from hidet.ir.stmt import Stmt as HidetStmt
from hidet.utils.doc import Doc, Text

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
    TensorElemPtrStmt,
    TensorElemValueStmt,
    ThreadGroupStmt,
    WhileStmt,
)
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor
from tilus.ir.tools import IRPrinter
from tilus.ir.tools.instruction_collector import collect_instructions
from tilus.ir.utils.normalize import normalize_dim3
from tilus.ir.utils.thread_group_stack import ThreadGroupStack
from tilus.target import Target, get_current_target, gpgpu_any, match_target


class InvalidInstruction(Exception):
    def __init__(self, inst):
        self.inst = inst


class CodeGenerationFailed(Exception):
    pass


class BaseInstEmitter(StmtBuilder):
    # inst -> emitter
    REGISTRY: Dict[Type[Instruction], Dict[Target, Type["BaseInstEmitter"]]] = {}

    def __init__(self, codegen: FunctionCodegen) -> None:
        super().__init__()
        self._codegen: FunctionCodegen = codegen

    def sync(self):
        from hidet.ir.primitives.cuda import syncthreads

        if self._codegen.thread_group_stack.stack_depth() == 1:  # all threads in the cta
            self.append(syncthreads())
        else:
            from hidet.ir.primitives.cuda.barrier import barrier_sync

            barrier = self._codegen.thread_group_stack.stack_depth() - 1
            count = self._codegen.thread_group_stack.group_size[-1]
            self.append(barrier_sync(barrier=barrier, count=count))

    def sync_reduce(self, value: Expr, op: str) -> Expr:
        if get_current_target().is_nvgpu():
            from hidet.ir.primitives.cuda.barrier import barrier_sync
            from hidet.ir.primitives.cuda.sync import syncthreads_and, syncthreads_or

            op2sync = {"and": syncthreads_and, "or": syncthreads_or}
            syncthreads_op = op2sync[op]

            if self._codegen.thread_group_stack.stack_depth() == 1:  # all threads in the cta
                return syncthreads_op(value)
            else:
                barrier = self._codegen.thread_group_stack.stack_depth() - 1
                count = self._codegen.thread_group_stack.group_size[-1]
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
                name = name if name else "tmem"
                var = self.declare_var(name, tp=int32)
            self.tensor2var[tensor] = var
            return var

    @property
    def current_thread(self) -> Expr:
        return self._codegen.current_thread

    @property
    def current_num_threads(self) -> int:
        return self._codegen.thread_group_stack.group_size[-1]

    @property
    def current_thread_group_begin(self) -> int:
        return self._codegen.thread_group_stack.thread_begin[-1]

    @property
    def current_thread_group_end(self) -> int:
        return self._codegen.thread_group_stack.thread_end[-1]

    @property
    def block_rank_in_cluster(self) -> Expr:
        return this_cluster.block_rank

    @property
    def blockIdx(self) -> dim3:
        return blockIdx

    @property
    def thread_groups(self):
        return self._codegen.thread_group_stack

    @property
    def tensor2var(self) -> Dict[Tensor, Var]:
        return self._codegen.tensor2var

    @property
    def shared_tensor_shared_space_addr(self):
        return self._codegen.shared_tensor_addr

    @property
    def num_warps(self) -> int:
        return self._codegen.function.metadata.num_warps

    @property
    def function(self) -> Function:
        return self._codegen.function

    @property
    def analysis(self):
        return self._codegen.function.metadata.analysis

    @property
    def kernel_params(self) -> Sequence[Var]:
        return self._codegen.builder.params

    def kernel_prepend(self, stmt: Expr | HidetStmt) -> None:
        """Prepend a statement to the kernel function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be prepended.
        """
        self._codegen.builder.scope_stack[-1].insert(0, stmt)

    def kernel_append(self, stmt: Expr | HidetStmt) -> None:
        """Append a statement to the kernel function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be appended.
        """
        self._codegen.builder.append(stmt)

    @property
    def host_builder(self) -> FunctionBuilder:
        return self._codegen.host_builder

    @property
    def builder(self) -> FunctionBuilder:
        return self._codegen.builder

    def append_extra_param(self, var: Var) -> None:
        """Append an extra parameter to the kernel function.

        This method marks a variable in the host function to be passed as an extra parameter to the kernel function.
        The `var` must be a variable defined in the host function. The kernel function can directly use the `var` in the
        kernel body after this method is called.
        """
        self._codegen.extra_params.append(var)

    def emit(self, inst: Instruction) -> None:
        raise NotImplementedError()


class BaseEmitContext:
    REGISTRY: list[Type[BaseEmitContext]] = []

    def __init__(self, codegen: FunctionCodegen):
        self.codegen = codegen

    @staticmethod
    def current() -> Any:
        raise NotImplementedError()

    def host_prepend(self, stmt: Expr | HidetStmt) -> None:
        """Prepend a statement to the host function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be prepended.
        """
        self.codegen.host_builder.scope_stack[-1].insert(0, stmt)

    def host_append(self, stmt: Expr | HidetStmt) -> None:
        """Append a statement to the host function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be appended.
        """
        self.codegen.host_builder.append(stmt)

    def kernel_prepend(self, stmt: Expr | HidetStmt) -> None:
        """Prepend a statement to the kernel function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be prepended.
        """
        self.codegen.builder.scope_stack[-1].insert(0, stmt)

    def kernel_append(self, stmt: Expr | HidetStmt) -> None:
        """Append a statement to the kernel function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be appended.
        """
        self.codegen.builder.append(stmt)

    def append_extra_param(self, var: Var) -> None:
        """Append an extra parameter to the kernel function.

        This method marks a variable in the host function to be passed as an extra parameter to the kernel function.
        The `var` must be a variable defined in the host function. The kernel function can directly use the `var` in the
        kernel body after this method is called.
        """
        self.codegen.extra_params.append(var)

    def initialize(self):
        """Initialize the context.

        This method is called before the codegen starts for all instructions.
        """
        pass

    def finalize(self):
        """Finalize the context.

        This method is called when the codegen is finished for all instructions.
        """
        pass


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


def register_emit_context(ctx_cls):
    if ctx_cls in BaseEmitContext.REGISTRY:
        raise ValueError(f"Emit context {ctx_cls} already registered")
    BaseEmitContext.REGISTRY.append(ctx_cls)
    return ctx_cls


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


class CommentInlinedIRPrinter(IRPrinter):
    def add_key_comment(self, key_hint: str, comment: str | Doc) -> Doc:
        return Text(comment) if isinstance(comment, str) else comment


class FunctionCodegen(IRFunctor):
    def __init__(self) -> None:
        super().__init__()
        self._function: Optional[Function] = None
        self._builder: Optional[FunctionBuilder] = None
        self._host_builder: Optional[FunctionBuilder] = None
        self.printer: IRPrinter = CommentInlinedIRPrinter()

        # extra parameters that computed in host function and passed to device kernel
        self.extra_params: list[Var] = []

        # tensor mapping
        self.tensor2var: Dict[Tensor, Var] = {}
        self.shared_tensor_addr: dict[SharedTensor, Var] = {}  # shared tensor to uint32 addr in shared space

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
        return self._current_thread

    def check_emitter_existence(self) -> None:
        failed_instructions: Set[str] = set()
        for inst in collect_instructions(self.function):
            if resolve_inst_emitter(inst.__class__) is None:
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

            # set max dynamic shared memory bytes if needed
            with self.host_builder.if_then(dynamic_shared_bytes > 48 * 1024):
                self.host_builder.append(set_kernel_max_dynamic_smem_bytes(func_var, dynamic_shared_bytes))

            # launch the kernel
            kernel_args = list(self.host_builder.params) + list(self.extra_params)
            self.host_builder.append(
                LaunchKernelStmt(
                    func_var=func_var,
                    args=kernel_args,
                    grid_dim=normalize_dim3(kernel_func.get_attr("cuda.grid_dim")),
                    cluster_dim=normalize_dim3(kernel_func.get_attr("cuda.cluster_dim", default=1)),
                    block_dim=normalize_dim3(kernel_func.get_attr("cuda.block_dim")),
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

        # create all contexts
        contexts = {cls: cls(self) for cls in BaseEmitContext.REGISTRY}

        for ctx in contexts.values():
            type(ctx)._current = ctx

        # initialize all contexts
        for ctx in contexts.values():
            ctx.initialize()

        # emit body
        self.visit(func.body)

        # finalize all contexts
        for ctx in contexts.values():
            ctx.finalize()

        for ctx in contexts.values():
            type(ctx)._current = None

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
        from tilus.backends.contexts.invariant_ctx import InvariantTrackingContext

        with self.builder.lets(bind_vars=stmt.bind_vars, values=stmt.bind_values):
            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                ctx: InvariantTrackingContext = InvariantTrackingContext.current()
                ctx.bind(bind_var, bind_value)
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
            addr = self.shared_tensor_addr[shared_tensor]
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
