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

from typing import Callable, Dict, Optional, Sequence, Type

from hidet.ir.builders import FunctionBuilder
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var, tensor_pointer_var, tensor_var
from hidet.ir.primitives.cuda import syncthreads
from hidet.ir.primitives.cuda.vars import blockIdx, dim3
from hidet.ir.stmt import DeclareScope
from hidet.ir.stmt import Stmt as HidetStmt

from tilus.extensions.hidet.ir.builders.stmt_builder import TypedStmtBuilder as StmtBuilder
from tilus.ir.func import Function
from tilus.ir.inst import Instruction
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor
from tilus.target import Target, get_current_target, gpgpu_any


class BaseInstEmitter(StmtBuilder):
    # inst -> emitter
    REGISTRY: Dict[Type[Instruction], Dict[Target, Type["BaseInstEmitter"]]] = {}

    def __init__(self, codegen):
        super().__init__()

        from tilus.backends.codegen import FunctionCodegen

        assert isinstance(codegen, FunctionCodegen)
        self._codegen: FunctionCodegen = codegen

    def sync(self):
        if self._codegen.thread_group_stack.stack_depth() == 1:  # all threads in the cta
            self.append(syncthreads())
        else:
            self.append(self.contexts.sync_ctx.sync())

    def sync_reduce(self, value: Expr, op: str) -> Expr:
        if get_current_target().is_nvgpu():
            from hidet.ir.primitives.cuda.sync import syncthreads_and, syncthreads_or

            op2sync = {"and": syncthreads_and, "or": syncthreads_or}
            syncthreads_op = op2sync[op]

            if self._codegen.thread_group_stack.stack_depth() == 1:  # all threads in the cta
                return syncthreads_op(value)
            else:
                raise NotImplementedError("sync_reduce in sub-cta thread groups is not implemented")
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
        if self._codegen.current_thread is None:
            raise RuntimeError("Current thread is not set")
        return self._codegen.current_thread

    @property
    def current_num_threads(self) -> int:
        return self._codegen.thread_group_stack.num_threads[-1]

    @property
    def current_thread_group_begin(self) -> int:
        return self._codegen.thread_group_stack.thread_begin[-1]

    @property
    def current_thread_group_end(self) -> int:
        return self._codegen.thread_group_stack.thread_end[-1]

    @property
    def block_rank_in_cluster(self) -> Expr:
        from tilus.extensions.hidet.ir.primitives.cuda.cluster import block_rank_in_cluster

        return block_rank_in_cluster()

    @property
    def blocks_per_cluster(self) -> Expr:
        from tilus.extensions.hidet.ir.primitives.cuda.cluster import cluster_blocks

        return cluster_blocks()

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

    @property
    def contexts(self):
        return self._codegen.contexts

    def append_extra_param(self, var: Var) -> None:
        """Append an extra parameter to the kernel function.

        This method marks a variable in the host function to be passed as an extra parameter to the kernel function.
        The `var` must be a variable defined in the host function. The kernel function can directly use the `var` in the
        kernel body after this method is called.
        """
        self._codegen.extra_params.append(var)

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
            msg = [
                f"Emitter for instruction {inst_cls} and target {target} already exists",
                f" Registered emitter: {BaseInstEmitter.REGISTRY[inst_cls][target].__module__}",
                f"Registering emitter: {emitter_cls.__module__}",
            ]
            raise ValueError("\n".join(msg))

        BaseInstEmitter.REGISTRY[inst_cls][target] = emitter_cls
        return emitter_cls

    return decorator
