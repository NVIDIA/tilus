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
from tilus.backends.context import BaseEmitContext
from tilus.backends.contexts.global_view_ctx import GlobalTensorViewContext
from tilus.backends.contexts.gmem_alloc_ctx import GlobalMemoryAllocationContext
from tilus.backends.contexts.invariant_ctx import InvariantTrackingContext
from tilus.backends.contexts.mbarrier_alloc_ctx import BarrierAllocContext
from tilus.backends.contexts.smem_alloc_ctx import SharedMemoryAllocationContext
from tilus.backends.contexts.sync_ctx import SyncContext
from tilus.backends.contexts.tcgen05_ctx import Tcgen05EmitContext


class EmitContexts:
    def __init__(self, codegen):
        from tilus.backends.codegen import FunctionCodegen

        assert isinstance(codegen, FunctionCodegen)
        self.codegen: FunctionCodegen = codegen

        self.global_view_ctx: GlobalTensorViewContext = GlobalTensorViewContext(codegen)
        self.gmem_alloc_ctx: GlobalMemoryAllocationContext = GlobalMemoryAllocationContext(codegen)
        self.invariant_ctx: InvariantTrackingContext = InvariantTrackingContext(codegen)
        self.smem_alloc_ctx: SharedMemoryAllocationContext = SharedMemoryAllocationContext(codegen)
        self.tcgen05_ctx: Tcgen05EmitContext = Tcgen05EmitContext(codegen)
        self.barrier_alloc_ctx: BarrierAllocContext = BarrierAllocContext(codegen)
        self.sync_ctx: SyncContext = SyncContext(codegen)

    def contexts(self) -> list[BaseEmitContext]:
        """Get all contexts as a list.

        Returns
        -------
        ret: list[BaseEmitContext]
            A list of all contexts.
        """
        return [ctx for ctx in self.__dict__.values() if isinstance(ctx, BaseEmitContext)]

    def initialize(self):
        """Initialize the context.

        This method is called before the codegen starts for all instructions.
        """
        for ctx in self.contexts():
            ctx.initialize()

    def finalize(self):
        """Finalize the context.

        This method is called when the codegen is finished for all instructions.
        """
        for ctx in reversed(self.contexts()):
            ctx.finalize()
