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
from typing import Optional

from hidet.ir import StmtBuilder
from hidet.ir.dtypes import uint8
from hidet.ir.expr import Expr, Var
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory

from tilus.backends.codegen import BaseEmitContext, FunctionCodegen, register_emit_context
from tilus.ir.tensor import SharedTensor
from tilus.target import get_current_target


class SharedMemoryAllocator:
    def __init__(self) -> None:
        self.free_slots: list[tuple[int, int]] = [(0, (1 << 32) - 1)]
        self.addr2nbytes: dict[int, int] = {}
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


@register_emit_context
class SharedMemoryAllocationContext(BaseEmitContext):
    def __init__(self, codegen: FunctionCodegen):
        super().__init__(codegen)
        # shared memory allocator
        self.smem_allocator: SharedMemoryAllocator = SharedMemoryAllocator()

        # mapping from shared value to the address in shared memory allocator for all allocated shared values
        self.shared_value_allocator_addr: dict[SharedTensor, int] = {}

        # maximum shared workspace bytes requested by all instructions
        self.shared_workspace_var: Optional[Var] = None
        self.shared_workspace_bytes: int = 0

    def allocate_shared_tensor(self, tensor: SharedTensor, nbytes: int) -> int:
        addr: int = self.smem_allocator.allocate(nbytes)
        assert tensor not in self.shared_value_allocator_addr
        self.shared_value_allocator_addr[tensor] = addr
        return addr

    def free_shared_tensor(self, tensor: SharedTensor) -> None:
        assert tensor in self.shared_value_allocator_addr
        self.smem_allocator.free(addr=self.shared_value_allocator_addr[tensor])
        del self.shared_value_allocator_addr[tensor]

    def request_shared_workspace(self, nbytes: int) -> Expr:
        if self.shared_workspace_var is None:
            self.shared_workspace_var = Var("shared_workspace", type=~uint8)
            self.shared_workspace_bytes = nbytes
        else:
            self.shared_workspace_bytes = max(self.shared_workspace_bytes, nbytes)
        return self.shared_workspace_var

    def finalize(self):
        maximum_allocated = self.smem_allocator.maximum_allocated
        target = get_current_target()

        # define the shared workspace variable if needed
        if self.shared_workspace_var is not None:
            workspace_offset = (maximum_allocated + 127) // 128 * 128  # align to 128 bytes
            maximum_allocated += self.shared_workspace_bytes
            sb = StmtBuilder()
            sb.declare(self.shared_workspace_var, init=dynamic_shared_memory(workspace_offset, dtype=uint8))
            self.prepend_kernel(sb.finish())

        # set the dynamic shared memory size
        if target.is_nvgpu():
            self.codegen.builder.extend_attrs({"cuda.dynamic_smem_bytes": maximum_allocated})
        else:
            raise NotImplementedError()
