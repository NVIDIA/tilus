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
from hidet.ir.dtypes import int32
from hidet.ir.expr import Var
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.type import tensor_pointer_type

from tilus.backends.codegen import BaseInstEmitter, BaseEmitContext, register_emitter, FunctionCodegen
from tilus.ir.instructions import AllocateSharedInst, FreeSharedInst, SharedSliceInst
from tilus.ir.tensor import SharedTensor


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


class SharedMemoryAllocationContext(BaseEmitContext):
    def __init__(self, codegen: FunctionCodegen):
        super().__init__(codegen)

        # shared memory workspace
        self.smem_workspace: Optional[SharedTensor] = None


        # shared memory allocator
        self.smem_allocator: SharedMemoryAllocator = SharedMemoryAllocator()

        # mapping from shared value to the address in shared memory allocator for all allocated shared values
        self.shared_value_allocator_addr: dict[SharedTensor, int] = {}


    def allocate_shared_tensor(self, tensor: SharedTensor, nbytes: int) -> int:
        addr: int = self.smem_allocator.allocate(nbytes)
        assert tensor not in self.shared_value_allocator_addr
        self.shared_value_allocator_addr[tensor] = addr
        return addr


    def free_shared_tensor(self, tensor: SharedTensor) -> None:
        assert tensor in self.shared_value_allocator_addr
        self.smem_allocator.free(addr=self.shared_value_allocator_addr[tensor])
        del self.shared_value_allocator_addr[tensor]

    def init_smem_workspace(self, program: Function) -> None:
        smem_workspace_nbytes: int = 0
        for inst in collect_instructions(program):  # todo: add this to emiter
            # smem_workspace_nbytes = max(smem_workspace_nbytes, inst.request_shared_workspace())
            emitter = resolve_inst_emitter(inst.__class__)(self)
            smem_workspace_nbytes = max(smem_workspace_nbytes, emitter.request_shared_workspace(inst))
        if smem_workspace_nbytes > 0:
            smem_workspace = SharedTensor.create(dtype=uint8, optional_layout=shared_row_major([smem_workspace_nbytes]))
            self.allocate_shared_tensor(smem_workspace, nbytes=smem_workspace_nbytes)
            self.tensor2var[smem_workspace] = self.builder.declare(
                v=Var("temp_smem", type=void_p),
                init=dynamic_shared_memory(byte_offset=self.shared_value_allocator_addr[smem_workspace], dtype=uint8),
            )
            self.smem_workspace = smem_workspace

        # # init pre-defined variables
        # self.init_smem_workspace(func)

        # # check shared memory allocation and set dynamic shared memory size
        # if self.smem_workspace:
        #     self.free_shared_tensor(self.smem_workspace)
        #     self.smem_workspace = None
        # # if self.smem_allocator.allocated != 0:
        # #     raise ValueError("Shared memory is not properly allocated/freed")
        # if self.smem_allocator.maximum_allocated > get_current_target().properties.shared_memory_per_block:
        #     raise CodeGenerationFailed(
        #         "Request shared memory {} bytes, but the device only allows {} bytes.".format(
        #             self.smem_allocator.maximum_allocated, get_current_target().properties.shared_memory_per_block
        #         )
        #     )
        # if is_nvgpu():
        #     self.builder.attrs["cuda.dynamic_smem_bytes"] = self.smem_allocator.maximum_allocated
        # elif is_amdgpu():
        #     self.builder.attrs["hip.dynamic_smem_bytes"] = self.smem_allocator.maximum_allocated
        # else:
        #     assert False
