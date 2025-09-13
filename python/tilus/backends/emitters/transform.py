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
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import RepeatInst, RepeatInterleaveInst, SqueezeInst, TransposeInst, UnsqueezeInst


@register_emitter(RepeatInst)
class RepeatInstEmitter(BaseInstEmitter):
    def emit(self, inst: RepeatInst) -> None:
        src = inst.register_input
        dst = inst.register_output

        src_buf = self.tensor2var[src]
        dst_buf = self.get_or_allocate_var(dst)

        # check the layout is valid
        assert len(src.shape) == len(dst.shape)
        repeats = [a // b for a, b in zip(dst.shape, src.shape)]
        # t1 = time.time()
        # for worker in range(self.current_num_workers):
        #     for i in range(dst.local_size):
        #         global_indices = dst.layout.local2global(as_expr(i), as_expr(worker))
        #         global_indices = [a // b for a, b in zip(global_indices, repeats)]
        #         is_valid = src.layout.is_valid(global_indices, worker=as_expr(worker))
        #         if not is_valid:
        #             raise RuntimeError(
        #                 f"Invalid global index {global_indices} for worker {worker} in RepeatInst'\n"
        #                 f"src_layout: {src.layout} \n"
        #                 f"dst_layout: {dst.layout}"
        #             )
        # t2 = time.time()
        # print(f"Check layout time: {t2 - t1:.4f} seconds")

        # emit the code
        with self.for_range(dst.local_size, attr="u") as local:
            global_indices = dst.layout.get_global(local_index=local, spatial_index=self.current_thread)
            global_indices = [a // b for a, b in zip(global_indices, repeats)]
            src_local = src.layout.get_local(global_indices)
            self.buffer_store(dst_buf, indices=[local], value=src_buf[src_local])


@register_emitter(RepeatInterleaveInst)
class RepeatInterleaveInstEmitter(BaseInstEmitter):
    def emit(self, inst: RepeatInterleaveInst) -> None:
        src = inst.register_input
        dst = inst.register_output

        src_buf = self.tensor2var[src]
        dst_buf = self.get_or_allocate_var(dst)

        # check the layout is valid
        assert len(src.shape) == len(dst.shape)
        # t1 = time.time()
        # for worker in range(self.current_num_workers):
        #     for i in range(dst.local_size):
        #         global_indices = dst.layout.local2global(as_expr(i), as_expr(worker))
        #         global_indices = [a % b for a, b in zip(global_indices, src.shape)]
        #         is_valid = src.layout.is_valid(global_indices, worker=as_expr(worker))
        #         if not is_valid:
        #             raise RuntimeError(
        #                 f"Invalid global index {global_indices} for worker {worker} in RepeatInst'\n"
        #                 f"src_layout: {src.layout} \n"
        #                 f"dst_layout: {dst.layout}"
        #             )
        # t2 = time.time()
        # print(f"Check layout time: {t2 - t1:.4f} seconds")

        # emit the code
        with self.for_range(dst.local_size, attr="u") as local:
            global_indices = dst.layout.get_global(local_index=local, spatial_index=self.current_thread)
            global_indices = [a % b for a, b in zip(global_indices, src.shape)]
            src_local = src.layout.get_local(global_indices)
            self.buffer_store(dst_buf, indices=[local], value=src_buf[src_local])


@register_emitter(UnsqueezeInst)
@register_emitter(SqueezeInst)
class SqueezeUnsqueezeInstEmitter(BaseInstEmitter):
    def emit(self, inst: SqueezeInst) -> None:
        src = inst.register_input
        dst = inst.register_output

        src_buf = self.tensor2var[src]
        dst_buf = self.get_or_allocate_var(dst)

        # emit the code
        with self.for_range(dst.local_size, attr="u") as local:
            self.buffer_store(dst_buf, indices=[local], value=src_buf[local])


@register_emitter(TransposeInst)
class TransposeInstEmitter(BaseInstEmitter):
    def emit(self, inst: TransposeInst) -> None:
        src = inst.register_input
        dst = inst.register_output

        src_buf = self.tensor2var[src]
        dst_buf = self.get_or_allocate_var(dst)

        # emit the code
        with self.for_range(dst.local_size, attr="u") as local:
            self.buffer_store(dst_buf, indices=[local], value=src_buf[local])
