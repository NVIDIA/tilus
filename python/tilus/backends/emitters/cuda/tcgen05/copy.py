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
from tilus.ir.instructions.cuda.tmem import (
    TMemoryCommitInst,
    TMemoryCopyInst,
)
from tilus.target import nvgpu_sm100

#    tmem addr: 0xAAAABBBB where AAAA is the lane index and BBBB is the column index
#   lane index: 0x0000 to 0x007F
# column index: 0x0000 to 0x01FF
LANE_STRIDE = 0x00010000
COLUMN_STRIDE = 0x00000001


@register_emitter(TMemoryCopyInst, target=nvgpu_sm100)
class TMemoryCopyEmitter(BaseInstEmitter):
    def emit(self, inst: TMemoryCopyInst) -> None:
        raise NotImplementedError("TMemoryCopyInst is not supported yet")


@register_emitter(TMemoryCommitInst, target=nvgpu_sm100)
class TMemoryCommitEmitter(BaseInstEmitter):
    def emit(self, inst: TMemoryCommitInst) -> None:
        raise NotImplementedError("TMemoryCommitInst is not supported yet")
