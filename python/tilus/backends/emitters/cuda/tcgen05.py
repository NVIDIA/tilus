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
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import tcgen05_relinquish_alloc_permit
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05RelinquishAllocPermitInst
from tilus.target import nvgpu_sm100


@register_emitter(Tcgen05RelinquishAllocPermitInst, target=nvgpu_sm100)
class Tcgen05RelinquishAllocPermitEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05RelinquishAllocPermitInst) -> None:

        self.append(tcgen05_relinquish_alloc_permit(inst.cta_group))
