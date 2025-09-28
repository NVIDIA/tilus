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
from tilus.extensions.hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    Tcgen05CommitMulticastKind,
    Tcgen05CtaGroupKind,
    tcgen05_commit,
)
from tilus.ir.instructions.cuda.tmem import (
    Tcgen05CommitInst,
)
from tilus.target import nvgpu_sm100


@register_emitter(Tcgen05CommitInst, target=nvgpu_sm100)
class TMemoryCommitEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05CommitInst) -> None:
        with self.if_then(self.current_thread == 0):
            self.append(
                tcgen05_commit(
                    mbarrier=cvta_generic_to_shared(inst.mbarrier),
                    cta_mask=inst.cta_mask,
                    cta_group=Tcgen05CtaGroupKind.CTA_1,
                    multicast=Tcgen05CommitMulticastKind.NONE,
                )
            )
