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
from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.primitives.cuda.setmaxnreg import setmaxnreg
from tilus.ir.instructions.cuda.setmaxnreg import SetMaxNRegInst
from tilus.target import nvgpu_sm90


@register_emitter(SetMaxNRegInst, target=nvgpu_sm90)
class SetMaxNRegInstEmitter(BaseInstEmitter):
    def emit(self, inst: SetMaxNRegInst) -> None:
        self.assert_is_warpgroup_aligned(inst, "setmaxnreg.sync.aligned adjusts the limit of entire warp groups")
        self.append(setmaxnreg(inst.nregs, action=inst.action))
