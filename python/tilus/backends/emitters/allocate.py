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
from hidet.ir.expr import tensor_var
from hidet.ir.tools import rewrite

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir import RegisterTensor
from tilus.ir.instructions import AllocateRegisterInst


@register_emitter(AllocateRegisterInst)
class AllocateInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateRegisterInst) -> None:  # type: ignore
        output: RegisterTensor = inst.register_output
        var = self.declare(tensor_var("regs", shape=[output.local_size], dtype=output.dtype))
        if inst.init is not None:
            axes = inst.axes
            init = inst.init
            with self.for_range(output.local_size) as i:
                global_indices = output.layout.get_global(local_index=i, spatial_index=self.current_worker)
                self.buffer_store(
                    buf=var,
                    indices=[i],
                    value=rewrite(
                        init,
                        rewrite_map={axis: global_index for axis, global_index in zip(axes, global_indices)},
                    ),
                )
        self.tensor2var[output] = var
