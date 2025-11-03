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
from hidet.ir.expr import Var, cast, tensor_pointer_var

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.ir.instructions import ViewInst


@register_emitter(ViewInst)
class ViewInstEmitter(BaseInstEmitter):
    def emit(self, inst: ViewInst) -> None:
        out_value = inst.register_output
        in_var = self.tensor2var[inst.inputs[0]]
        out_var: Var = self.declare(
            v=tensor_pointer_var("viewed", shape=[out_value.layout.local_size], dtype=out_value.dtype),
            init=cast(~in_var[inst.local_offset], ~out_value.dtype),
        )
        self.tensor2var[out_value] = out_var
