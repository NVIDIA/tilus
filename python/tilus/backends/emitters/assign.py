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
from tilus.ir.instructions import AssignInst


@register_emitter(AssignInst)
class AllocateInstEmitter(BaseInstEmitter):
    def emit(self, inst: AssignInst) -> None:  # type: ignore
        value = inst.register_output
        input_value = inst.inputs[0].as_register_tensor()
        var = self.get_or_allocate_var(tensor=value, name="regs")
        assert input_value.dtype == value.dtype
        assert input_value.layout == value.layout
        with self.for_range(value.layout.local_size) as i:
            self.buffer_store(buf=var, indices=[i], value=self.tensor2var[input_value][i])

        self.tensor2var[value] = var
