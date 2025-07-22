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
from typing import no_type_check

from hidet.ir.func import Function
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import asm, attrs, script  # pylint: disable=import-outside-toplevel

    @no_type_check
    @script
    def exit_primitive():
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "cuda_exit"

        asm("exit;", outputs=[], inputs=[], is_volatile=True)

    assert isinstance(exit_primitive, Function)
    register_primitive_function(name=exit_primitive.name, func_or_type=exit_primitive)


def exit():
    return call_primitive_func("cuda_exit", args=[])
