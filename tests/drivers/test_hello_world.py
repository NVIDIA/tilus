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
import pytest
from tilus.drivers import build_and_load_program
from tilus.ir.builders import IRBuilder
from tilus.runtime import CompiledProgram


def test_compile_hello_world():
    ib = IRBuilder()

    with ib.function("hello_world", num_warps=1, params=[]):
        ib.num_blocks = [1]
        ib.printf("Hello, world!\n")

    program = ib.flush_program()

    compiled_module: CompiledProgram = build_and_load_program(program)

    compiled_module()


if __name__ == "__main__":
    pytest.main([__file__])
