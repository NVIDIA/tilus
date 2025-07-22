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
from hidet.ir.dtypes import int32
from tilus.ir.builders import IRBuilder


def test_program_builder():
    ib = IRBuilder()

    with ib.program():
        with ib.function(name="hello", num_warps=1, params={"n": int32}) as n:
            ib.num_blocks = 1
            ib.printf("Hello, world!\n")
            ib.printf("n = %d\n", n)

    program = ib.flush_program()
    print(program)
