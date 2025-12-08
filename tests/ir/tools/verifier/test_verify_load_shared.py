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
import tilus
from tilus.ir.tools import VerificationError, verify


class DemoLoadShared(tilus.Script):
    def __init__(self, shared_shape, register_shape):
        super().__init__()
        self.shared_shape = shared_shape
        self.register_shape = register_shape

    def __call__(self):
        self.attrs.warps = 2
        self.attrs.blocks = 1

        regs = self.register_tensor(dtype=tilus.float32, shape=self.register_shape)
        smem = self.shared_tensor(dtype=tilus.float32, shape=self.shared_shape)
        self.load_shared(src=smem, out=regs)


@pytest.mark.parametrize(
    "shared_shape, register_shape, success",
    [
        ([4, 4], [8, 8], False),
        ([8, 8], [8, 8], True),
    ],
)
def test_verify_load_shared(shared_shape, register_shape, success):
    script = DemoLoadShared(shared_shape=shared_shape, register_shape=register_shape)
    program = script._jit_instance_for().transpiled_programs[0]  # type: ignore

    if success:
        verify(program)
    else:
        try:
            verify(program)
        except VerificationError:
            pass
        else:
            assert False
