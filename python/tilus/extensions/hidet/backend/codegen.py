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
import os
from typing import Sequence, Union

from hidet.backend.codegen import Codegen, CPUCodegen, CUDACodegen, HIPCodegen
from hidet.ir.expr import Expr
from hidet.ir.module import IRModule
from hidet.ir.target import Target
from hidet.ir.type import DataType
from hidet.utils.doc import Doc, Text

from tilus.extensions.hidet.ir.dtypes.vector import uint32x1, uint32x2, uint32x4


class UpdatedCUDACodeGen(CUDACodegen):
    def scalar_literal(self, value: Expr, dtype: DataType) -> Doc:
        ret: Union[str, Doc]
        if dtype == uint32x1:
            ret = "make_uint1({})".format(int(value[0]))
        elif dtype == uint32x2:
            ret = "make_uint2({}, {})".format(int(value[0]), int(value[1]))
        elif dtype == uint32x4:
            ret = "make_uint4({}, {}, {}, {})".format(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
        else:
            ret = super().scalar_literal(value, dtype)
        if isinstance(ret, str):
            ret = Text(ret)
        return ret


def codegen(ir_module: Union[IRModule, Sequence[IRModule]], src_out_path: str, target: Union[str, Target]) -> str:
    if isinstance(target, str):
        target = Target.from_string(target)

    gen: Codegen
    if target.name == "cuda":
        gen = UpdatedCUDACodeGen()
    elif target.name == "hip":
        gen = HIPCodegen()
    elif target.name == "cpu":
        gen = CPUCodegen()
    else:
        raise ValueError(f"Unknown target: {target}")

    code = ""
    if isinstance(ir_module, Sequence):
        for m in ir_module:
            doc = gen(m)
            code += str(doc) + "\n"
    else:
        doc = gen(ir_module)
        code = str(doc)
    if src_out_path is not None:
        dir_path = os.path.dirname(src_out_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(src_out_path, "w") as f:
            f.write(code)
    return code
