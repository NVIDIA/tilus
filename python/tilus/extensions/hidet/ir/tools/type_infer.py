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
from hidet.ir.expr import default_float_dtype
from hidet.ir.tools.type_infer import TypeInfer

from tilus.extensions import update


class UpdatedTypeInfer(TypeInfer):
    def visit_PyConstant(self, c):
        if isinstance(c, float):
            return default_float_dtype
        else:
            return super().visit_PyConstant(c)


@update("hidet.ir.tools.infer_type")
@update("hidet.ir.tools.type_infer.infer_type")
def infer_type(expr):
    infer = UpdatedTypeInfer()
    return infer(expr)
