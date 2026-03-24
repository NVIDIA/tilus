# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Union

from tilus.hidet.ir.dtypes import boolean, int64
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.ir.type import FuncType, void_p
from tilus.hidet.utils import initialize


@initialize()
def register_functions():
    register_primitive_function(
        name="get_cuda_stream", func_or_type=FuncType([], void_p), codegen_name="get_cuda_stream"
    )
    register_primitive_function(
        name="request_cuda_workspace",
        func_or_type=FuncType([int64, boolean], void_p),
        codegen_name="request_cuda_workspace",
    )


def get_cuda_stream() -> void_p:
    return call_primitive_func("get_cuda_stream", [])


def request_cuda_workspace(nbytes: Union[int, Expr], require_clean: Union[bool, Expr] = False) -> void_p:
    return call_primitive_func("request_cuda_workspace", [nbytes, require_clean])
