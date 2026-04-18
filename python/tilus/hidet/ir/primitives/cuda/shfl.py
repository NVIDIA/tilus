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
from tilus.hidet.ir.dtypes import bfloat16, float16, float32, float64, int32, int64, uint32, uint64
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.ir.tools import infer_type
from tilus.hidet.ir.type import DataType, FuncType
from tilus.hidet.utils import initialize

# dtypes supported by the shfl intrinsics (signature: T __shfl*_sync(unsigned, T, int, int=warpSize))
_shfl_dtypes = [int32, uint32, int64, uint64, float16, bfloat16, float32, float64]


_shfl_bases = {
    "shfl_sync": "__shfl_sync",
    "shfl_up_sync": "__shfl_up_sync",
    "shfl_down_sync": "__shfl_down_sync",
    "shfl_xor_sync": "__shfl_xor_sync",
}


@initialize()
def register_primitive_functions():
    register_primitive_function(name="cuda_activemask", func_or_type=FuncType([], int32), codegen_name="__activemask")
    for base, codegen_name in _shfl_bases.items():
        for dtype in _shfl_dtypes:
            # T __shfl*_sync(unsigned mask, T var, int srcLane_or_delta_or_laneMask, int width=warpSize)
            func_type = FuncType(param_types=[uint32, dtype, int32, int32], ret_type=dtype)
            register_primitive_function(
                name=f"cuda_{base}_{dtype.name}", codegen_name=codegen_name, func_or_type=func_type
            )


def _shfl_dispatch(base: str, mask: Expr, var: Expr, lane_arg: Expr, width: Expr) -> Expr:
    var_type = infer_type(var)
    if not isinstance(var_type, DataType):
        raise TypeError(f"{base} expects a scalar data-typed value, got {var_type}")
    name = f"cuda_{base}_{var_type.name}"
    return call_primitive_func(name, [mask, var, lane_arg, width])


def shfl_sync(mask, var, src_lane, width=32):
    return _shfl_dispatch("shfl_sync", mask, var, src_lane, width)


def shfl_up_sync(mask, var, delta, width=32):
    return _shfl_dispatch("shfl_up_sync", mask, var, delta, width)


def shfl_down_sync(mask, var, delta, width=32):
    return _shfl_dispatch("shfl_down_sync", mask, var, delta, width)


def shfl_xor_sync(mask, var, lane_mask, width=32):
    return _shfl_dispatch("shfl_xor_sync", mask, var, lane_mask, width)


def active_mask():
    return call_primitive_func("cuda_activemask", [])
