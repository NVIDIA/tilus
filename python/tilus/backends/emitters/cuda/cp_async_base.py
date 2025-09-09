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
from dataclasses import dataclass
from typing import Optional, Sequence

from hidet.ir import logical_and
from hidet.ir.dtypes import boolean
from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType

from tilus.backends.codegen import BaseInstEmitter
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir import GlobalTensor
from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid
from tilus.ir.tensor import SharedLayout, SharedTensor
from tilus.utils import prod


@dataclass
class AxesInfo:
    axes: list[Var]
    mask_expr: Expr
    global_offset: Expr
    shared_offset: Expr

@dataclass
class CopyAsyncAnalysisResult:
    shared_info: TensorInfo
    global_info: TensorInfo
    mask_info: TensorInfo
    contiguous_dim: int
    cp_size: int

class CopyAysncBaseEmitter(BaseInstEmitter):
    @staticmethod
    def get_axes_info(
        shared_tensor: SharedTensor,
        global_tensor: GlobalTensor,
        offsets: Sequence[Expr],
        dims: Sequence[int],
        check_bounds: bool
    ) -> AxesInfo:
        # axes
        axes = index_vars(num_vars=len(shared_tensor.shape))

        # shared_offset
        shared_offset = shared_tensor.layout(*axes)

        # global_offset
        assert len(dims) == len(set(dims)) and tuple(dims) == tuple(sorted(dims))
        global_offsets = list(offsets)
        for i, dim in enumerate(dims):
            global_offsets[dim] = global_offsets[dim] + axes[i]
        global_offset = global_tensor.layout(*global_offsets)

        # mask_expr
        mask_expr = boolean.true
        if check_bounds:
            for i, offset in enumerate(offsets):
                mask_expr = logical_and(mask_expr, logical_and(0 <= offset, offset < global_tensor.shape[i]))

        return AxesInfo(axes=axes, mask_expr=mask_expr, global_offset=global_offset, shared_offset=shared_offset)


    def analyze(
        self,
        shared_tensor: SharedTensor,
        global_tensor: GlobalTensor,
        offsets: Sequence[Expr],
        dims: Sequence[int],
        check_bounds: bool
    ) -> CopyAsyncAnalysisResult:
        dtype: DataType = shared_tensor.dtype
        layout: SharedLayout = shared_tensor.layout
        shape: Sequence[int] = layout.shape

        # get shared, global, and mask info
        axes_info = self.get_axes_info(shared_tensor, global_tensor, offsets, dims, check_bounds=check_bounds)
        analysis = self.codegen.function.metadata.analysis
        axes = axes_info.axes
        shared_info: TensorInfo = analyze_grid(shape=shape, axes=axes, analysis=analysis, expr=axes_info.shared_offset)
        mask_info: TensorInfo = analyze_grid(shape=shape, axes=axes, analysis=analysis, expr=axes_info.mask_expr)
        global_info: TensorInfo = analyze_grid(shape=shape, axes=axes, analysis=analysis, expr=axes_info.global_offset)

        contiguous_dim: Optional[int] = None
        cp_size: Optional[int] = None
        for nbytes in [16, 8, 4]:
            nbits = nbytes * 8
            for dim in reversed(range(len(shape))):
                if global_info.infos[dim].continuity == 1:
                    continue
                if global_info.infos[dim].divisibility * dtype.nbits % nbits != 0:
                    continue
                if shared_info.infos[dim].continuity * dtype.nbits % nbits != 0:
                    continue
                if shared_info.infos[dim].divisibility * dtype.nbits % nbits != 0:
                    continue
                if mask_info.infos[dim].constancy * dtype.nbits % nbits != 0:
                    continue
                if prod(shape) * dtype.nbits // nbits % 32 != 0 and nbytes != 4:
                    # when possible, we hope at least use 32 threads to perform cp.async
                    continue
                contiguous_dim = dim
                cp_size = nbytes
                break
            if contiguous_dim is not None:
                break

        if contiguous_dim is not None:
            assert cp_size is not None
            return CopyAsyncAnalysisResult(
                shared_info=shared_info,
                global_info=global_info,
                mask_info=mask_info,
                contiguous_dim=contiguous_dim,
                cp_size=cp_size
            )
        else:
            return CopyAsyncAnalysisResult(
                shared_info=shared_info,
                global_info=global_info,
                mask_info=mask_info,
                contiguous_dim=-1,
                cp_size=-1
            )
