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
from typing import Sequence

from hidet.ir import logical_and
from hidet.ir.dtypes import boolean
from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType
from hidet.utils import gcd
from hidet.utils.doc import doc_join_lines

from tilus.backends.codegen import BaseInstEmitter
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir import GlobalTensor
from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid
from tilus.ir.tensor import SharedLayout, SharedTensor


@dataclass
class AxesInfo:
    axes: list[Var]
    mask_expr: Expr
    global_offset: Expr
    shared_offset: Expr


@dataclass
class CopyAsyncAnalysisResult:
    dtype: DataType
    shared_info: TensorInfo
    global_info: TensorInfo
    mask_info: TensorInfo
    contiguous_dim: int
    cp_size_bits: int

    def __str__(self):
        seq = []
        seq.append(f"dtype={self.dtype.name}")
        for name, info in [
            ("shared_info", self.shared_info),
            ("global_info", self.global_info),
            ("mask_info", self.mask_info),
        ]:
            seq.append(f"{name}={info}")
        seq.append(f"contiguous_dim={self.contiguous_dim}")
        seq.append(f"cp_size_bits={self.cp_size_bits}")
        return str(doc_join_lines(seq=seq, left="CopyAsyncAnalysisResult(", right=")", indent=4))

@dataclass
class CopyAsyncAnalysisSharedToSharedResult:
    dtype: DataType
    shared_src_info: TensorInfo
    shared_dst_info: TensorInfo
    contiguous_dim: int
    cp_size_bits: int

    def __str__(self):
        seq = []
        seq.append(f"dtype={self.dtype.name}")
        for name, info in [
            ("shared_src_info", self.shared_src_info),
            ("shared_dst_info", self.shared_dst_info),
        ]:
            seq.append(f"{name}={info}")
        seq.append(f"contiguous_dim={self.contiguous_dim}")
        seq.append(f"cp_size_bits={self.cp_size_bits}")
        return str(doc_join_lines(seq=seq, left="CopyAsyncAnalysisSharedToSharedResult(", right=")", indent=4))


class CopyAysncBaseEmitter(BaseInstEmitter):
    @staticmethod
    def get_axes_info(
        shared_tensor: SharedTensor,
        global_tensor: GlobalTensor,
        offsets: Sequence[Expr],
        dims: Sequence[int],
        check_bounds: bool,
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

    @staticmethod
    def get_dim_vec_size(src_info: TensorInfo, dst_info: TensorInfo, mask_info: TensorInfo, dim: int) -> int:
        return gcd(
            src_info[dim].continuity,
            dst_info[dim].continuity,
            src_info[dim].divisibility,
            dst_info[dim].divisibility,
            mask_info[dim].constancy,
        )

    def analyze(
        self,
        shared_tensor: SharedTensor,
        global_tensor: GlobalTensor,
        offsets: Sequence[Expr],
        dims: Sequence[int],
        check_bounds: bool,
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

        assert len(shape) > 0

        contiguous_dim: int = len(shape) - 1
        for dim in reversed(range(len(shape))):
            if self.get_dim_vec_size(shared_info, global_info, mask_info, dim) > self.get_dim_vec_size(
                shared_info, global_info, mask_info, contiguous_dim
            ):
                contiguous_dim = dim

        # determine number of bytes to perform the cp.async
        cp_size_bits: int = self.get_dim_vec_size(shared_info, global_info, mask_info, contiguous_dim) * dtype.nbits
        return CopyAsyncAnalysisResult(
            dtype=shared_tensor.dtype,
            shared_info=shared_info,
            global_info=global_info,
            mask_info=mask_info,
            contiguous_dim=contiguous_dim,
            cp_size_bits=cp_size_bits,
        )

    def analyze_shared_to_shared(
        self,
        shared_src: SharedTensor,
        shared_dst: SharedTensor,
    ) -> CopyAsyncAnalysisSharedToSharedResult:
        dtype: DataType = shared_src.dtype
        layout_src: SharedLayout = shared_src.layout
        layout_dst: SharedLayout = shared_dst.layout
        shape: Sequence[int] = layout_src.shape
        assert shape == layout_dst.shape

        # get shared src and dst info
        analysis = self.codegen.function.metadata.analysis
        axes = index_vars(num_vars=len(shape))
        shared_src_info: TensorInfo = analyze_grid(
            shape=shape, axes=axes, analysis=analysis, expr=shared_src.layout(*axes)
        )
        shared_dst_info: TensorInfo = analyze_grid(
            shape=shape, axes=axes, analysis=analysis, expr=shared_dst.layout(*axes)
        )
        mask_info: TensorInfo = TensorInfo.from_constant(shape=shape, value=1)

        assert len(shape) > 0

        contiguous_dim: int = len(shape) - 1
        for dim in reversed(range(len(shape))):
            if self.get_dim_vec_size(shared_src_info, shared_dst_info, shared_dst_info, dim) > self.get_dim_vec_size(
                shared_src_info, shared_dst_info, shared_dst_info, contiguous_dim
            ):
                contiguous_dim = dim

        # determine number of bytes to perform the cp.async
        cp_size_bits: int = self.get_dim_vec_size(shared_src_info, shared_dst_info, mask_info, contiguous_dim) * dtype.nbits
        return CopyAsyncAnalysisSharedToSharedResult(
            dtype=dtype,
            shared_src_info=shared_src_info,
            shared_dst_info=shared_dst_info,
            contiguous_dim=contiguous_dim,
            cp_size_bits=cp_size_bits,
        )
