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
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from hidet.graph.frontend.torch.utils import dtype_from_torch, dtype_to_torch
from hidet.ir.dtypes import float32, int32
from hidet.ir.type import DataType
from hidet.utils import prod, same_list


class Tensor:
    def __init__(self, dtype: DataType, shape: Sequence[int], storage: torch.Tensor):
        self.dtype: DataType = dtype
        self.shape: Sequence[int] = shape
        self.storage: torch.Tensor = storage

    def __str__(self):
        from tilus.kernels import cast

        if self.dtype.is_integer():
            storage = cast(self, int32).storage
            torch_tensor = storage.view(torch.int32).to(torch.int64).reshape(self.shape)
        elif self.dtype.is_float():
            storage = cast(self, float32).storage
            torch_tensor = storage.view(torch.float32).reshape(self.shape)
        else:
            raise ValueError()
        return "{}{}:\n{}".format(self.dtype.name, list(self.shape), torch_tensor)

    def __setitem__(self, key, value):
        # analyze the key
        offsets = []
        slice_shape = []
        if isinstance(key, int):
            key = (key,)
        if len(key) > len(self.shape):
            raise ValueError("Too many indices.")
        if len(key) < len(self.shape):
            key = key + (slice(None),) * (len(self.shape) - len(key))
        for i, item in enumerate(key):
            if isinstance(item, int):
                offsets.append(item)
                slice_shape.append(1)
            elif isinstance(item, slice):
                if item.step is not None:
                    raise ValueError("Slice step is not supported.")
                start = 0 if item.start is None else int(item.start)
                stop = self.shape[i] if item.stop is None else int(item.stop)
                offsets.append(start)
                slice_shape.append(stop - start)
            else:
                raise ValueError("Unsupported index: {}.".format(item))

        # check the value
        if isinstance(value, (float, int)):
            value = full(slice_shape, value, self.dtype)
        if not same_list(slice_shape, value.shape):
            raise ValueError("Shape mismatch: {} vs {}.".format(slice_shape, value.shape))
        if self.dtype != value.dtype:
            from tilus.kernels import cast

            value = cast(value, self.dtype)
        # set_slice(self, offsets=offsets, value=value)

    def view(self, *, dtype: Optional[DataType] = None, shape: Optional[Sequence[int]] = None) -> Tensor:
        return view(self, dtype, shape)

    def clone(self) -> Tensor:
        return Tensor(self.dtype, list(self.shape), self.storage.clone())

    def torch(self) -> torch.Tensor:
        torch_dtype = dtype_to_torch(self.dtype)
        if torch_dtype is None:
            raise ValueError("PyTorch does not support dtype {} for now.".format(self.dtype.name))
        return self.storage.view(torch_dtype).reshape(self.shape)

    def to(self, dtype: DataType) -> Tensor:
        from tilus.kernels import cast

        return cast(self, dtype)

    def data_ptr(self) -> int:
        return self.storage.data_ptr()


def from_torch(torch_tensor: torch.Tensor) -> Tensor:
    dtype = dtype_from_torch(torch_tensor.dtype)
    return Tensor(dtype, torch_tensor.shape, torch_tensor)


def view_torch(torch_tensor: torch.Tensor, *, dtype: DataType, shape: List[int]) -> Tensor:
    assert (dtype.nbits * prod(shape) + 7) // 8 == torch_tensor.nbytes
    return Tensor(dtype, shape, torch_tensor)


def empty(shape: Sequence[int], dtype: DataType) -> Tensor:
    nbytes = (dtype.nbits * prod(shape) + 7) // 8
    storage = torch.empty([nbytes], dtype=torch.uint8, device="cuda")
    return Tensor(dtype, shape, storage)


def rand(shape: List[int], dtype: DataType) -> Tensor:
    from tilus.kernels import cast

    tensor = from_torch(torch.rand(shape, dtype=torch.float32, device="cuda"))
    tensor = cast(tensor, dtype)
    return tensor


def randn(shape: List[int], dtype: DataType) -> Tensor:
    from tilus.kernels import cast

    tensor = from_torch(torch.randn(shape, dtype=torch.float32, device="cuda"))
    tensor = cast(tensor, dtype)
    return tensor


def randint(low: int, high: int, shape: List[int], dtype: DataType) -> Tensor:
    from tilus.kernels import cast

    tensor = from_torch(torch.randint(low=low, high=high, size=shape, dtype=torch.int32, device="cuda"))
    tensor = cast(tensor, dtype)
    return tensor


def ones(shape: List[int], dtype: DataType) -> Tensor:
    from tilus.kernels import cast

    tensor = from_torch(torch.ones(shape, dtype=torch.float32, device="cuda"))
    tensor = cast(tensor, dtype)
    return tensor


def zeros(shape: List[int], dtype: DataType) -> Tensor:
    from tilus.kernels import cast

    tensor = from_torch(torch.zeros(shape, dtype=torch.float32, device="cuda"))
    tensor = cast(tensor, dtype)
    return tensor


def full(shape: List[int], fill_value: Union[float, int], dtype: DataType) -> Tensor:
    from tilus.kernels import cast

    tensor = from_torch(torch.full(shape, fill_value, dtype=torch.float32, device="cuda"))
    tensor = cast(tensor, dtype)
    return tensor


def view(tensor: Tensor, dtype: Optional[DataType] = None, shape: Optional[Sequence[int]] = None) -> Tensor:
    if dtype is None:
        dtype = tensor.dtype
    if shape is None:
        shape = list(tensor.shape)
        if dtype.nbits == tensor.dtype.nbits:
            pass  # no change
        elif (
            dtype.nbits % tensor.dtype.nbits == 0 and shape[-1] % (dtype.nbits // tensor.dtype.nbits) == 0
        ) or tensor.dtype.nbits % dtype.nbits == 0:
            shape[-1] = shape[-1] * tensor.dtype.nbits // dtype.nbits
        else:
            raise ValueError("Cannot infer shape.")

    actual_nbits = dtype.nbits * prod(shape)
    expect_nbits = tensor.dtype.nbits * prod(tensor.shape)
    assert actual_nbits == expect_nbits, f"{actual_nbits} != {expect_nbits}"
    return Tensor(dtype, shape, tensor.storage)
