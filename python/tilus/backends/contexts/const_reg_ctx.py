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

from dataclasses import dataclass
from typing import Sequence

from hidet.ir.expr import Expr, Var

from tilus.backends.context import BaseEmitContext
from tilus.extensions.hidet.ir.tools import rewrite
from tilus.ir.tensor import RegisterTensor


@dataclass
class ConstRegTensorInfo:
    """Represents a CTA-invariant register tensor whose elements can be computed from their logical indices.

    The value at logical indices ``(i0, i1, ...)`` is obtained by substituting
    ``axes[0] = i0, axes[1] = i1, ...`` into ``expr``.

    Attributes
    ----------
    axes: list[Var]
        Variables representing the logical indices, one per tensor dimension.
    expr: Expr
        The expression computing the tensor element value, parameterized by ``axes``.
    """

    axes: list[Var]
    expr: Expr


class ConstRegTensorEmitContext(BaseEmitContext):
    """Tracks RegisterTensors whose elements are CTA-invariant expressions.

    Some RegisterTensors (e.g., barrier addresses from AllocBarrierInst) have values that are
    constant during the lifetime of a CTA and can be expressed as a closed-form function of the
    logical indices. Instead of materializing these as arrays (which may be spilled to local memory
    by nvcc), consumers can use this context to obtain the value via arithmetic.

    The normal array materialization is still emitted as a fallback. This context enables emitters
    for specific instructions (e.g., SliceRegisterInst) to bypass array indexing and use the
    arithmetic expression directly.
    """

    def __post_init__(self):
        self._tracked: dict[RegisterTensor, ConstRegTensorInfo] = {}

    def register(self, tensor: RegisterTensor, axes: list[Var], expr: Expr) -> None:
        """Register a CTA-invariant register tensor.

        Parameters
        ----------
        tensor: RegisterTensor
            The tensor to track.
        axes: list[Var]
            Variables representing the logical indices used in ``expr``, one per tensor dimension.
        expr: Expr
            The expression computing the tensor element value, parameterized by ``axes``.
        """
        self._tracked[tensor] = ConstRegTensorInfo(axes=axes, expr=expr)

    def is_tracked(self, tensor: RegisterTensor) -> bool:
        """Check if a tensor is tracked as CTA-invariant."""
        return tensor in self._tracked

    def get_info(self, tensor: RegisterTensor) -> ConstRegTensorInfo:
        """Get the tracking info for a CTA-invariant tensor."""
        return self._tracked[tensor]

    def get_value(self, tensor: RegisterTensor, logical_indices: Sequence[Expr]) -> Expr:
        """Compute the value of a CTA-invariant tensor at the given logical indices.

        Parameters
        ----------
        tensor: RegisterTensor
            The tracked tensor.
        logical_indices: Sequence[Expr]
            The logical index expressions (may be runtime variables), one per tensor dimension.

        Returns
        -------
        ret: Expr
            The value expression with axis variables substituted by the given indices.
        """
        info = self._tracked[tensor]
        if len(info.axes) != len(logical_indices):
            raise ValueError(
                f"Expected {len(info.axes)} indices, got {len(logical_indices)} for tensor with shape {tensor.shape}"
            )
        mapping = dict(zip(info.axes, logical_indices))
        return rewrite(info.expr, mapping)
