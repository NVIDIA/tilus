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
from __future__ import annotations

import tvm_ffi
from tvm_ffi.dataclasses import py_class


@py_class("tilus.hidet.ir.Node", structural_eq="tree")
class Node(tvm_ffi.Object):
    """Base class of all hidet IR nodes.

    Subclasses use :func:`tvm_ffi.dataclasses.py_class` so their annotated
    fields become FFI-backed attributes. Python ``==`` / ``hash()`` come
    from ``tvm_ffi.Object``'s default handle-address-based implementation
    — two wrappers of the same C handle collide, two distinct IR
    fragments stay as separate entries. That's already the identity
    semantics compiler passes want, so plain ``dict`` / ``set`` on IR
    nodes Just Works and no wrapper class is needed.

    For structural comparison, use :func:`tvm_ffi.structural_equal` /
    :func:`tvm_ffi.structural_hash` (each subclass declares its kind via
    ``structural_eq="tree"`` on the decorator), or wrap values in
    :class:`tvm_ffi.StructuralKey` as a dict key.
    """

    def __str__(self) -> str:
        from tilus.hidet.ir.tools.printer import astext  # noqa: PLC0415

        return astext(self)

    def __repr__(self) -> str:
        return str(self)


def is_seq(value: object) -> bool:
    """Duck-typed sequence check covering Python ``tuple``/``list`` and
    ``tvm_ffi`` ``Array``/``List`` containers.

    Useful at any ``isinstance(x, (list, tuple))`` site that also needs
    to accept FFI-backed fields rewritten from ``tuple[T, ...]`` /
    ``list[T]`` annotations (those aren't ``isinstance`` of the Python
    builtins even though they iterate like them).
    """
    if isinstance(value, (tuple, list)):
        return True
    return type(value).__name__ in ("Array", "List")
