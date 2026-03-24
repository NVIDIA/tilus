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
"""
Tilus shared context for CUDA workspace allocation.

Allocates a TilusContext struct (matching the C definition in tilus/runtime.h)
and registers its address with tvm_ffi via TVMFFIEnvModRegisterContextSymbol.

When tvm_ffi loads a tilus-generated library (tvm_ffi.load_module), it calls
InitContextSymbols, which finds the "tilus_context" symbol in the library and
writes the registered pointer into it. The library's request_cuda_workspace()
function then uses that pointer to allocate per-device CUDA workspace.
"""

import ctypes

TILUS_MAX_GPUS = 8
TILUS_CONTEXT_ABI_VERSION = 1


class _TilusContext(ctypes.Structure):
    """Mirror of the TilusContext C struct in tilus/runtime.h."""

    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("_reserved", ctypes.c_int32),
        ("workspace", ctypes.c_void_p * TILUS_MAX_GPUS),
        ("workspace_size", ctypes.c_int64 * TILUS_MAX_GPUS),
        ("clean_workspace", ctypes.c_void_p * TILUS_MAX_GPUS),
        ("clean_workspace_size", ctypes.c_int64 * TILUS_MAX_GPUS),
    ]


# Module-level singleton — must outlive all tilus-generated libraries.
_context = _TilusContext()
_context.abi_version = TILUS_CONTEXT_ABI_VERSION


def _register_context() -> None:
    """Register the TilusContext pointer with tvm_ffi.

    This must be called before any tilus-generated library is loaded so that
    tvm_ffi can inject the pointer during InitContextSymbols.
    """
    libc = ctypes.CDLL(None)
    reg_fn = libc.TVMFFIEnvModRegisterContextSymbol
    reg_fn.restype = ctypes.c_int
    reg_fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    ret = reg_fn(b"tilus_context", ctypes.addressof(_context))
    if ret != 0:
        raise RuntimeError("TVMFFIEnvModRegisterContextSymbol failed")


# Register at module import time.
_register_context()
