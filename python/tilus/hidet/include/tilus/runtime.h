// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tilus runtime helpers for generated CUDA kernels.
// CUDA stream is provided by tvm_ffi (TVMFFIEnvGetStream).
#pragma once

// ---------------------------------------------------------------------------
// tvm_ffi stream API — provides TVMFFIEnvGetStream / TVMFFIEnvSetStream
// ---------------------------------------------------------------------------
#include <tvm/ffi/extra/c_env_api.h>

// ---------------------------------------------------------------------------
// CUDA stream helper
// kDLCUDA = 2 per DLDeviceType; device_id 0 is the current device
// ---------------------------------------------------------------------------
static inline void* get_cuda_stream() {
    return TVMFFIEnvGetStream(2, 0);
}

// ---------------------------------------------------------------------------
// Workspace stub — tilus kernels receive pre-allocated tensors from the
// caller and do not need dynamic workspace allocation.
// TODO: replace with tvm_ffi workspace API once contributed.
// ---------------------------------------------------------------------------
static inline void* request_cuda_workspace(int64_t /*nbytes*/, bool /*require_clean*/) {
    return nullptr;
}
