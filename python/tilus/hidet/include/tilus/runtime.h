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
// Workspace — not yet implemented; tvm_ffi workspace API support pending.
// Kernels that use AllocateGlobal (e.g. split-K semaphores) will not work
// until this is wired up.
// ---------------------------------------------------------------------------
static inline void* request_cuda_workspace(int64_t /*nbytes*/, bool /*require_clean*/) {
    return nullptr;
}

// ---------------------------------------------------------------------------
// Error handling — HidetException and hidet_set_last_error for generated code
// ---------------------------------------------------------------------------
#include <stdexcept>
#include <string>

struct HidetException : public std::exception {
    std::string message;
    explicit HidetException(const std::string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

static thread_local std::string hidet_last_error;
static inline void hidet_set_last_error(const char* msg) {
    hidet_last_error = msg;
}
static inline const char* hidet_get_last_error() {
    return hidet_last_error.c_str();
}
