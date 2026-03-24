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

// ---------------------------------------------------------------------------
// TilusContext — shared per-process context for all tilus-generated kernels.
//
// Lifetime: allocated by tilus Python runtime (tilus.runtime.context) as a
// ctypes struct, registered via TVMFFIEnvModRegisterContextSymbol("tilus_context", ptr).
// tvm_ffi injects the pointer into each library's tilus_context global when
// the library is loaded via tvm_ffi.load_module().
//
// Per-device CUDA workspace: lazily allocated on first use per device.
// If the existing allocation is large enough, it is reused.
// If require_clean=true, the workspace is zeroed before use.
//
// Note: not thread-safe for concurrent workspace (re)allocation from multiple
// threads on the same device. This is acceptable since Python's GIL serializes
// kernel launches unless the user explicitly enables multi-threading.
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#define TILUS_CONTEXT_ABI_VERSION 1
#define TILUS_MAX_GPUS 8

struct TilusContext {
    int32_t abi_version;                             // must equal TILUS_CONTEXT_ABI_VERSION
    int32_t _reserved;                               // padding / future use
    // Dirty workspace: may contain arbitrary data; caller must not assume zero-init.
    void*   workspace[TILUS_MAX_GPUS];               // per-GPU device pointer (or nullptr)
    int64_t workspace_size[TILUS_MAX_GPUS];          // current dirty allocation size in bytes
    // Clean workspace: guaranteed to be zero-initialized on first allocation.
    // The caller is responsible for preserving the zero-init invariant after use.
    void*   clean_workspace[TILUS_MAX_GPUS];         // per-GPU device pointer (or nullptr)
    int64_t clean_workspace_size[TILUS_MAX_GPUS];    // current clean allocation size in bytes
};

// Defined in each generated .cu as an exported symbol.
// Injected by tvm_ffi at library load time via InitContextSymbols.
extern TilusContext* tilus_context;

// Called from generated host-side launch functions.
// require_clean=false  → dirty workspace (no initialization guarantee).
// require_clean=true   → clean workspace (zero-initialized on first allocation;
//                        caller must restore zeros after use).
static inline void* request_cuda_workspace(int64_t nbytes, bool require_clean) {
    if (tilus_context == nullptr) {
        return nullptr;
    }
    int device_id = 0;
    cudaGetDevice(&device_id);
    if (device_id < 0 || device_id >= TILUS_MAX_GPUS) {
        return nullptr;
    }

    if (require_clean) {
        if (tilus_context->clean_workspace_size[device_id] < nbytes) {
            if (tilus_context->clean_workspace[device_id] != nullptr) {
                cudaFree(tilus_context->clean_workspace[device_id]);
                tilus_context->clean_workspace[device_id] = nullptr;
            }
            // cudaMallocManaged / cudaMemset ensures zero-init on first allocation.
            cudaMalloc(&tilus_context->clean_workspace[device_id], static_cast<size_t>(nbytes));
            cudaMemset(tilus_context->clean_workspace[device_id], 0, static_cast<size_t>(nbytes));
            tilus_context->clean_workspace_size[device_id] = nbytes;
        }
        return tilus_context->clean_workspace[device_id];
    } else {
        if (tilus_context->workspace_size[device_id] < nbytes) {
            if (tilus_context->workspace[device_id] != nullptr) {
                cudaFree(tilus_context->workspace[device_id]);
                tilus_context->workspace[device_id] = nullptr;
            }
            cudaMalloc(&tilus_context->workspace[device_id], static_cast<size_t>(nbytes));
            tilus_context->workspace_size[device_id] = nbytes;
        }
        return tilus_context->workspace[device_id];
    }
}
