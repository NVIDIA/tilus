// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TilusContext — shared per-process context for all tilus-generated kernels.
//
// Shared across all tilus-generated .so files in the process via tvm_ffi's
// process-global function registry ("tilus.context_ptr"). The first library
// to load allocates the context; subsequent libraries find and reuse it.
// The __attribute__((constructor)) handles init at dlopen time, so no
// Python-side setup is required.
//
// Per-device CUDA workspace: lazily allocated on first use per device.
// Two pools:
//   dirty  — no initialization guarantee; caller may write freely.
//   clean  — zero-initialized on first allocation; caller must restore zeros
//            after use (no automatic re-zeroing on reuse).
// Allocations are rounded up to the next power of two to amortize
// reallocations as the requested size grows.
//
// Note: not thread-safe for concurrent workspace (re)allocation from multiple
// threads on the same device. Acceptable since Python's GIL serializes kernel
// launches unless the user explicitly enables multi-threading.
#pragma once

#include <tvm/ffi/function.h>
#include <cuda_runtime.h>
#include <cstdint>

#define TILUS_CONTEXT_ABI_VERSION 1
#define TILUS_MAX_GPUS 8

struct TilusContext {
    int32_t abi_version;                             // must equal TILUS_CONTEXT_ABI_VERSION
    int32_t _reserved;                               // padding / future use
    // Dirty workspace: may contain arbitrary data; caller must not assume zero-init.
    void*   workspace[TILUS_MAX_GPUS];               // per-GPU device pointer (or nullptr)
    int64_t workspace_size[TILUS_MAX_GPUS];          // current dirty allocation size in bytes
    // Clean workspace: zero-initialized on first allocation.
    // Caller is responsible for preserving the zero invariant after use.
    void*   clean_workspace[TILUS_MAX_GPUS];         // per-GPU device pointer (or nullptr)
    int64_t clean_workspace_size[TILUS_MAX_GPUS];    // current clean allocation size in bytes
};

// Defined in each generated .cu; set by the constructor below.
extern TilusContext* tilus_context;

// Library constructor: runs at dlopen time for every tilus-generated .so.
// Uses tvm_ffi's process-global function registry to share the TilusContext
// across all libraries in the process without any Python-side involvement.
__attribute__((constructor))
static void tilus_context_init() {
    auto f = tvm::ffi::Function::GetGlobal("tilus.context_ptr");
    if (f.has_value()) {
        // A previously loaded tilus library already created the context.
        int64_t addr = (*f)().cast<int64_t>();
        TilusContext* ctx = reinterpret_cast<TilusContext*>(static_cast<uintptr_t>(addr));
        if (ctx->abi_version != TILUS_CONTEXT_ABI_VERSION) {
            TVM_FFI_THROW(RuntimeError)
                << "TilusContext ABI mismatch: expected version " << TILUS_CONTEXT_ABI_VERSION
                << " but found version " << ctx->abi_version
                << ". All tilus-generated libraries in the process must use the same ABI version.";
        }
        tilus_context = ctx;
    } else {
        // First tilus library in this process: allocate the shared context.
        TilusContext* ctx = new TilusContext{};  // zero-initializes all fields
        ctx->abi_version = TILUS_CONTEXT_ABI_VERSION;
        // Register in tvm_ffi's function registry so subsequent libraries find it.
        tvm::ffi::Function::SetGlobal(
            "tilus.context_ptr",
            tvm::ffi::Function::FromPacked([ctx](tvm::ffi::PackedArgs, tvm::ffi::Any* rv) {
                *rv = static_cast<int64_t>(reinterpret_cast<uintptr_t>(ctx));
            }),
            /*override=*/false);
        tilus_context = ctx;
    }
}

// Round up n to the next power of two (returns n if already a power of two).
static inline int64_t tilus_next_pow2(int64_t n) {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// Called from generated host-side launch functions.
// require_clean=false  → dirty workspace (no initialization guarantee).
// require_clean=true   → clean workspace (zero-initialized on first allocation;
//                        caller must restore zeros after use).
// Allocation size is rounded up to the next power of two to reduce reallocations.
static inline void* request_cuda_workspace(int64_t nbytes, bool require_clean) {
    if (tilus_context == nullptr) {
        return nullptr;
    }
    int device_id = 0;
    cudaGetDevice(&device_id);
    if (device_id < 0 || device_id >= TILUS_MAX_GPUS) {
        return nullptr;
    }

    const int64_t alloc_size = tilus_next_pow2(nbytes);

    if (require_clean) {
        if (tilus_context->clean_workspace_size[device_id] < nbytes) {
            if (tilus_context->clean_workspace[device_id] != nullptr) {
                cudaFree(tilus_context->clean_workspace[device_id]);
                tilus_context->clean_workspace[device_id] = nullptr;
            }
            cudaMalloc(&tilus_context->clean_workspace[device_id], static_cast<size_t>(alloc_size));
            cudaMemset(tilus_context->clean_workspace[device_id], 0, static_cast<size_t>(alloc_size));
            tilus_context->clean_workspace_size[device_id] = alloc_size;
        }
        return tilus_context->clean_workspace[device_id];
    } else {
        if (tilus_context->workspace_size[device_id] < nbytes) {
            if (tilus_context->workspace[device_id] != nullptr) {
                cudaFree(tilus_context->workspace[device_id]);
                tilus_context->workspace[device_id] = nullptr;
            }
            cudaMalloc(&tilus_context->workspace[device_id], static_cast<size_t>(alloc_size));
            tilus_context->workspace_size[device_id] = alloc_size;
        }
        return tilus_context->workspace[device_id];
    }
}
