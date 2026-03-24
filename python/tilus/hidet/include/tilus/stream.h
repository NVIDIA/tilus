// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA stream helper for tilus-generated kernels.
// The active stream is managed by tvm_ffi (TVMFFIEnvGetStream / TVMFFIEnvSetStream).
#pragma once

#include <tvm/ffi/extra/c_env_api.h>

// kDLCUDA = 2 per DLDeviceType; device_id 0 is the current device.
static inline void* get_cuda_stream() {
    return TVMFFIEnvGetStream(2, 0);
}
