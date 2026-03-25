// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tilus runtime helpers for generated CUDA kernels.
// Include this header in generated .cu files to get stream access, error
// handling, and the shared CUDA workspace context.
#pragma once

#include <tilus/stream.h>
#include <tilus/context.h>
