// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Error handling helpers for tilus-generated CUDA kernels.
#pragma once

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
