# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused FP8 epilogues for bandwidth-bound quantization kernels."""

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.dtypes import bfloat16, float8_e4m3, float32, uint32
from tilus.hidet.ir.expr import cast
from tilus.hidet.ir.primitives.cuda.float8 import scale_bf16x4_to_fp8e4m3x4
from tilus.hidet.ir.type import void_p
from tilus.ir.instructions import StoreScaledFp8E4M3FromSharedInst
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor


@register_emitter(StoreScaledFp8E4M3FromSharedInst)
class StoreScaledFp8E4M3FromSharedEmitter(BaseInstEmitter):
    """Lower the 128x128 per-channel FP8 epilogue without register tiles."""

    def emit(self, inst: StoreScaledFp8E4M3FromSharedInst) -> None:
        dst: GlobalTensor = inst.inputs[0].as_global_tensor()
        src: SharedTensor = inst.inputs[1].as_shared_tensor()
        inv_scale: RegisterTensor = inst.inputs[2].as_register_tensor()
        if (
            dst.dtype != float8_e4m3
            or src.dtype != bfloat16
            or inv_scale.dtype != float32
            or tuple(src.shape) != (128, 128)
            or tuple(inv_scale.shape) != (1, 128)
            or inv_scale.layout.local_size != 4
        ):
            raise ValueError("fused FP8 epilogue requires shared bf16[128,128] and register float32[1,128]")

        dst_buf = self.tensor2var[dst]
        src_buf = self.tensor2var[src]
        scale_buf = self.tensor2var[inv_scale]
        offset_m, offset_n = inst.offsets
        lane_id = self.lane_id()
        warp_id = self.warp_id()
        col = lane_id * 4
        with self.for_range(16, attr="u+") as i:
            row = warp_id * 16 + i
            src_offset = src.layout(row, col)
            dst_offset = dst.layout(offset_m + row, offset_n + col)
            self.append(
                scale_bf16x4_to_fp8e4m3x4(
                    cast(~dst_buf[dst_offset], void_p),
                    cast(~src_buf[src_offset], void_p),
                    cast(~src_buf[src_offset + 2], void_p),
                    scale_buf[0],
                    scale_buf[1],
                    scale_buf[2],
                    scale_buf[3],
                )
            )
