# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pandas
import tilus
import torch
from hidet import bfloat16
from hidet.ir.dtypes import int4b
from hidet.ir.type import DataType, void_p
from tilus import (
    Script,
    float3_e1m1,
    float4_e2m1,
    float6_e3m2,
    float8_e5m2,
    float16,
    float32,
    int3b,
    int6b,
    int8,
    int32,
    uint8,
)
from tilus.ir.layout.ops import concat, local, reduce, spatial
from tilus.utils import benchmark_func, cdiv, dtype_to_torch, gcd
from torch import nn

tilus.option.cache_dir("./cache")


class QuantizedMatmulCommon(Script):
    def __init__(
        self, weight_tile: tuple[int, int], a_dtype: DataType, b_dtype: DataType
    ):
        super().__init__()
        tile_k, tile_n = weight_tile
        assert a_dtype in [float16, bfloat16], (
            "this kernel only supports float16/bfloat16 as activation data type"
        )
        if a_dtype == float16:
            self.atomic_mma = self.cuda.atomic_mma_configs.m16n8k16_f16_f32
        else:
            self.atomic_mma = self.cuda.atomic_mma_configs.m16n8k16_bf16_f32
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.tile_k = weight_tile[0]
        self.tile_n = weight_tile[1]
        self.tile_layout = (
            local(tile_k // self.atomic_mma.k, tile_n // self.atomic_mma.n)
            * self.atomic_mma.lb
        )

        bits_per_threads = self.tile_layout.local_size * b_dtype.nbits

        assert bits_per_threads % 8 == 0, "bits_per_threads must be divisible by 8"

        bytes_per_threads = bits_per_threads // 8

        # view as bytes
        inner_size = gcd(bytes_per_threads, 16)
        outer_size = bytes_per_threads // inner_size

        self.flatten_tile_layout = local(outer_size).spatial(32).local(inner_size)


class QuantizedMatmulChangeLayout(QuantizedMatmulCommon):
    def __call__(self, k_size: int, n_size: int, src_ptr: void_p, dst_ptr: void_p):
        self.static_assert(
            k_size % self.tile_k == 0, "k_size must be divisible by tile_k"
        )
        self.static_assert(
            n_size % self.tile_n == 0, "n_size must be divisible by tile_n"
        )
        self.attrs.warps = 1
        self.attrs.blocks = [k_size // self.tile_k, n_size // self.tile_n]

        offset_k = self.blockIdx.x * self.tile_k
        offset_n = self.blockIdx.y * self.tile_n

        g_src = self.global_view(src_ptr, dtype=self.b_dtype, shape=[k_size, n_size])
        r_src = self.load_global(
            g_src, offsets=[offset_k, offset_n], shape=self.tile_layout.shape
        )
        self.annotate_layout(r_src, self.tile_layout)
        r_dst = self.view(r_src, layout=self.flatten_tile_layout, dtype=uint8)
        g_dst = self.global_view(
            dst_ptr,
            dtype=uint8,
            shape=[
                k_size // self.tile_k,
                n_size // self.tile_n,
                self.flatten_tile_layout.shape[0],
            ],
        )
        self.store_global(
            g_dst,
            r_dst,
            offsets=[self.blockIdx.x, self.blockIdx.y, 0],
            dims=[2],
        )


class QuantizedMatmulRestoreLayout(QuantizedMatmulCommon):
    def __call__(self, k_size: int, n_size: int, src_ptr: void_p, dst_ptr: void_p):
        self.static_assert(
            k_size % self.tile_k == 0, "k_size must be divisible by tile_k"
        )
        self.static_assert(
            n_size % self.tile_n == 0, "n_size must be divisible by tile_n"
        )
        self.attrs.warps = 1
        self.attrs.blocks = [k_size // self.tile_k, n_size // self.tile_n]

        offset_k = self.blockIdx.x * self.tile_k
        offset_n = self.blockIdx.y * self.tile_n

        g_src = self.global_view(
            src_ptr,
            dtype=uint8,
            shape=[
                k_size // self.tile_k,
                n_size // self.tile_n,
                self.flatten_tile_layout.shape[0],
            ],
        )
        r_src = self.load_global(
            g_src,
            offsets=[self.blockIdx.x, self.blockIdx.y, 0],
            shape=self.flatten_tile_layout.shape,
        )
        self.annotate_layout(r_src, self.flatten_tile_layout)
        r_dst = self.view(r_src, layout=self.tile_layout, dtype=self.b_dtype)
        g_dst = self.global_view(dst_ptr, dtype=self.b_dtype, shape=[k_size, n_size])
        self.store_global(g_dst, r_dst, offsets=[offset_k, offset_n], dims=[2])


@tilus.autotune("warp_spatial", [[1, 4], [1, 8], [2, 4]])
@tilus.autotune("warp_repeat", [[2, 4, 1], [1, 8, 1], [2, 4, 2], [4, 2, 2], [1, 8, 2]])
@tilus.autotune("num_stages", [3, 4])
@tilus.autotune("split_k_factor", [1, 8, 16])
class QuantizedMatmul(QuantizedMatmulCommon):
    debug_schedule = dict(
        warp_spatial=[2, 2],
        warp_repeat=[2, 4, 2],
        num_stages=3,
        split_k_factor=1,
    )

    def __init__(
        self,
        weight_tile: tuple[int, int],
        group_size: int,
        a_dtype: DataType,
        b_dtype: DataType,
        warp_spatial: tuple[int, int],
        warp_repeat: tuple[int, int, int],
        num_stages: int,
        split_k_factor: int,
    ):
        super().__init__(weight_tile=weight_tile, a_dtype=a_dtype, b_dtype=b_dtype)

        assert a_dtype.is_any_float16(), (
            "this kernel only supports float16/bfloat16 as activation data type"
        )
        assert 1 <= b_dtype.nbits <= 8, (
            "this kernel only supports dtype with 1-8 bits as weight data type"
        )

        self.weight_tile = weight_tile
        self.group_size = group_size
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.warp_spatial = tuple(warp_spatial)
        self.warp_repeat = tuple(warp_repeat)
        self.num_stages = num_stages
        self.split_k_factor = split_k_factor
        self.mma = self.cuda.mma_dot_config(
            atomic_mma=self.atomic_mma,
            warp_spatial=self.warp_spatial,
            warp_repeat=self.warp_repeat,
        )

        wsm, wsn = warp_spatial
        wrm, wrn, wrk = warp_repeat

        assert (
            weight_tile[0] % self.atomic_mma.k == 0
            and weight_tile[1] % self.atomic_mma.n == 0
        )
        tk, tn = (
            weight_tile[0] // self.atomic_mma.k,
            weight_tile[1] // self.atomic_mma.n,
        )

        assert wrk % tk == 0 and wrn % tn == 0, (wrk, tk, wrn, tn)
        self.block_m = self.atomic_mma.m * wsm * wrm
        self.block_n = self.atomic_mma.n * wsn * wrn
        self.block_k = self.atomic_mma.k * wrk
        self.num_warps = wsm * wsn

        k_tiles = wrk // tk
        n_tiles = wsn * wrn // tn

        # we make sure that each weight_tile will be loaded by one warp
        assert wrk * self.atomic_mma.k % weight_tile[0] == 0
        assert wrn * self.atomic_mma.n % weight_tile[1] == 0

        # make sure the group size is divisible by the block_k
        assert self.group_size % self.block_k == 0

        self.tile_bytes = self.flatten_tile_layout.size

        self.layout_rb_flattened = concat(
            reduce(spatial(1, wsn, wsm, ranks=[0, 2, 1]), dims=[2]).local(
                wrk // tk, wrn // tn
            ),
            self.flatten_tile_layout,
        )
        self.layout_rs = reduce(self.mma.lb, dims=[0], keepdims=True)

        self.layout_sa = self.cuda.swizzled_shared_layout(
            self.a_dtype, shape=[num_stages, self.block_m, self.block_k]
        )
        self.layout_sb = self.cuda.shared_layout(
            shape=[self.num_stages, k_tiles, n_tiles, self.tile_bytes]
        )
        self.layout_sc = self.cuda.swizzled_shared_layout(
            self.a_dtype, shape=[self.block_m, self.block_n]
        )
        self.layout_ss = self.cuda.shared_layout(shape=[self.num_stages, 1, self.block_n])

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: void_p,
        b_ptr: void_p,
        scale_ptr: void_p,
        c_ptr: void_p,
    ):
        self.attrs.blocks = [
            cdiv(m_size, self.block_m),
            cdiv(n_size, self.block_n),
            self.split_k_factor,
        ]
        self.attrs.warps = self.num_warps

        # the k_size for each thread block
        block_k_size = (
            cdiv(cdiv(k_size, self.split_k_factor), self.block_k) * self.block_k
        )
        start_offset_k = self.blockIdx.z * block_k_size
        end_offset_k = min(start_offset_k + block_k_size, k_size)

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=self.a_dtype, shape=[m_size, k_size])
        gb = self.global_view(
            b_ptr,
            dtype=uint8,
            shape=[
                k_size // self.tile_k,
                n_size // self.tile_n,
                self.tile_bytes,
            ],
        )
        gs = self.global_view(
            scale_ptr,
            dtype=self.a_dtype,
            shape=[k_size // self.group_size, n_size],
        )

        sa = self.shared_tensor(dtype=self.a_dtype, layout=self.layout_sa)
        sb = self.shared_tensor(dtype=uint8, layout=self.layout_sb)
        ss = self.shared_tensor(dtype=self.a_dtype, layout=self.layout_ss)
        acc = self.register_tensor(
            dtype=float32,
            shape=[self.block_m, self.block_n],
            layout=self.mma.lc,
            init=0.0,
        )

        for stage in range(self.num_stages - 1):
            offset_k = start_offset_k + stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(
                src=gb,
                dst=sb[stage],
                offsets=[offset_k // self.tile_k, offset_n // self.tile_n, 0],
            )
            self.copy_async(
                src=gs,
                dst=ss[stage],
                offsets=[offset_k // self.group_size, offset_n],
            )
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(
            start_offset_k, end_offset_k, block_k, unroll=self.num_stages
        ):
            # computation for current tile
            a = self.load_shared(sa[current_stage], layout=self.mma.la)
            scale = self.load_shared(ss[current_stage], layout=self.layout_rs)
            b_flattened = self.load_shared(
                sb[current_stage], layout=self.layout_rb_flattened
            )
            b_low_precision = self.view(
                b_flattened, dtype=self.b_dtype, layout=self.mma.lb
            )
            b_unscaled = self.cast(b_low_precision, dtype=self.a_dtype)
            b = b_unscaled * scale
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            self.copy_async(
                src=ga,
                dst=sa[preload_stage],
                offsets=[offset_m, preload_offset_k],
            )
            self.copy_async(
                src=gb,
                dst=sb[preload_stage],
                offsets=[
                    preload_offset_k // self.tile_k,
                    offset_n // self.tile_n,
                    0,
                ],
            )
            self.copy_async(
                src=gs,
                dst=ss[preload_stage],
                offsets=[preload_offset_k // self.group_size, offset_n],
            )
            self.copy_async_commit_group()
            self.copy_async_wait_group(n=self.num_stages - 2)

            # update the stage
            current_stage = (current_stage + 1) % self.num_stages
            preload_stage = (preload_stage + 1) % self.num_stages
            self.sync()

        # there might be on-fly copy_async in the pipeline, we need to wait for all of them
        self.copy_async_wait_all()
        self.sync()
        self.free_shared(sa)
        self.free_shared(sb)
        self.free_shared(ss)

        # cast the accumulator to float16 and change the register tensor's layout
        sc = self.shared_tensor(dtype=float16, layout=self.layout_sc)
        casted_acc = self.cast(acc, dtype=float16)
        self.store_shared(sc, casted_acc)
        self.sync()
        rc = self.load_shared(sc)
        self.free_shared(sc)

        m_blocks, n_blocks = cdiv(m_size, block_m), cdiv(n_size, block_n)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        if self.split_k_factor == 0:
            self.store_global(gc, rc, offsets=[offset_m, offset_n])
        else:
            semaphores = self.global_tensor(
                dtype=int32, shape=[m_blocks, n_blocks], requires_clean=True
            )
            semaphore: ~int32 = semaphores[self.blockIdx.x, self.blockIdx.y].item_ptr()

            # load and accumulate the partial result in global memory
            if self.blockIdx.z > 0:
                self.lock_semaphore(semaphore, value=self.blockIdx.z)
                partial_rc = self.load_global(
                    gc, offsets=[offset_m, offset_n], shape=[block_m, block_n]
                )
                self.add(rc, partial_rc, out=rc)

            # store the result to global memory and release the semaphore
            self.store_global(gc, rc, offsets=[offset_m, offset_n])

            # release the semaphore
            self.sync()  # we need to make sure the previous store_global is finished
            self.release_semaphore(
                semaphore, value=(self.blockIdx.z + 1) % self.split_k_factor
            )


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        x_dtype: DataType,
        w_dtype: DataType,
        group_size: int,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        assert x_dtype in [float16, bfloat16], (
            "this kernel only supports float16/bfloat16 as activation data type"
        )
        assert w_dtype.is_float() or (w_dtype.is_integer() and w_dtype.min_value < 0), (
            "this kernel only supports symmetric quantization, which requires signed quantized weight"
        )
        assert in_features % group_size == 0, (
            "out_features must be divisible by group_size"
        )
        assert in_features * out_features * w_dtype.nbits % 8 == 0, (
            "in_features * out_features * w_dtype.nbits must be divisible by 8"
        )

        self.x_dtype = x_dtype
        self.w_dtype = w_dtype
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight_tile = (16, 16)
        self.change_layout_kernel = QuantizedMatmulChangeLayout(
            weight_tile=self.weight_tile,
            a_dtype=x_dtype,
            b_dtype=w_dtype,
        )
        self.restore_layout_kernel = QuantizedMatmulRestoreLayout(
            weight_tile=self.weight_tile,
            a_dtype=x_dtype,
            b_dtype=w_dtype,
        )
        self.quantized_matmul_kernel = QuantizedMatmul(
            weight_tile=self.weight_tile,
            group_size=group_size,
            a_dtype=x_dtype,
            b_dtype=w_dtype,
        )

        self.quantized_weight = nn.Parameter(
            torch.empty(
                size=[out_features * in_features * w_dtype.nbits // 8],
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.empty(
                size=[in_features // group_size, out_features],
                dtype=dtype_to_torch(x_dtype),
            ),
            requires_grad=False,
        )

    def load_and_quantize(self, weight: torch.Tensor):
        assert (self.out_features, self.in_features) == weight.shape, (
            "weight shape mismatch"
        )
        out_channels, in_channels, group_size = (
            self.out_features,
            self.in_features,
            self.group_size,
        )

        # convert weight to float32 cuda tensor with shape [in_channels, out_channels]
        weight = weight.cuda().float().transpose(1, 0).contiguous()

        # regroup the weight to [in_channels // group_size, group_size, out_channels]
        weight = weight.view(in_channels // group_size, group_size, out_channels)

        # get the maximum abs value of each group: [in_channels // group_size, 1, out_channels]
        max_weight: torch.Tensor = weight.abs().max(dim=1, keepdim=True)[0]
        scales = torch.max(
            max_weight / float(self.w_dtype.max_value),
            torch.tensor(1e-6, dtype=weight.dtype, device=weight.device),
        )
        scales = scales.to(dtype=dtype_to_torch(self.x_dtype))

        # compute the quantized weight (before rounding): [in_channels // group_size, group_size, out_channels]
        quantized_weight = weight / scales

        # clamp the quantized weight to the range of the quantized type
        quantized_weight = torch.clamp(
            quantized_weight,
            float(self.w_dtype.min_value),
            float(self.w_dtype.max_value),
        )

        # rounding to nearest number in the quantized type, but the data type is still float32
        if self.w_dtype.is_integer():
            quantized_weight = torch.round(quantized_weight)
        else:
            # for floating-point quantized weight, the rounding will be done in the casting
            quantized_weight = (
                tilus.from_torch(quantized_weight)
                .to(self.w_dtype)
                .to(tilus.float32)
                .torch()
            )

        # convert the quantized weight to the quantized type
        quantized_weight = tilus.from_torch(quantized_weight).to(self.w_dtype).storage

        # change the layout of quantized weight, and store it to the self.quantized_weight
        self.change_layout_kernel(
            self.in_features,
            self.out_features,
            quantized_weight.data_ptr(),
            self.quantized_weight.data_ptr(),
        )

        # save the scales type
        self.scales[:] = scales.view(in_channels // group_size, out_channels)

        # # validate the quantized weight
        # restored_weight = torch.empty_like(quantized_weight)
        # self.restore_layout_kernel(self.in_features, self.out_features, self.quantized_weight.data_ptr(), restored_weight.data_ptr())
        # tilus.from_torch(restored_weight).view(dtype=self.w_dtype, shape=(self.in_features, self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            size=[x.size(0), self.out_features], dtype=x.dtype, device=x.device
        )
        self.quantized_matmul_kernel(
            x.size(0),
            self.out_features,
            self.in_features,
            x.data_ptr(),
            self.quantized_weight.data_ptr(),
            self.scales.data_ptr(),
            out.data_ptr(),
        )
        return out


def main():
    headers = [
        "dtype",
        "m",
        "n",
        "k",
        "torch",
        "torch(TFLOPs)",
        "tilus",
        "tilus(TFLOPs)",
    ]
    group_size = 128
    workloads = []
    for k, n in [
        [4096, 4096 * 3],
        # [4096, 4096],
        # [4096, 14336 * 2],
        # [14336, 4096],
    ]:
        for m in [1, 16, 4096, 4097]:
            workloads.append([m, n, k])
    dtypes = [
        float8_e5m2,
        # float7_e4m2,
        float6_e3m2,
        # float5_e3m1,
        float4_e2m1,
        float3_e1m1,
        int8,
        # int7b,
        int6b,
        # int5b,
        int4b,
        int3b,
    ]

    rows = []
    for dtype in dtypes:
        for m, n, k in workloads:
            quantized_linear = QuantizedLinear(
                x_dtype=float16,
                w_dtype=dtype,
                group_size=group_size,
                in_features=k,
                out_features=n,
            ).cuda()
            a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
            b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
            c_expect = a @ b

            quantized_linear.load_and_quantize(b.T.contiguous())
            c_actual = quantized_linear(a)

            torch.cuda.synchronize()

            # check correctness
            torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)

            # benchmark
            row = [dtype.short_name, m, n, k]
            for name, func in [
                ("torch", lambda: torch.matmul(a, b, out=c_expect)),
                ("tilus", lambda: quantized_linear(a)),
            ]:
                latency = benchmark_func(func, warmup=5, repeat=20)
                tflops = 2 * m * n * k / latency * 1e-9
                row.append("{:.3f}".format(latency))
                row.append("{:.2f}".format(tflops))
            rows.append(row)

            df = pandas.DataFrame(rows, columns=headers)
            print(df)


if __name__ == "__main__":
    main()
