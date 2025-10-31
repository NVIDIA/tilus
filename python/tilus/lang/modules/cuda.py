# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
from dataclasses import dataclass
from typing import Optional, Sequence

import cuda.bindings.runtime as cudart
from hidet.ir.dtypes import DataType, bfloat16, float16, float32, int8, int32
from hidet.ir.expr import as_expr

from tilus import RegisterLayout
from tilus.backends.emitters.cuda.mma_dot import AtomicMmaConfig
from tilus.ir.layout import SharedLayout
from tilus.ir.layout.ops import auto_local_spatial, reduce, shared_compose, shared_row_major, spatial
from tilus.ir.utils import vector
from tilus.utils import gcd, idiv, prod


@dataclass(frozen=True, eq=False)
class BlockMmaConfig:
    operand_dtype: DataType
    accumulator_dtype: DataType
    m: int
    n: int
    k: int
    num_warps: int
    la: RegisterLayout
    lb: RegisterLayout
    lc: RegisterLayout

    @property
    def lb_T(self) -> RegisterLayout:
        from tilus.ir.layout.ops import permute

        return permute(self.lb, [1, 0])


class cuda:
    class atomic_mma_configs:
        m16n8k16_f16_f32: AtomicMmaConfig = AtomicMmaConfig.m16n8k16_f16_f32()
        m16n8k16_f16_f16: AtomicMmaConfig = AtomicMmaConfig.m16n8k16_f16_f16()
        m16n8k16_bf16_f32: AtomicMmaConfig = AtomicMmaConfig.m16n8k16_bf16_f32()
        m8n8k16_i8_i32: AtomicMmaConfig = AtomicMmaConfig.m8n8k16_i8_i32()
        m16n8k16_i8_i32: AtomicMmaConfig = AtomicMmaConfig.m16n8k16_i8_i32()
        m16n8k32_i8_i32: AtomicMmaConfig = AtomicMmaConfig.m16n8k32_i8_i32()

        @staticmethod
        @functools.lru_cache
        def from_dtypes(operand_dtype, accumulator_dtype):
            table = {
                (float16, float32): cuda.atomic_mma_configs.m16n8k16_f16_f32,
                (float16, float16): cuda.atomic_mma_configs.m16n8k16_f16_f16,
                (bfloat16, float32): cuda.atomic_mma_configs.m16n8k16_bf16_f32,
                (int8, int32): cuda.atomic_mma_configs.m16n8k32_i8_i32,
            }
            if (operand_dtype, accumulator_dtype) not in table:
                raise ValueError(
                    f"Unsupported MMA config for operand dtype {operand_dtype} and accumulator dtype {accumulator_dtype}"
                )
            return table[(operand_dtype, accumulator_dtype)]

    class runtime:
        _property_cache: dict[int, cudart.cudaDeviceProp] = {}

        @staticmethod
        @functools.lru_cache
        def get_device_properties(device_id: Optional[int] = 0) -> cudart.cudaDeviceProp:
            if device_id not in cuda.runtime._property_cache:
                errno, prop = cudart.cudaGetDeviceProperties(device_id)
                cuda.runtime._property_cache[device_id] = prop
            return cuda.runtime._property_cache[device_id]

    @staticmethod
    @functools.lru_cache
    def resolve_dot_config(
        operand_dtype: DataType,
        acc_dtype: DataType,
        *,
        m: int,
        n: int,
        k: int,
        num_warps: Optional[int] = None,
        warp_m: Optional[int] = None,
        warp_n: Optional[int] = None,
    ) -> BlockMmaConfig:
        if num_warps is None:
            if warp_m is None or warp_n is None:
                raise ValueError("num_warps must be specified if warp_m or warp_n is not specified.")
            num_warps = warp_m * warp_n
        atomic_mma = cuda.atomic_mma_configs.from_dtypes(operand_dtype, acc_dtype)
        if any(vector(m, n, k) % vector(atomic_mma.m, atomic_mma.n, atomic_mma.k) != 0):
            raise ValueError(f"block_m, block_n, block_k ({m}, {n}, {k}) must be multiples are illegal.")
        mma_count_m = m // atomic_mma.m
        mma_count_n = n // atomic_mma.n
        mma_count_k = k // atomic_mma.k

        spatial_repeat_candidates: list[tuple[tuple[int, int], tuple[int, int, int]]] = []
        for wsm in range(1, num_warps + 1):
            if num_warps % wsm != 0:
                continue
            wsn = num_warps // wsm
            warp_spatial = (wsm, wsn)
            if warp_m is not None and warp_m != wsm:
                continue
            if warp_n is not None and warp_n != wsn:
                continue
            if mma_count_m % wsm != 0 or mma_count_n % wsn != 0:
                continue
            warp_repeat = (mma_count_m // wsm, mma_count_n // wsn, mma_count_k)
            spatial_repeat_candidates.append((warp_spatial, warp_repeat))

        if len(spatial_repeat_candidates) == 0:
            raise ValueError(f"Can not find a proper spatial repeat for block_m, block_n, block_k ({m}, {n}, {k})")

        # for all spatial-local candidates, they share
        #   1. number of mma instructions in total for the block: mma_count_m * mma_count_n * mma_count_k
        #   2. number of warps: num_warps
        #   3. number of mma instructions per warp: mma_count_m * mma_count_n * mma_count_k / num_warps
        # they differ in
        #   1. the register per warp
        # we select the one with the minimum register per warp among all candidates
        def count_registers(candidate: tuple[tuple[int, int], tuple[int, int, int]]) -> int:
            # calculate the number of registers used by the candidate
            _, (wrm, wrn, wrk) = candidate
            a_bytes = wrm * wrk * atomic_mma.la.local_size * atomic_mma.operand_type.nbytes
            b_bytes = wrn * wrk * atomic_mma.lb.local_size * atomic_mma.operand_type.nbytes
            c_bytes = wrm * wrn * atomic_mma.lc.local_size * atomic_mma.acc_type.nbytes
            return (a_bytes + b_bytes + c_bytes) // 4

        best_candidate = min(spatial_repeat_candidates, key=count_registers)
        estimate_registers = count_registers(best_candidate)
        if estimate_registers >= 256 - 32:
            raise ValueError("The register usage ({}) of given config is too high.".format(estimate_registers))
        warp_spatial, warp_repeat = best_candidate
        return cuda.mma_dot_config(atomic_mma, warp_spatial, warp_repeat)

    @staticmethod
    @functools.lru_cache
    def mma_dot_config(
        atomic_mma: AtomicMmaConfig,
        warp_spatial: tuple[int, int],
        warp_repeat: tuple[int, int, int],
    ) -> BlockMmaConfig:
        wsm, wsn = warp_spatial
        wrm, wrn, wrk = warp_repeat
        block_m = atomic_mma.m * wsm * wrm
        block_n = atomic_mma.n * wsn * wrn
        block_k = atomic_mma.k * wrk * atomic_mma.vec_k
        num_warps = wsm * wsn
        layout_ra = reduce(spatial(wsm, 1, wsn, ranks=[1, 0, 2]), dims=[2]).local(wrm, wrk) * atomic_mma.la
        layout_rb = reduce(spatial(1, wsn, wsm, ranks=[0, 2, 1]), dims=[2]).local(wrk, wrn) * atomic_mma.lb
        layout_rc = spatial(wsm, wsn).local(wrm, wrn) * atomic_mma.lc

        def count_registers(candidate: tuple[tuple[int, int], tuple[int, int, int]]) -> int:
            # calculate the number of registers used by the candidate
            _, (wrm, wrn, wrk) = candidate
            a_bytes = wrm * wrk * atomic_mma.la.local_size * atomic_mma.operand_type.nbytes
            b_bytes = wrn * wrk * atomic_mma.lb.local_size * atomic_mma.operand_type.nbytes
            c_bytes = wrm * wrn * atomic_mma.lc.local_size * atomic_mma.acc_type.nbytes
            return (a_bytes + b_bytes + c_bytes) // 4

        if count_registers((warp_spatial, warp_repeat)) >= 256 - 32:
            raise ValueError(
                "The register usage ({}) of given config is too high.".format(
                    count_registers((warp_spatial, warp_repeat))
                )
            )

        return BlockMmaConfig(
            operand_dtype=atomic_mma.operand_type,
            accumulator_dtype=atomic_mma.acc_type,
            m=block_m,
            n=block_n,
            k=block_k,
            num_warps=num_warps,
            la=layout_ra,
            lb=layout_rb,
            lc=layout_rc,
        )

    @staticmethod
    def shared_layout(shape: Sequence[int]) -> SharedLayout:
        """Generate a row-major shared layout.

        Parameters
        ----------
        shape: Sequence[int]
            The shape of the shared layout.

        Returns
        -------
        shared_layout: SharedLayout
            The row-major shared layout with the given shape.
        """
        return shared_row_major(*shape)

    @staticmethod
    def swizzled_shared_layout(dtype: DataType, *, shape: Sequence[int]) -> SharedLayout:
        """
        Generate a shared layout that could be used to generate ldmatrix instruction when using LoadSharedInst.

        Both m and n must be a multiple of 8.

        We will divide each row into bank groups, and bank group has 16 bytes (16 x uint8, 8 x fp16, or 4 x fp32, etc.).
        They correspond to 4 banks in shared memory. For example, if we have m = n = 8, we can represent bank groups as

        0   # bank group 0, banks from 0 to 3
        1   # bank group 1, banks from 4 to 7
        2   # ...
        3
        4
        5
        6
        7   # bank groups 7, banks from 28 to 31

        Given m, and n, we need to find a proper way to organize the m x (n / 8) bank groups in shared memory, so that
        1) each row has different bank groups
        2) each column has different bank groups

        When we have m = 8 and n = 64, we have 8 x 8 bank groups. If we store the elements in row-major order, we will
        have the bank groups as

        0  1  2  3  4  5  6  7
        0  1  2  3  4  5  6  7
        0  1  2  3  4  5  6  7
        0  1  2  3  4  5  6  7
        0  1  2  3  4  5  6  7
        0  1  2  3  4  5  6  7
        0  1  2  3  4  5  6  7
        0  1  2  3  4  5  6  7

        If we use ldmatrix to load the above 8 x 64 shared memory, we will need 8 ldmatrix.v1 instructions. Each instruction
        loads one column (8 x 8 elements, or 8 x 1 bank groups). Since each instruction will access the same bank group,
        severe bank conflicts will occur. Thus, we need to change the layout of shared memory to avoid bank conflicts.

        Let layout(i, j) be the shared memory address of logical elements (each element has 16 bytes) when we use
        a specific `layout`. For example, the row-major layout row-major(i, j) = i * n + j * 8 (we assume the dtype has 2
        bytes). If we use the swizzled layout swizzled(i, j) = row-major(i, j ^ i) = i * n + (j ^ i) * 8, we can have the
        following bank groups in shared memory.

        0  1  2  3  4  5  6  7
        1  0  3  2  5  4  7  6
        2  3  0  1  6  7  4  5
        3  2  1  0  7  6  5  4
        4  5  6  7  0  1  2  3
        5  4  7  6  1  0  3  2
        6  7  4  5  2  3  0  1
        7  6  5  4  3  2  1  0

        (reader may need some time to figure out the above layout...)

        This layout has two benefits:
        1) Each row has different bank groups. In above example, we have 32 banks per row.
        2) Each column has different bank groups. In above example, we have 32 banks per column.

        The benefit 1 makes sure that when we load data from global memory to shared memory, we can store efficiently.
        The benefit 2 makes sure that when we load data from shared memory to register memory, we can load efficiently.

        We can always generate the swizzled layout for arbitrary m and n as long as they are multiple of 8. See the
        implementation for more details.

        Parameters
        ----------
        dtype: DataType
            The element data type for both the shared memory and the register memory.

        shape: Sequence[int]
            The shape of the shared memory. The shape must have at least two dimensions.

        Returns
        -------
        shared_layout: SharedLayout
            The shared layout that could be used to generate ldmatrix instruction when using LoadSharedInst.
        """
        return cuda._swizzled_shared_layout_new(dtype, shape=tuple(shape))

    @staticmethod
    @functools.lru_cache
    def _swizzled_shared_layout(dtype: DataType, shape: tuple[int, ...]) -> SharedLayout:
        if len(shape) < 2:
            raise ValueError("The shape of swizzled shared layout must have at least two dimensions.")
        m, n = shape[-2:]
        group_elements = idiv(16, dtype.nbytes)
        if m % 8 != 0 or n % group_elements != 0:
            raise ValueError("m must be a multiple of 8, and n must be a multiple of dtype.nbytes * 8.")
        rows = m
        columns = n // group_elements

        if columns % 8 == 0:
            """
            0 1 2 3 4 5 6 7
            1 0 3 2 5 4 7 6
            2 3 0 1 6 7 4 5
            3 2 1 0 7 6 5 4
            4 5 6 7 0 1 2 3
            5 4 7 6 1 0 3 2
            6 7 4 5 2 3 0 1
            7 6 5 4 3 2 1 0
            """
            core = shared_row_major(rows, columns).swizzle(dim=1, regards_dim=0, log_step=0)
        elif columns % 4 == 0:
            """
            0 1 2 3
            4 5 6 7
            1 0 3 2
            5 4 7 6
            2 3 0 1
            6 7 4 5
            3 2 1 0
            7 6 5 4
            """
            core = shared_row_major(rows, 4).swizzle(dim=1, regards_dim=0, log_step=1)
        elif columns % 2 == 0:
            """
            0 1
            2 3
            4 5
            6 7
            1 0
            3 2
            5 4
            7 6
            """
            core = shared_row_major(rows, 2).swizzle(dim=1, regards_dim=0, log_step=2)
        else:
            """
            0
            1
            2
            3
            4
            5
            6
            7
            """
            core = shared_row_major(rows, 1)
        layout = shared_compose(core, shared_row_major(1, group_elements))
        if m > layout.shape[0] or n > layout.shape[1]:
            layout = shared_compose(shared_row_major(m // layout.shape[0], n // layout.shape[1]), layout)
        if len(shape) > 2:
            for extent in reversed(shape[:-2]):
                layout = layout.prepend_dim(extent=extent)
        return layout

    @staticmethod
    @functools.lru_cache
    def _swizzled_shared_layout_new(dtype: DataType, shape: tuple[int, ...]) -> SharedLayout:
        if len(shape) < 2:
            raise ValueError("The shape of swizzled shared layout must have at least two dimensions.")
        m, n = shape[-2:]
        group_elements = idiv(16, dtype.nbytes)
        if m % 8 != 0 or n % group_elements != 0:
            raise ValueError("m must be a multiple of 8, and n must be a multiple of dtype.nbytes * 8.")

        def f_offset(axes):
            strides: list[int] = [prod(shape[i + 1 :]) for i in range(len(shape))]

            columns: int = n // group_elements
            columns_vec_size: int = gcd(columns, 8)

            i, j = axes[-2:]
            if columns_vec_size == 8:
                i, j = i, j ^ ((i % 8) * group_elements)
            elif columns_vec_size == 4:
                i, j = i, j ^ (i // 2 % 4 * group_elements)
            elif columns_vec_size == 2:
                i, j = i, j ^ (i // 4 % 2 * group_elements)
            else:
                i, j = i, j
            swizzled_axes = axes[:-2] + [i, j]
            offset = as_expr(sum(axis * stride for axis, stride in zip(swizzled_axes, strides)))
            return offset

        layout = SharedLayout.create(shape=shape, size=prod(shape), f_offset=f_offset).simplify()
        return layout

    @staticmethod
    def default_register_layout(
        num_warps: int, dtype: DataType, shape: Sequence[int], vector_size: Optional[int] = None
    ) -> RegisterLayout:
        return cuda._default_register_layout(num_warps, dtype, tuple(shape), vector_size)

    @staticmethod
    @functools.lru_cache
    def _default_register_layout(
        num_warps: int, dtype: DataType, shape: tuple[int, ...], vector_size: Optional[int] = None
    ) -> RegisterLayout:
        num_threads = num_warps * 32
        num_elements = prod(shape)
        if num_elements % num_threads != 0:
            raise RuntimeError(
                "Can not automatically generate register layout for shape {} and num_warps {}.".format(shape, num_warps)
            )
        elements_per_thread = num_elements // num_threads
        if vector_size is None:
            vector_size = gcd(elements_per_thread, 16 // dtype.nbytes, shape[-1])
        else:
            assert elements_per_thread % vector_size == 0

        if vector_size > 1:
            vector_shape = list(shape)
            vector_shape[-1] = shape[-1] // vector_size
            repeat_shape = [1 for _ in shape]
            repeat_shape[-1] = vector_size
            return auto_local_spatial(num_threads=num_threads, shape=vector_shape).local(*repeat_shape)
        else:
            return auto_local_spatial(num_threads=num_threads, shape=shape)
