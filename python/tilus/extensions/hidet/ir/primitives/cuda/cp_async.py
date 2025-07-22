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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=line-too-long
from typing import Optional, no_type_check

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.ir.primitives.func import primitive_func_pool, register_primitive_function
from hidet.ir.stmt import asm
from hidet.ir.type import void_p
from hidet.utils import initialize


def resolve_name_cp_async(
    use_shared_space_dst: bool,
    cp_size: int,
    cache_level: str = "always",
    evict: Optional[str] = None,
    prefetch_bytes: int = 0,
) -> str:
    if evict is None:
        evict_part = ""
    elif evict == "evict_first":
        evict_part = "_evict_first"
    else:
        assert False
    if prefetch_bytes:
        prefetch_part = "_l2_{}B".format(prefetch_bytes)
    else:
        prefetch_part = ""
    cache_part = "c" + cache_level[0]  # 'ca' or 'cg'
    dst_space = "_shared_dst" if use_shared_space_dst else "generic_dst"
    return "cp_async_size_{}_{}{}{}{}".format(cp_size, evict_part, cache_part, prefetch_part, dst_space)


@initialize()
def register_cp_async():
    from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
    from hidet.lang import attrs, i32, script

    for cp_size in [4, 8, 16]:
        for prefetch_bytes in [0, 64, 128, 256]:
            for cache_level in ["always", "global"]:
                for evict in [None, "evict_first"]:
                    if cache_level == "global" and cp_size != 16:
                        # cache level 'global' only support copy size of 16 bytes.
                        continue
                    if evict == "evict_first":
                        template_string = (
                            "{{\n"
                            "    .reg .b64 p;\n"
                            "    createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
                            "    cp.async.{cache_level}.shared.global.L2::cache_hint{prefetch} [%0], [%1], %2, %3, p;\n"
                            "}}\n"
                        )
                    else:
                        template_string = "cp.async.{cache_level}.shared.global{prefetch} [%0], [%1], %2, %3;"
                    template_string = template_string.format(
                        cache_level={"always": "ca", "global": "cg"}[cache_level],
                        prefetch=".L2::{}B".format(prefetch_bytes) if prefetch_bytes != 0 else "",
                    )

                    func_name = "cuda_" + resolve_name_cp_async(False, cp_size, cache_level, evict, prefetch_bytes)

                    @no_type_check
                    @script
                    def cuda_cp_async(generic_dst: void_p, src: void_p, src_size: i32):
                        attrs.func_name = func_name
                        attrs.func_kind = "cuda_internal"
                        dst_smem_ptr = cvta_generic_to_shared(generic_dst)
                        asm(template=template_string, inputs=[dst_smem_ptr, src, cp_size, src_size])

                    assert isinstance(cuda_cp_async, Function)
                    register_primitive_function(name=cuda_cp_async.name, func_or_type=cuda_cp_async)

                    func_name = "cuda_" + resolve_name_cp_async(True, cp_size, cache_level, evict, prefetch_bytes)

                    @no_type_check
                    @script
                    def cuda_cp_async(shared_dst: int32, src: void_p, src_size: i32):
                        attrs.func_name = func_name
                        attrs.func_kind = "cuda_internal"
                        asm(template=template_string, inputs=[shared_dst, src, cp_size, src_size])

                    assert isinstance(cuda_cp_async, Function)
                    register_primitive_function(name=cuda_cp_async.name, func_or_type=cuda_cp_async)


@initialize()
def register_cp_async_commit_group():
    from hidet.lang import attrs, script

    @no_type_check
    @script
    def cuda_cp_async_commit_group():
        attrs.func_name = "cuda_cp_async_commit_group"
        attrs.func_kind = "cuda_internal"
        asm("cp.async.commit_group;", is_volatile=True)

    assert isinstance(cuda_cp_async_commit_group, Function)
    del primitive_func_pool.name2func[cuda_cp_async_commit_group.name]
    register_primitive_function(cuda_cp_async_commit_group.name, cuda_cp_async_commit_group)


@initialize()
def register_cp_async_wait_group():
    from hidet.lang import attrs, script

    for groups in range(10):
        func_name = "cuda_cp_async_wait_group_{}".format(groups)

        @no_type_check
        @script
        def cuda_cp_async_wait_group():
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            asm("cp.async.wait_group {};".format(groups), is_volatile=True)

        assert isinstance(cuda_cp_async_wait_group, Function)
        del primitive_func_pool.name2func[cuda_cp_async_wait_group.name]
        register_primitive_function(cuda_cp_async_wait_group.name, cuda_cp_async_wait_group)


@initialize()
def register_cp_async_wait_all():
    from hidet.lang import attrs, script

    @no_type_check
    @script
    def cuda_cp_async_wait_all():
        attrs.func_name = "cuda_cp_async_wait_all"
        attrs.func_kind = "cuda_internal"
        asm("cp.async.wait_all;", is_volatile=True)

    assert isinstance(cuda_cp_async_wait_all, Function)
    del primitive_func_pool.name2func[cuda_cp_async_wait_all.name]
    register_primitive_function(cuda_cp_async_wait_all.name, cuda_cp_async_wait_all)


def cp_async(
    dst: Expr,
    src: Expr,
    cp_size: int,
    use_shared_space_dst: bool = False,
    src_size: Optional[int | Expr] = None,
    cache_level: str = "always",
    evict: Optional[str] = None,
    prefetch_bytes: int = 0,
) -> Expr:
    """
    Copy data from global memory to shared memory asynchronously.

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

    Parameters
    ----------
    dst: Expr
        The address of the destination in shared memory.
    src: Expr
        The address of the source in global memory.
    cp_size: int
        The number of bytes to be copied to the destination. Candidates: 4, 8 and 16.
    use_shared_space_dst: bool
        Whether the dst is in shared memory space. If True, the dst should be an uint32 address in shared memory space.
        Otherwise, the dst should be in the generic memory space and will be converted to shared memory space inside the
        primitive function.
    src_size: Union[Expr, int], optional
        The number of bytes in the source to be copied. If src_size < cp_size, the remaining part of destination will be filled with 0.
    cache_level: str
        The cache level. Candidates: 'always' and 'global'. When cache_level is 'global', the cp_size must be 16.
    evict: Optional[str]
        The evict priority. Can be 'evict_first', or None
    prefetch_bytes: int
        The number of bytes to be prefetched in L2 cache. Candidates: 0, 64, 128, 256.

    Returns
    -------
    ret: Expr
        The call expression.
    """
    if not (isinstance(cp_size, int) and cp_size in [4, 8, 16]):
        raise ValueError("cp_size must be either 4, 8, or 16, got {}.".format(cp_size))
    if not (isinstance(prefetch_bytes, int) and prefetch_bytes in [0, 64, 128, 256]):
        raise ValueError("prefetch_bytes must be either None, 64, 128 or 256, got {}.".format(prefetch_bytes))
    if cache_level not in ["global", "always"]:
        raise ValueError("Cache level candidates: {}, got {}".format(["always", "global"], cache_level))
    if evict not in [None, "evict_first"]:
        raise ValueError("Evict candidates: {}, got {}".format([None, "evict_first"], evict))
    if cache_level == "global":
        if cp_size != 16:
            raise ValueError("When cache_level is global, the cp_size must be 16, got {}".format(cp_size))
    if src_size is None:
        src_size = int32.constant(cp_size)
    elif isinstance(src_size, int):
        src_size = int32.constant(src_size)
    assert isinstance(src_size, Expr)

    func_name = resolve_name_cp_async(use_shared_space_dst, cp_size, cache_level, evict, prefetch_bytes)
    return call_cuda(func_name, [dst, src, src_size])
