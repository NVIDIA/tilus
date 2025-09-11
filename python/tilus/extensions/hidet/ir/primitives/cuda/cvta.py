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
from hidet.utils import initialize
from hidet.ir.type import PointerType, VoidType
from hidet.ir.expr import Expr
from hidet.ir.stmt import asm
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.lang import script


def resolve_cvta_func_name(src_space: str, dst_space: str) -> str:
    return 'cvta_{}_to_{}'.format(src_space, dst_space)


@initialize()
def register_cvta_instructions():
    from hidet.lang import attrs, u32

    for src_space in ['generic']:
        for dst_space in ['cta_shared', 'cluster_shared']:
            func_name = 'cuda_' + resolve_cvta_func_name(src_space, dst_space)

            if src_space == 'generic' and dst_space == 'cta_shared':
                template = "{.reg.u64 smem_ptr; cvta.to.shared::cta.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr;}"
            elif src_space == 'generic' and dst_space == 'cluster_shared':
                template = "{.reg.u64 smem_ptr; cvta.to.shared::cluster.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr;}"
            else:
                raise NotImplementedError()

            @script
            def cvta(src: PointerType(VoidType())) -> u32:
                attrs.func_name = func_name
                attrs.func_kind = 'cuda_internal'
                ret: u32 = 0
                asm(
                    template=template,
                    outputs=[ret],
                    inputs=[src],
                )
                return ret

            register_primitive_function(name=cvta.name, func_or_type=cvta)


def cvta_generic_to_shared(generic_addr: Expr) -> Expr:
    """
    Convert the address from generic memory space to shared memory space.

    In PTX, there are 5 memory space: generic, const, param, local, and shared. The later four are in side
    the generic memory space, and each is models as a window in generic space. The cvta (convert address)
    instructions are used to convert the address between these memory spaces.

    See Also:
    1. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#generic-addressing
    2. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta

    Parameters
    ----------
    generic_addr: Expr
        The address in generic memory space, should be a pointer.

    Returns
    -------
    ret: Expr
        The corresponding address in shared memory space. The returned address is an unsigned integer representing
        the address in shared memory space.
    """
    func_name = resolve_cvta_func_name(src_space='generic', dst_space='cta_shared')
    return call_cuda(func_name, args=[generic_addr])

def cvta_generic_to_cluster_shared(generic_addr: Expr) -> Expr:
    func_name = resolve_cvta_func_name(src_space='generic', dst_space='cluster_shared')
    return call_cuda(func_name, args=[generic_addr])

