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
from hidet.ir.type import PointerType, VoidType, void_p
from hidet.ir.dtypes import uint16, uint32
from hidet.ir.expr import Expr
from hidet.ir.stmt import asm
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.lang import script


def resolve_mapa_func_name(space: str) -> str:
    if space not in ('generic', 'shared'):
        raise ValueError(f'Unsupported memory space: {space}')
    return 'mapa_{}'.format(space)


@initialize()
def register_mapa_instructions():
    from hidet.lang import attrs, u32

    @script
    def mapa_generic(ptr: void_p, cta_rank: uint32) -> void_p:
        attrs.func_name = 'cuda_mapa_generic'
        attrs.func_kind = 'cuda_internal'
        ret: void_p = 0
        asm(
            template='mapa.u64 %0, %1, %2;',
            outputs=[ret],
            inputs=[ptr, cta_rank]
        )
        return ret

    @script
    def mapa_shared(ptr: uint32, cta_rank: uint32) -> uint32:
        attrs.func_name = 'cuda_mapa_shared'
        attrs.func_kind = 'cuda_internal'
        ret: uint32 = 0
        asm(
            template='mapa.shared::cluster.u32 %0, %1, %2;',
            outputs=[ret],
            inputs=[ptr, cta_rank]
        )
        return ret

    for func in [mapa_generic, mapa_shared]:
        register_primitive_function(name=func.name, func_or_type=func)

def mapa_generic(ptr: Expr, cta_rank: Expr | int) -> Expr:
    return call_cuda('mapa_generic', args=[ptr, uint32(cta_rank)])

def mapa_shared(ptr: Expr, cta_rank: Expr | int) -> Expr:
    return call_cuda('mapa_shared', args=[ptr, uint32(cta_rank)])

