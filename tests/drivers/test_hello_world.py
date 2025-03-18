from hidet.runtime import CompiledModule
from tilus.ir.builders import IRBuilder
from tilus.drivers import build_program


def test_compile_hello_world():
    ib = IRBuilder()

    with ib.function("hello_world", num_warps=1, params=[]):
        ib.num_blocks = [1]
        ib.printf("Hello, world!\n")

    program = ib.flush_program()

    compiled_module: CompiledModule = build_program(program)

    compiled_module()
