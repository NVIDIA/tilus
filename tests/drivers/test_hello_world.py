from tilus.drivers import build_program
from tilus.ir.builders import IRBuilder
from tilus.runtime import CompiledProgram


def test_compile_hello_world():
    ib = IRBuilder()

    with ib.function("hello_world", num_warps=1, params=[]):
        ib.num_blocks = [1]
        ib.printf("Hello, world!\n")

    program = ib.flush_program()

    compiled_module: CompiledProgram = build_program(program)

    compiled_module()
