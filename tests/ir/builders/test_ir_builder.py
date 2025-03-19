from tilus.ir.builders import IRBuilder

from hidet.ir.dtypes import int32


def test_program_builder():
    ib = IRBuilder()

    with ib.program():
        with ib.function(name="hello", num_warps=1, params={"n": int32}) as n:
            ib.num_blocks = 1
            ib.printf("Hello, world!\n")
            ib.printf("n = %d\n", n)

    program = ib.flush_program()
    print(program)
