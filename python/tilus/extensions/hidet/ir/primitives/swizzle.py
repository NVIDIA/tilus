from hidet.ir.expr import Expr
from hidet.utils import initialize
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func


@initialize()
def register_swizzle_primitive():
    from hidet.lang import script, attrs
    from hidet.ir.dtypes import int32

    @script
    def swizzle_impl(x: int32, mbase: int32, bbits: int32, sshift: int32) -> int32:
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "swizzle"

        return (x ^ ((x & ((1 << bbits) - 1) << (mbase + sshift)) >> sshift)) if bbits > 0 else x

    register_primitive_function(swizzle_impl.name, swizzle_impl)


def swizzle(x: Expr, mbase: Expr, bbits: Expr, sshift: Expr) -> Expr:
    """
    Using the swizzle from cute:

    0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase is the number of least-sig bits to keep constant
                 ^-^       ^-^     BBits is the number of bits in the mask
                   ^---------^     SShift is the distance to shift the YYY mask
                                      (pos shifts YYY to the right, neg shifts YYY to the left)

    e.g. Given
    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx

    the result is
    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ `xor` YY
    """
    return call_primitive_func('swizzle', [x, mbase, bbits, sshift])
