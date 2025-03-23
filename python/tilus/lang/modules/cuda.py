from hidet.ir.dtypes import DataType
from tilus.ir.inst import MmaConfig
from tilus.ir.layout import RegisterLayout, SharedLayout


class cuda:
    class mma:
        m16n8k16_f16_f32: MmaConfig = MmaConfig.m16n8k16_f16_f32()
        m16n8k16_f16_f16: MmaConfig = MmaConfig.m16n8k16_f16_f16()
        m16n8k16_bf16_f32: MmaConfig = MmaConfig.m16n8k16_bf16_f32()


def load_friendly_shared_layout(dtype: DataType, register_layout: RegisterLayout) -> SharedLayout:
    """
    Construct a shared layout that is efficient to be used as a bridge between global memory to register memory loading.

    Parameters
    ----------
    dtype: DataType
        The element data type for both the shared memory and the register memory.

    register_layout: RegisterLayout
        The register layout that will be used to store the data loaded from global memory.

    Returns
    -------
    shared_layout: SharedLayout
        The shared layout that is efficient to be used as a bridge between global memory to register memory loading.
    """
    pass
