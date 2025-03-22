import pytest
import tilus
from tilus.ir.tools import VerificationError, verify


class DemoLoadShared(tilus.Script):
    def __init__(self, shared_shape, register_shape):
        super().__init__()
        self.shared_shape = shared_shape
        self.register_shape = register_shape

    def __call__(self):
        self.attrs.warps = 2
        self.attrs.blocks = 1

        regs = self.register_tensor(dtype=tilus.float32, shape=self.register_shape)
        smem = self.shared_tensor(dtype=tilus.float32, shape=self.shared_shape)
        self.load_shared(src=smem, out=regs)


@pytest.mark.parametrize(
    "shared_shape, register_shape, success",
    [
        ([4, 4], [8, 8], False),
        ([8, 8], [8, 8], True),
        ([16, 16], [8, 8], True),
    ],
)
def test_verify_load_shared(shared_shape, register_shape, success):
    script = DemoLoadShared(shared_shape=shared_shape, register_shape=register_shape)
    program = script.program()

    if success:
        verify(program)
    else:
        try:
            verify(program)
        except VerificationError:
            pass
        else:
            assert False
