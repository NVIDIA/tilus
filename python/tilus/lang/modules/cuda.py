from typing import Optional

from hidet.ir.dtypes import DataType
from tilus.ir.instructions.cuda import MmaDotConfig
from tilus.ir.layout import SharedLayout, shared_compose, shared_repeat
from tilus.utils import idiv


class cuda:
    class mma:
        m16n8k16_f16_f32: MmaDotConfig = MmaDotConfig.m16n8k16_f16_f32()
        m16n8k16_f16_f16: MmaDotConfig = MmaDotConfig.m16n8k16_f16_f16()
        m16n8k16_bf16_f32: MmaDotConfig = MmaDotConfig.m16n8k16_bf16_f32()

    @staticmethod
    def swizzled_shared_layout(dtype: DataType, *, m: int, n: int, bs: Optional[int] = None) -> SharedLayout:
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

        m: int
            The number of rows in the shared memory.

        n: int
            The number of columns in the shared memory.

        bs: Optional[int]
            The batch size of the shared memory. When it's not None, the returned layout will have three dimensions
            (bs, m, n). Default is None.

        Returns
        -------
        shared_layout: SharedLayout
            The shared layout that could be used to generate ldmatrix instruction when using LoadSharedInst.
        """
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
            core = shared_repeat(rows, columns).swizzle(dim=1, regards_dim=0, log_step=0)
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
            core = shared_repeat(rows, 4).swizzle(dim=1, regards_dim=0, log_step=1)
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
            core = shared_repeat(rows, 2).swizzle(dim=1, regards_dim=0, log_step=2)
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
            core = shared_repeat(rows, 1)
        layout = shared_compose(core, shared_repeat(1, group_elements))
        if m > layout.shape[0] or n > layout.shape[1]:
            layout = shared_compose(shared_repeat(m // layout.shape[0], n // layout.shape[1]), layout)
        if bs is not None:
            layout = layout.prepend_dim(extent=bs)
        return layout
