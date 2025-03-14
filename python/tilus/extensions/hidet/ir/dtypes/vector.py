from hidet.ir.type import DataType
from hidet.ir.dtypes.vector import VectorType
from hidet.ir.dtypes.floats import float16, float32
from hidet.ir.dtypes.integer import int8, uint8, uint16, uint32, uint64
from hidet.ir.dtypes.vector import float32x2, float32x4, float32x8, float16x2
from .integer_subbyte import uint4b

float32x1 = VectorType(float32, 1)
f32x1 = float32x1

float16x1 = VectorType(float16, 1)
f16x1 = float16x1

float16x4 = VectorType(float16, 4)
f16x4 = float16x4

float16x8 = VectorType(float16, 8)
f16x8 = float16x8

int8x4 = VectorType(int8, 4)
i8x4 = int8x4

uint4bx2 = VectorType(uint4b, 2)
u4x2 = uint4bx2

uint4bx8 = VectorType(uint4b, 8)
u4x8 = uint4bx8

uint8x1 = VectorType(uint8, 1)
u8x1 = uint8x1

uint8x2 = VectorType(uint8, 2)
u8x2 = uint8x2

uint8x4 = VectorType(uint8, 4)
u8x4 = uint8x4

uint16x1 = VectorType(uint16, 1)
u16x1 = uint16x1

uint16x2 = VectorType(uint16, 2)
u16x2 = uint16x2

uint16x4 = VectorType(uint16, 4)
u16x4 = uint16x4

uint32x1 = VectorType(uint32, 1)
u32x1 = uint32x1

uint32x2 = VectorType(uint32, 2)
u32x2 = uint32x2

uint32x4 = VectorType(uint32, 4)
u32x4 = uint32x4

uint64x1 = VectorType(uint64, 1)
u64x1 = uint64x1

uint64x2 = VectorType(uint64, 2)
u64x2 = uint64x2

uint64x4 = VectorType(uint64, 4)
u64x4 = uint64x4

_vectorize_table = {
    (float32, 1): float32x1,
    (float32, 2): float32x2,
    (float32, 4): float32x4,
    (float32, 8): float32x8,
    (float16, 1): float16x1,
    (float16, 2): float16x2,
    (float16, 4): float16x4,
    (float16, 8): float16x8,
    (int8, 4): int8x4,
    (uint8, 1): uint8x1,
    (uint8, 2): uint8x2,
    (uint8, 4): uint8x4,
    (uint16, 1): uint16x1,
    (uint16, 2): uint16x2,
    (uint16, 4): uint16x4,
    (uint32, 1): uint32x1,
    (uint32, 2): uint32x2,
    (uint32, 4): uint32x4,
    (uint64, 1): uint64x1,
    (uint64, 2): uint64x2,
    (uint64, 4): uint64x4,
}


def vectorize(base_dtype: DataType, num_lanes: int) -> VectorType:
    if (base_dtype, num_lanes) in _vectorize_table:
        return _vectorize_table[(base_dtype, num_lanes)]
    else:
        raise ValueError("Cannot vectorize {}x{}".format(base_dtype, num_lanes))
