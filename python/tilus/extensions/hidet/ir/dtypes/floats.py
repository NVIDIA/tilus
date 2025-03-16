from hidet.ir.dtypes.floats import FloatType
from hidet.ir.dtypes.floats import float64, float32, tfloat32, bfloat16, float16

float8_e4m3 = FloatType(
    "float8_e4m3", "f8e4m3", 1, min_value=float(-448), max_value=float(448), eps=2 ** (-2), smallest_normal=2 ** (-6)
)
float8_e5m2 = FloatType(
    "float8_e5m2",
    "f8e5m2",
    1,
    min_value=float(-57344),
    max_value=float(57344),
    eps=2 ** (-2),
    smallest_normal=2 ** (-14),
)

# todo: add the mantissa and exponent bits
_mantissa_bits = {
    "float64": 52,
    "float32": 23,
    "tfloat32": 10,
    "float16": 10,
    "bfloat16": 7,
    "float8_e5m2": 2,
    "float8_e4m3": 3,
}
_exponent_bits = {
    "float64": 11,
    "float32": 8,
    "tfloat32": 8,
    "float16": 5,
    "bfloat16": 8,
    "float8_e5m2": 5,
    "float8_e4m3": 4,
}

for float_dtype in [float64, float32, tfloat32, bfloat16, float16]:
    float_dtype.mantissa_bits = _mantissa_bits[float_dtype.name]  # type: ignore
    float_dtype.exponent_bits = _exponent_bits[float_dtype.name]  # type: ignore
