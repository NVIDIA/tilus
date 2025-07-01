tilus.ir.DataType
=================


.. py:class:: tilus.ir.DataType


    .. py:method:: __eq__(other)

        Compares this data type with another data type.

        :param other: The other data type to compare against.
        :type other: tilus.ir.DataType

        :return: True if the data types are equivalent, False otherwise.
        :rtype: bool


    .. py:property:: name

        The name of the data type.

        :type: str

    .. py:property:: nbytes

        The number of bytes in the data type.

        :type: int

    .. py:property:: nbits

        The number of bits in the data type.

        :type: int

    .. py:property:: one

        A constant with value 1 of this data type.

        :type: tilus.ir.Constant

    .. py:property:: zero

        A constant with value 0 of this data type.

        :type: tilus.ir.Constant

    .. py:method:: constant

        Create a constant of this data type.

        :param value: The value of the constant.
        :type value: int, float, bool, or str (for string types)

        :return: A constant of this data type.
        :rtype: tilus.ir.Constant


Supported Data Types
--------------------

.. py:data:: tilus.int64

    A 64-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int32

    A 32-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int16

    A 16-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int8

    An 8-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint64

    A 64-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint32

    A 32-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint16

    A 16-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint8

    An 8-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.float64

    A 64-bit floating-point data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.float32

    A 32-bit floating-point data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.float16

    A 16-bit floating-point data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.bfloat16

    A 16-bit brain floating-point data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.float8_e4m3

    An 8-bit floating-point data type with exponent bias of 4 and 3 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float8_e5m2

    An 8-bit floating-point data type with exponent bias of 5 and 2 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.boolean

    A boolean data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int1b

    A 1-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int2b

    A 2-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int3b

    A 3-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int4b

    A 4-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int5b

    A 5-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int6b

    A 6-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int7b

    A 7-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.int8b

    An 8-bit signed integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint1b

    A 1-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint2b

    A 2-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint3b

    A 3-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint4b

    A 4-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint5b

    A 5-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint6b

    A 6-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.uint7b

    A 7-bit unsigned integer data type.

    :type: tilus.ir.DataType

.. py:data:: tilus.float7_e5m1

    An 7-bit floating-point data type with exponent bias of 5 and 1 bit for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float7_e4m2

    An 7-bit floating-point data type with exponent bias of 4 and 2 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float7_e3m3

    An 7-bit floating-point data type with exponent bias of 3 and 3 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float7_e2m4

    An 7-bit floating-point data type with exponent bias of 2 and 4 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float6_e4m1

    A 6-bit floating-point data type with exponent bias of 4 and 1 bit for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float6_e3m2

    A 6-bit floating-point data type with exponent bias of 3 and 2 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float6_e2m3

    A 6-bit floating-point data type with exponent bias of 2 and 3 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float5_e3m2

    A 5-bit floating-point data type with exponent bias of 3 and 2 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float5_e3m1

    A 5-bit floating-point data type with exponent bias of 3 and 1 bit for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float5_e2m2

    A 5-bit floating-point data type with exponent bias of 2 and 2 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float4_e2m1

    A 4-bit floating-point data type with exponent bias of 1 and 2 bits for the mantissa.

    :type: tilus.ir.DataType

.. py:data:: tilus.float3_e1m1

    A 3-bit floating-point data type with exponent bias of 1 and 1 bit for the mantissa.

    :type: tilus.ir.DataType
