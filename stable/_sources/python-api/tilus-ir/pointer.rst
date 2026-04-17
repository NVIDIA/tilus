tilus.ir.PointerType
====================


.. py:class:: tilus.ir.PointerType

    A pointer type that stores the address to a value of a specific data type.

    .. py:method:: __invert__(self)

        Create a new pointer type that points to the current pointer type.

        :return: A new pointer type that points to the current pointer type.
        :rtype: tilus.ir.PointerType


.. py:data:: tilus.ir.void_p

    A pointer to an unspecified data type, often used as a generic pointer type.

    :type: tilus.ir.PointerType
