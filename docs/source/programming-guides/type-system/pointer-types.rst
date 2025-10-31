Pointer Types
=============

Tilus supports pointer types that represent addresses of data in memory. They are mainly used as parameters of a tilus
kernel to accept torch tensors.


The syntax to define a pointer type is as follows:

.. code-block:: python

    ~<dtype>

Where ``dtype`` is one of the scalar types (e.g., :py:data:`~tilus.float32`) in :doc:`scalar-types`, or a pointer type itself.

**Pointer to Scalar** We can define a pointer to a scalar type like ``~float32``.

**Pointer to Pointer** We can also define a pointer to another pointer type, like ``~(~float32))`` to represent a pointer to a pointer to a
32-bit float type.

**Void Pointer** A special pointer type :py:data:`tilus.ir.void_p`, which represents a pointer to an unspecified data
type, is often used as a generic pointer type.
