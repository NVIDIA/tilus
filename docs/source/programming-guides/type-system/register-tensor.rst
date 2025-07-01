Register Tensor
===============

A register tensor (i.e., :py:class:`~tilus.ir.RegisterTensor`) is a tensor stored in the registers of the GPU.
It is stored among threads in the thread block in a distributed manner. Each register tensor have the following properties:

- **dtype**: the data type of the tensor elements, which can be any scalar type.
- **shape**: the shape of the tensor, which is a tuple of integers representing the size of each dimension.
- **layout**: (optional) the layout of the tensor, which defines how the tensor elements are stored among the threads.


Defining a Register Tensor
--------------------------

We can use :py:meth:`~tilus.Script.register_tensor` to define a register tensor in Tilus Script. In most cases, we do
not need to specify the layout, as it could be inferred automatically with our layout inference system (See :doc:`layout-inference`).

.. code-block:: python

   self.register_tensor(dtype=float32, shape=[32, 64])

The above code defines a register tensor with the data type of 32-bit float and a shape of (32, 64).

