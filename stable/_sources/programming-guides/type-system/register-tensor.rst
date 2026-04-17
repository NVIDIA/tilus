Register Tensor
===============

A register tensor (i.e., :py:class:`~tilus.ir.RegisterTensor`) is a tensor stored in the registers of the GPU.
It is stored among threads in the thread block in a distributed manner. Each register tensor have the following properties:

- **dtype**: the data type of the tensor elements, which can be any scalar type.
- **shape**: the shape of the tensor, which is a tuple of integers representing the size of each dimension.
- **layout**: (optional) the layout of the tensor, which defines how the tensor elements are stored among the threads.


Defining a Register Tensor
--------------------------

We can use :py:meth:`~tilus.Script.register_tensor` to define a register tensor in Tilus Script.

.. code-block:: python

   self.register_tensor(dtype=float32, shape=[32, 64])

The above code defines a register tensor with the data type of 32-bit float and a shape of (32, 64).

Register Layout
---------------

The :py:meth:`~tilus.Script.register_tensor` instruction has an optional parameter ``layout`` that can be specified to
define the layout of the tensor, but it is not required. When not specified, the layout will be inferred automatically
based on the shape, data type and the instructions operating on the tensor.

A register layout, :py:class:`~tilus.ir.RegisterLayout`, defines how the tensor elements are distributed among the
threads in the thread block. It determines what operations we can perform on the tensor and the performance of those
operations.
In most cases, we can rely on the automatic layout inference system to determine the best layout for the tensor.
We have a dedicated section talking about the layout system of Tilus Script in :doc:`../layout-system/__init__`.
Please refer to that section if our layout inference system does not pick the best layout for you or you want to
control the layout of the tensor manually for more fine-grained control of your kernel.
