Shared Tensor
=============

A shared tensor (i.e., :py:class:`~tilus.ir.SharedTensor`) is a tensor stored in the shared memory of the GPU.

- **dtype**: the data type of the tensor elements, which can be any scalar type.
- **shape**: the shape of the tensor, which is a tuple of integers representing the size of each dimension.
- **layout**: (optional) the layout of the tensor, which defines how the tensor elements are stored in the linear shared memory.


Shared Tensor Instructions
--------------------------

We can use :py:meth:`~tilus.Script.shared_tensor` to define a shared tensor in Tilus Script.

.. code-block:: python

   self.shared_tensor(dtype=float32, shape=[32, 64])

The above code defines a shared tensor with the data type of 32-bit float and a shape of (32, 64).

Unlike register tensor, every shared tensor must be explicitly allocated using :py:meth:`~tilus.Script.shared_tensor`, and
explicitly freed using :py:meth:`~tilus.Script.free_shared` when it is no longer needed. Before freeing the shared tensor,
we must ensure that there are no pending asynchronous operations on the shared tensor (see below).

We have a bunch of instructions related to shared tensors:

.. py:currentmodule:: tilus.Script


**Allocate and Free**

.. autosummary::

    shared_tensor
    free_shared

**Load and Store**

.. autosummary::

    load_shared
    store_shared

**Asynchronous Copy from Global Tensor**

.. autosummary::

    copy_async
    copy_async_commit_group
    copy_async_wait_group
    copy_async_wait_all

We do not provide arithmetic instructions for shared tensors. To perform computation on shared tensors, we must first
load the data into register tensors using the :py:meth:`~tilus.Script.load_shared` instruction,
perform the computation on the register tensors,
and then store the results back to shared memory using the :py:meth:`~tilus.Script.store_shared` instruction.


Shared Layout
-------------
The :py:meth:`~tilus.Script.shared_tensor` instruction has an optional parameter ``layout`` that can be specified to
define the layout of the tensor, but it is not required. When not specified, the layout will be inferred automatically
based on the shape, data type and the instructions operating on the tensor.

A shared layout, :py:class:`~tilus.ir.SharedLayout`, defines how the tensor elements are stored in the linear shared memory.
You can think of the shared layout as a mapping from the multi-dimensional shape of the tensor to a linear memory
address in the shared memory.

All threads in the thread block can access the shared memory. However, to achieve the best performance, we need to take
care of the access patterns of the threads to the shared memory to avoid `bank conflicts <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory>`_.
The access patterns of the threads to the shared memory are determined by both the layout of the shared tensor and the
layout of the register tensor that will interact with the shared tensor. Our layout inference system will try to infer
the best layout for the shared tensor based on the access patterns of the threads (like automatically employing a swizzle layout).
But it's okay if the user wants to control the layout of the shared tensor manually for more fine-grained control of their kernel,
