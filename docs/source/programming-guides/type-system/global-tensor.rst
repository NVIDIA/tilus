Global Tensor
=============

A global tensor (i.e., :py:class:`~tilus.ir.GlobalTensor`) is a tensor stored in the global memory of the GPU. It has
the following attributes:

- **dtype**: the data type of the tensor elements, which can be any scalar type.
- **shape**: the shape of the tensor, which is a tuple of integers representing the size of each dimension. The dimension
  sizes can be constant or any grid-invariant expressions such as the kernel parameters.
- **layout**: the layout of the tensor, which defines how the tensor elements are stored in the linear global memory.

Defining a Global Tensor
------------------------

We usually use pointers as kernel parameters and use :py:meth:`~tilus.Script.global_view` to define a global tensor
with global pointer to the first element of the tensor.

Besides, we can also use :py:meth:`~tilus.Script.global_tensor` to allocate a global tensor shared by all thread blocks
in the kernel. The global tensor is managed by the runtime system and has a lifetime that spans the entire kernel
execution.

Loading and Storing Global Tensors
----------------------------------

We can use the following instructions to load and store global tensors:

.. py:currentmodule:: tilus.Script

**Load and Store**

.. autosummary::

    load_global
    store_global

Similar to shared tensors, we do not provide arithmetic instructions for global tensors. To perform computation on
global tensors, we must first load the data into register tensors using the :py:meth:`~tilus.Script.load_global`
instruction, perform the computation on the register tensors, and then store the results back to global memory using
the :py:meth:`~tilus.Script.store_global` instruction.

Global Layout
-------------

Each global tensor has a layout that defines how the tensor elements are stored in the linear global memory. It can be
arbitrary mapping from the multi-dimensional indices to the linear memory address.

The global tensor creation instructions (:py:meth:`~tilus.Script.global_view` and :py:meth:`~tilus.Script.global_tensor`)
have an optional parameter ``layout`` that can be specified to define the layout of the tensor.

There are several cases for specifying the layout of global tensors:

- Not provided: if neither ``layout`` nor ``strides`` is specified, we assume the compact row-major layout is used.
- ``strides`` provided: if the ``strides`` parameter is provided, it defines the strides of the tensor in each dimension.
  The strides are a tuple of integers representing the number of elements to skip in each dimension to get to the next
  element in that dimension.
- ``layout`` provided: if the ``layout`` parameter is provided, it defines the layout of the tensor. The layout can be
  any custom mapping from the multi-dimensional indices to the linear memory address.
