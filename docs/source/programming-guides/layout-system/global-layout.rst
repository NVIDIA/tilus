Global Layout
=============

Global layout defines how tilus get the address of an element in a global tensor. When we use :py:meth:`~tilus.Script.global_view` to
create a global tensor (view) from a pointer, or when we use :py:meth:`~tilus.Script.global_tensor` to allocate a global tensor, we can optionally specify the layout
of the global tensor via ``layout`` or ``strides`` argument. There are different cases:

- We do not provide either ``layout`` or ``strides`` argument, then tilus will use the default layout, which is a row-major compact layout.
- We provide ``strides`` argument, then tilus will use the provided strides to construct a layout that using the provided strides to access the elements in the global tensor.
- We provide ``layout`` argument, then tilus will use the provided layout directly.

No matter which case, tilus will generate a :py:class:`~tilus.ir.GlobalLayout` object to represent the layout of the global tensor.
To create a custom global layout, we can use the :py:meth:`~tilus.ir.GlobalLayout.create` method of the :py:class:`~tilus.ir.GlobalLayout` class,
which allows us to define a custom mapping function from indices to offset.

We support the following functions to create a global layout:

.. autosummary::

    tilus.ir.layout.global_row_major
    tilus.ir.layout.global_column_major
    tilus.ir.layout.global_strides
    tilus.ir.layout.global_compose
    tilus.ir.GlobalLayout.create

Please refer to the :py:class:`~tilus.ir.GlobalLayout` class for internals of the global layout.
