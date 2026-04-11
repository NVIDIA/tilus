Shared Layout
=============

Shared layout defines how tilus gets the offset of a tensor element in the shared memory. We can specify the layout of a shared
tensor using the ``layout`` argument in :py:meth:`~tilus.Script.shared_tensor`.
If we do not provide the ``layout`` argument, tilus will try to infer the layout based on the usage of the shared tensor.

The difference between the shared layout and the global layout is that we require the shared layout have a constant shape,
while the global layout supports symbolic shape. Another difference is that the offset computation in shared layout can
involve the invariant variables that are invariant in the lifetime (i.e., the time after the shared tensor allocation
and before the shared tensor free), while the global layout only supports grid-invariant variables.

We support the following functions to create a global layout:

.. autosummary::

    tilus.ir.layout.shared_row_major
    tilus.ir.layout.shared_column_major
    tilus.ir.layout.shared_compose
    tilus.ir.SharedLayout.create

Please refer to the :py:class:`~tilus.ir.SharedLayout` class for internals of the shared layout.
