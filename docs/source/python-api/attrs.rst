Script.attrs
============

The ``attrs`` object on a :class:`~tilus.Script` instance is used to set kernel launch configuration.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Attribute
     - Type
     - Description
   * - ``blocks``
     - ``Sequence[int] | int | None``
     - The grid dimensions of the kernel launch. Set as a list of up to 3 integers, e.g., ``self.attrs.blocks = [grid_x, grid_y]``.
   * - ``cluster_blocks``
     - ``Sequence[int] | int``
     - The cluster dimensions. Defaults to ``(1, 1, 1)``.
   * - ``warps``
     - ``int | None``
     - The number of warps per thread block. Must be a compile-time constant.
