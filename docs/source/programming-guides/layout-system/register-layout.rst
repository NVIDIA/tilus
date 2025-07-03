Register Layout
===============

A Register Layout (i.e., :py:class:`~tilus.ir.RegisterLayout`) is a layout that defines how the elements of a register
tensor are stored among the local registers of all threads in the entire thread block. The key difference between global/shared
layouts and register layout is that shared layout is a **distributed layout**.

In global/shared layout, the layout defines how to get the position of an element in the global/shared memory. However,
for register layout, the layout defines which thread(s) are storing the element and the position of the element in the
local register memory of the thread(s).

Formally, a register layout could be defined as a mapping from (thread_id, local_id) to the logical index of the element
in the tensor:

.. math::

    \text{layout} : (\text{thread_id}, \text{local_id}) \mapsto \text{index}

where :math:`\text{thread_id}\in \mathbb{N}` is the ID of the thread in the thread block, :math:`\text{local_id} \in \mathbb{N}`
is the ID of the element in the thread local storage, and :math:`\text{index} \in \mathbb{N}^d` is the logical index
of the element in the $d$-dimension tensor. We can also equivalently view the register layout as a mapping from
the logical index to the (thread_id, local_id) pairs.


Getting Started
---------------

We begin with some simple register layout.

.. doctest::

    >>> from tilus.ir.layout import spatial, local, visualize_layout

.. doctest::

    >>> print(visualize_layout(local(3, 4)))
    RegisterLayout(shape=[3, 4], mode_shape=[3, 4], spatial_modes=[], local_modes=[0, 1])
    ┌──────┬──────┬───────┬───────┐
    │ 0: 0 │ 0: 1 │ 0: 2  │ 0: 3  │
    ├──────┼──────┼───────┼───────┤
    │ 0: 4 │ 0: 5 │ 0: 6  │ 0: 7  │
    ├──────┼──────┼───────┼───────┤
    │ 0: 8 │ 0: 9 │ 0: 10 │ 0: 11 │
    └──────┴──────┴───────┴───────┘

Each entry ``t : i`` in the above layout represents that the element is stored in thread ``t`` at local index ``i``.
The above layout is a simple layout that maps each element in the grid (3, 4) to a single thread (thread_id=0), and
the elements are stored in a row-major order in the local storage of the thread.

We have a spatial layout that maps the elements to multiple threads.

.. doctest::

    >>> print(visualize_layout(spatial(3, 2)))
    RegisterLayout(shape=[3, 2], mode_shape=[3, 2], spatial_modes=[0, 1], local_modes=[])
    ┌──────┬──────┐
    │ 0: 0 │ 1: 0 │
    ├──────┼──────┤
    │ 2: 0 │ 3: 0 │
    ├──────┼──────┤
    │ 4: 0 │ 5: 0 │
    └──────┴──────┘

Above layout is a spatial layout that maps the elements to 6 threads, where each thread holds a single element.

There are some attributes inside the ``RegisterLayout(...)`` object that uniquely defines the layout shown in the
grid diagram.


Layout Composition
------------------

We can `compose` two layouts together to create complex layouts. The intuitive idea of composition is that we can
replace each element in one layout with a tensor with another layout. For example, if each element in the ``local(3, 4)``
layout is replaced with a tensor with the ``spatial(2, 3)`` layout, we can get a new layout with shape ``(3 * 2, 4 * 3)``.

.. doctest::

    >>> print(visualize_layout(local(3, 4).spatial(2, 3)))
    RegisterLayout(shape=[6, 12], mode_shape=[3, 2, 4, 3], spatial_modes=[1, 3], local_modes=[0, 2])
    ┌──────┬──────┬──────┬──────┬──────┬──────┬───────┬───────┬───────┬───────┬───────┬───────┐
    │ 0: 0 │ 1: 0 │ 2: 0 │ 0: 1 │ 1: 1 │ 2: 1 │ 0: 2  │ 1: 2  │ 2: 2  │ 0: 3  │ 1: 3  │ 2: 3  │
    ├──────┼──────┼──────┼──────┼──────┼──────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 3: 0 │ 4: 0 │ 5: 0 │ 3: 1 │ 4: 1 │ 5: 1 │ 3: 2  │ 4: 2  │ 5: 2  │ 3: 3  │ 4: 3  │ 5: 3  │
    ├──────┼──────┼──────┼──────┼──────┼──────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 0: 4 │ 1: 4 │ 2: 4 │ 0: 5 │ 1: 5 │ 2: 5 │ 0: 6  │ 1: 6  │ 2: 6  │ 0: 7  │ 1: 7  │ 2: 7  │
    ├──────┼──────┼──────┼──────┼──────┼──────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 3: 4 │ 4: 4 │ 5: 4 │ 3: 5 │ 4: 5 │ 5: 5 │ 3: 6  │ 4: 6  │ 5: 6  │ 3: 7  │ 4: 7  │ 5: 7  │
    ├──────┼──────┼──────┼──────┼──────┼──────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 0: 8 │ 1: 8 │ 2: 8 │ 0: 9 │ 1: 9 │ 2: 9 │ 0: 10 │ 1: 10 │ 2: 10 │ 0: 11 │ 1: 11 │ 2: 11 │
    ├──────┼──────┼──────┼──────┼──────┼──────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 3: 8 │ 4: 8 │ 5: 8 │ 3: 9 │ 4: 9 │ 5: 9 │ 3: 10 │ 4: 10 │ 5: 10 │ 3: 11 │ 4: 11 │ 5: 11 │
    └──────┴──────┴──────┴──────┴──────┴──────┴───────┴───────┴───────┴───────┴───────┴───────┘

In above code sample, we use ``local(3, 4).spatial(2, 3)`` to represent the composition of the two layouts: ``local(3, 4)`` and ``spatial(2, 3)``.
The composition is an operation over two layouts and returns a new layout that combines the two layouts together.
The composition operation is associative, meaning that $(a * b) * c = a * (b * c)$, where $*$ is the composition operation and $a$, $b$, and $c$ are layouts.
The composition operation is not commutative, meaning that $a * b \neq b * a$ in general.

For example, if we compose the ``spatial(2, 3)`` layout with the ``local(3, 4)`` layout, we can get a different layout:

.. doctest::

    >>> print(visualize_layout(spatial(2, 3).local(3, 4)))
    RegisterLayout(shape=[6, 12], mode_shape=[2, 3, 3, 4], spatial_modes=[0, 2], local_modes=[1, 3])
    ┌──────┬──────┬───────┬───────┬──────┬──────┬───────┬───────┬──────┬──────┬───────┬───────┐
    │ 0: 0 │ 0: 1 │ 0: 2  │ 0: 3  │ 1: 0 │ 1: 1 │ 1: 2  │ 1: 3  │ 2: 0 │ 2: 1 │ 2: 2  │ 2: 3  │
    ├──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┤
    │ 0: 4 │ 0: 5 │ 0: 6  │ 0: 7  │ 1: 4 │ 1: 5 │ 1: 6  │ 1: 7  │ 2: 4 │ 2: 5 │ 2: 6  │ 2: 7  │
    ├──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┤
    │ 0: 8 │ 0: 9 │ 0: 10 │ 0: 11 │ 1: 8 │ 1: 9 │ 1: 10 │ 1: 11 │ 2: 8 │ 2: 9 │ 2: 10 │ 2: 11 │
    ├──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┤
    │ 3: 0 │ 3: 1 │ 3: 2  │ 3: 3  │ 4: 0 │ 4: 1 │ 4: 2  │ 4: 3  │ 5: 0 │ 5: 1 │ 5: 2  │ 5: 3  │
    ├──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┤
    │ 3: 4 │ 3: 5 │ 3: 6  │ 3: 7  │ 4: 4 │ 4: 5 │ 4: 6  │ 4: 7  │ 5: 4 │ 5: 5 │ 5: 6  │ 5: 7  │
    ├──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┼──────┼──────┼───────┼───────┤
    │ 3: 8 │ 3: 9 │ 3: 10 │ 3: 11 │ 4: 8 │ 4: 9 │ 4: 10 │ 4: 11 │ 5: 8 │ 5: 9 │ 5: 10 │ 5: 11 │
    └──────┴──────┴───────┴───────┴──────┴──────┴───────┴───────┴──────┴──────┴───────┴───────┘


Represent Tensor Core Layouts
-----------------------------

All layouts shown in the the documentation for PTX MMA instructions are actually register layouts and can be
represented using tilus's layout system.

The layout for operand C of ``mma.sync.aligned.m16n8k8 f16, f16, f16, f16`` (`Figure 77 <https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-c-f16-f32>`_ of PTX manual):

.. doctest::

    >>> layout = repeat(2, 1).spatial(8, 4).repeat(1, 2)
    >>> print(visualize_layout(layout))
    RegisterLayout(shape=[16, 8], mode_shape=[2, 8, 4, 2], spatial_modes=[1, 2], local_modes=[0, 3])
    ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
    │ 0: 0  │ 0: 1  │ 1: 0  │ 1: 1  │ 2: 0  │ 2: 1  │ 3: 0  │ 3: 1  │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 4: 0  │ 4: 1  │ 5: 0  │ 5: 1  │ 6: 0  │ 6: 1  │ 7: 0  │ 7: 1  │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 8: 0  │ 8: 1  │ 9: 0  │ 9: 1  │ 10: 0 │ 10: 1 │ 11: 0 │ 11: 1 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 12: 0 │ 12: 1 │ 13: 0 │ 13: 1 │ 14: 0 │ 14: 1 │ 15: 0 │ 15: 1 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 16: 0 │ 16: 1 │ 17: 0 │ 17: 1 │ 18: 0 │ 18: 1 │ 19: 0 │ 19: 1 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 20: 0 │ 20: 1 │ 21: 0 │ 21: 1 │ 22: 0 │ 22: 1 │ 23: 0 │ 23: 1 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 24: 0 │ 24: 1 │ 25: 0 │ 25: 1 │ 26: 0 │ 26: 1 │ 27: 0 │ 27: 1 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 28: 0 │ 28: 1 │ 29: 0 │ 29: 1 │ 30: 0 │ 30: 1 │ 31: 0 │ 31: 1 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 0: 2  │ 0: 3  │ 1: 2  │ 1: 3  │ 2: 2  │ 2: 3  │ 3: 2  │ 3: 3  │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 4: 2  │ 4: 3  │ 5: 2  │ 5: 3  │ 6: 2  │ 6: 3  │ 7: 2  │ 7: 3  │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 8: 2  │ 8: 3  │ 9: 2  │ 9: 3  │ 10: 2 │ 10: 3 │ 11: 2 │ 11: 3 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 12: 2 │ 12: 3 │ 13: 2 │ 13: 3 │ 14: 2 │ 14: 3 │ 15: 2 │ 15: 3 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 16: 2 │ 16: 3 │ 17: 2 │ 17: 3 │ 18: 2 │ 18: 3 │ 19: 2 │ 19: 3 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 20: 2 │ 20: 3 │ 21: 2 │ 21: 3 │ 22: 2 │ 22: 3 │ 23: 2 │ 23: 3 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 24: 2 │ 24: 3 │ 25: 2 │ 25: 3 │ 26: 2 │ 26: 3 │ 27: 2 │ 27: 3 │
    ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
    │ 28: 2 │ 28: 3 │ 29: 2 │ 29: 3 │ 30: 2 │ 30: 3 │ 31: 2 │ 31: 3 │
    └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘


We only give one example here, but feel free to try other layouts.

Multiple Threads Hold the Same Element
--------------------------------------

In GPU programming, it is common to have some elements replicated in multiple threads. This is necessary if some
operation requires multiple threads to access the same element. Our layout system supports this feature by allowing
multiple threads to hold the same element in the register layout. The following example shows how to create a layout
where each element is held by two threads:


.. doctest::

    >>> from tilus.ir.layout import reduce
    >>> print(visualize_layout(spatial(3, 4)))
    RegisterLayout(shape=[3, 4], mode_shape=[3, 4], spatial_modes=[0, 1], local_modes=[])
    ┌──────┬──────┬───────┬───────┐
    │ 0: 0 │ 1: 0 │ 2: 0  │ 3: 0  │
    ├──────┼──────┼───────┼───────┤
    │ 4: 0 │ 5: 0 │ 6: 0  │ 7: 0  │
    ├──────┼──────┼───────┼───────┤
    │ 8: 0 │ 9: 0 │ 10: 0 │ 11: 0 │
    └──────┴──────┴───────┴───────┘
    >>> print(visualize_layout(reduce(spatial(3, 4), dims=[0])))
    RegisterLayout(shape=[4], mode_shape=[4], spatial_modes=[-3, 0], local_modes=[])
    ┌──────────────┬──────────────┬───────────────┬───────────────┐
    │ [0, 4, 8]: 0 │ [1, 5, 9]: 0 │ [2, 6, 10]: 0 │ [3, 7, 11]: 0 │
    └──────────────┴──────────────┴───────────────┴───────────────┘

If we perform a reduce operation over the dimension 0 on the ``spatial(3, 4)`` layout, we can get a new layout
where each element is held by multiple threads. The new layout has a shape of ``[4]``, and each element is replicated
in three threads. For example, the element at index ``0`` is held by threads ``0``, ``4``, and ``8``.


Unified Layout Representation
-----------------------------

In above examples, besides the grid diagram, we also show the attributes of the layout in the ``RegisterLayout(...)`` object.
Those attributes defined the layout in a unified way. Each register layout has the following four attributes:

- **shape**: the shape of the layout, which must match the shape of the tensor that the layout is applied to.
- **mode_shape**: the size of each `mode`.
- **spatial_modes**: the modes for the parallel workers.
- **local_modes**: the modes for the local storage of each thread.

There is an important concept called **mode** in the layout. We adopt the terminology from Graphene/Cute. Given a
shape of a tensor, we can (optionally) split each dimension into multiple sub-dimensions, and each sub-dimension is called a **mode**.

For example, if we have a tensor with shape ``[3, 4]``, we can split the second dimension into two modes of size 2 and 2 and keep the first dimension as a single mode of size 3.
Then the mode shape of the tensor is ``[3, 2, 2]``.
Take another example, if we have a tensor with shape ``[12, 1, 6]``, we can split the first dimension into two modes of size 3 and 4,
the second dimension into a single mode of size 1, and the third dimension into two modes of size 2 and 3.
Then the mode shape of the tensor is ``[3, 4, 1, 2, 3]``.
Since the modes with size 1 are redundant (we can always insert arbitrary number of 1s in the mode shape), we prune all the modes with size 1, and the mode shape of the tensor is ``[3, 4, 2, 3]``.

Given a shape and a mode shape, we can distribute the modes into two categories: **spatial modes** and **local modes**.
The spatial modes are the modes that are distributed among the parallel workers, while the local modes are the modes that are stored in the local array of each thread.
We use the ``spatial_modes`` and ``local_modes`` attributes to represent the spatial modes and local modes, respectively.

The order of the modes in the ``spatial_modes`` and ``local_modes`` attributes matters.

.. doctest::

    >>> from tilus.ir.layout import column_local, column_spatial
    >>> print(visualize_layout(local(2, 3)))
    RegisterLayout(shape=[2, 3], mode_shape=[2, 3], spatial_modes=[], local_modes=[0, 1])
    ┌──────┬──────┬──────┐
    │ 0: 0 │ 0: 1 │ 0: 2 │
    ├──────┼──────┼──────┤
    │ 0: 3 │ 0: 4 │ 0: 5 │
    └──────┴──────┴──────┘
    >>> print(visualize_layout(column_local(2, 3)))
    RegisterLayout(shape=[2, 3], mode_shape=[2, 3], spatial_modes=[], local_modes=[1, 0])
    ┌──────┬──────┬──────┐
    │ 0: 0 │ 0: 2 │ 0: 4 │
    ├──────┼──────┼──────┤
    │ 0: 1 │ 0: 3 │ 0: 5 │
    └──────┴──────┴──────┘
    >>> print(visualize_layout(spatial(2, 3)))
    RegisterLayout(shape=[2, 3], mode_shape=[2, 3], spatial_modes=[0, 1], local_modes=[])
    ┌──────┬──────┬──────┐
    │ 0: 0 │ 1: 0 │ 2: 0 │
    ├──────┼──────┼──────┤
    │ 3: 0 │ 4: 0 │ 5: 0 │
    └──────┴──────┴──────┘
    >>> print(visualize_layout(column_spatial(2, 3)))
    RegisterLayout(shape=[2, 3], mode_shape=[2, 3], spatial_modes=[1, 0], local_modes=[])
    ┌──────┬──────┬──────┐
    │ 0: 0 │ 2: 0 │ 4: 0 │
    ├──────┼──────┼──────┤
    │ 1: 0 │ 3: 0 │ 5: 0 │
    └──────┴──────┴──────┘

.. hint::
    :class: margin

    Think what happens if we allow one mode to be assigned to both spatial and local modes.

We do not allow one mode to be assigned to both spatial and local modes.

Mapping Process
~~~~~~~~~~~~~~~

I will use a simple example to illustrate how the mapping process works. Given a layout with the following attributes:

- shape: [4, 6]
- mode_shape: [2, 2, 3, 2]
- spatial_modes: [0, 2]
- local_modes: [3, 1]

Given an index (i, j) in the tensor, we hope to find the corresponding (thread_id, local_id) pair(s).
We first need to compute the index for the modes (i.e., the sub-dimensions) in the mode shape:

- mode_index: [i // 2, i % 2, j // 2, j % 2]

Then, we distribute the modes into spatial and local modes according to the layout attributes:

- spatial_index: [mode_index[0], mode_index[2]] ([i // 2, j // 2])
- spatial_shape: [mode_shape[0], mode_shape[2]] ([2, 3])
- local_index: [mode_index[3], mode_index[1]] ([j % 2, i % 2])
- local_shape: [mode_shape[3], mode_shape[1]] ([2, 2])

Next, we compute the thread_id and local_id:

- thread_id = spatial_index[0] * spatial_shape[1] + spatial_index[1]
- local_id = local_index[0] * local_shape[1] + local_index[1]

Thus, we know that the element at index (i, j) is stored in

- thread_id: (i // 2) * 3 + (j // 2)
- local_id: (j % 2) * 2 + (i % 2)


Replicated Spatial Modes
~~~~~~~~~~~~~~~~~~~~~~~~

We might store the same element in multiple threads. There are two kinds of spatial modes:

- **normal spatial modes**: the modes from the tensor sub-dimensions.
- **replicated modes**: the modes that representing a replication.

The reduce(spatial(3, 4), dims=[0]) operation creates a layout with ``spatial_modes=[-3, 0]``, and the ``-3`` means that
we have a replication of 3. See the reduce example above for what "replication" means.


Operations on Layouts
---------------------

We can treat layouts as a special kind of tensors, where each element in the layout is a list of (thread_id, local_id) pairs, representing the
locations with the element. We have a list of operations that can be performed on layouts to transform the layouts. All of them are defined
under the :py:mod:`tilus.ir.layout` module. Here is a list of the operations:

.. currentmodule:: tilus.ir.layout

Layout Creation
~~~~~~~~~~~~~~~

We can directly use :py:func:`tilus.ir.layout.register_layout` to create a register layout by specifying ``shape``, ``mode_shape``, ``spatial_modes``, and ``local_modes``.
We can also create a layout using the following functions.

.. autosummary::

    spatial
    local
    column_spatial
    column_local
    auto_local_spatial

Layout Transformation
~~~~~~~~~~~~~~~~~~~~~

We can transform one layout to another layout using the following functions. These functions does not change the number of threads and the number of elements per thread, but change the tensor shape and how we map the elements to threads/local elements.

.. autosummary::

    squeeze
    unsqueeze
    permute
    reshape
    flatten


Layout Composition
~~~~~~~~~~~~~~~~~~

We can compose two layouts together to create a new layout using the following functions. The two functions are
different in terms of the composition method.

.. autosummary::

    concat
    compose

Other Operations
~~~~~~~~~~~~~~~~

We can also perform some other operations on layouts, such as:

.. autosummary::

    divide
    reduce
