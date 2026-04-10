V4: Software Pipelining
=======================

This example demonstrates how to overlap computation and memory operations
using **software pipelining**.

The Problem
~~~~~~~~~~~

Without pipelining, a typical matmul main loop looks like this::

    for i in range(N):
        async_load(i)
        sync

        compute(i)
        sync

Data loading and computation execute sequentially within the thread block.
Although the GPU can schedule other thread blocks on the same SM to hide
latency, matrix multiplication kernels typically consume many registers and
much shared memory, limiting occupancy. Software pipelining addresses this by
overlapping the two phases within a single thread block.

The Pipelined Approach
~~~~~~~~~~~~~~~~~~~~~~

The core idea is to start loading the *next* tile while computing the
*current* one::

    async_load(0)
    for i in range(N):
        if i < N - 1:
            async_load(i + 1)
        compute(i)
        sync

This is generalized to multiple stages: by allocating ``num_stages`` copies of
the shared memory buffers and using
:meth:`~tilus.Script.copy_async_commit_group` /
:meth:`~tilus.Script.copy_async_wait_group` for fine-grained pipeline control,
the kernel keeps several loads in flight simultaneously.

For further reading, see `ALCOP <https://arxiv.org/abs/2210.16691>`_ and
`Hidet <https://arxiv.org/abs/2210.09603>`_.

Kernel Implementation
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../../examples/matmul/matmul_v4.py
   :language: python
   :lines: 51-128
   :emphasize-lines: 4,32-33,40-41,54-56,62,70-71

Key points in the code above:

* **``num_stages`` as a tuned parameter** (line 54) -- the autotune decorator
  explores 3, 4, and 5 stages.
* **Multi-stage shared memory** (lines 82--83) -- the shared tensors have an
  extra leading dimension of size ``num_stages``, so each pipeline stage has
  its own buffer.
* **Prologue** (lines 86--93) -- the first ``num_stages - 1`` tiles are
  preloaded before the main loop begins, using
  :meth:`~tilus.Script.copy_async_commit_group` to group each pair of copies.
* **Main loop** (lines 97--121) -- each iteration computes on
  ``sa[current_stage]`` / ``sb[current_stage]`` while simultaneously issuing
  an async copy for the tile ``num_stages - 1`` iterations ahead.
  :meth:`~tilus.Script.copy_async_wait_group` with ``n=num_stages - 2``
  ensures only the oldest in-flight group must complete before its data is
  consumed.
* **Stage rotation** (lines 118--119) -- ``current_stage`` and
  ``preload_stage`` advance modulo ``num_stages``.

Launch the Kernel
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../../examples/matmul/matmul_v4.py
   :language: python
   :lines: 131-161

.. note::

   The full source code for this example can be found at
   :download:`matmul_v4.py <../../../../../examples/matmul/matmul_v4.py>`.
