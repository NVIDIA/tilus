V1: Use Shared Memory
=====================

In the :doc:`previous tutorial <matmul_v0>`, every thread block loaded its tiles of
**A** and **B** directly from global memory into registers. Global memory has high
latency, so a natural next step is to stage the data through **shared memory** --
a small, fast on-chip scratchpad that is visible to all threads in the same block.

Why shared memory helps
-----------------------

Shared memory is much faster than global memory (on the order of 100x lower
latency). By first copying a tile from global memory into shared memory, all
threads in the block can then read from the shared copy at high bandwidth. This
is especially beneficial when the same data is read multiple times by different
threads, as is the case in matrix multiplication where every element of a tile
participates in multiple multiply-accumulate operations.

The data flow for each iteration of the inner loop becomes:

1. **Load** tiles from global memory into register tensors.
2. **Store** those register tensors into shared memory (``store_shared``).
3. **Synchronize** all threads so the shared data is fully written.
4. **Load** from shared memory back into registers (``load_shared``).
5. **Compute** the dot product.
6. **Synchronize** again before the next iteration overwrites shared memory.

Kernel implementation
---------------------

.. literalinclude:: ../../../../../examples/matmul/matmul_v1.py
   :language: python
   :lines: 17-91
   :caption: MatmulV1 -- matrix multiplication with shared memory

The kernel follows the same overall structure as ``MatmulV0``, with two key
additions: shared memory tiles and explicit synchronization.

Shared memory tiles
^^^^^^^^^^^^^^^^^^^

At the top of the ``__call__`` method we allocate two shared tensors, ``sa`` and
``sb``, to hold the current tiles of A and B respectively:

.. code-block:: python

   sa = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
   sb = self.shared_tensor(dtype=float16, shape=[self.block_k, self.block_n])

Inside the loop, data flows through shared memory before reaching the
accumulator:

.. code-block:: python

   # global -> registers -> shared memory
   lda = self.load_global(ga, offsets=[offset_m, offset_k], shape=[self.block_m, self.block_k])
   self.store_shared(sa, lda)

   # ... same for B ...

   self.sync()  # ensure all stores to shared memory are visible

   # shared memory -> registers
   a = self.load_shared(sa)
   b = self.load_shared(sb)

   acc = self.dot(a, b, acc)
   self.sync()  # ensure dot is done before next iteration overwrites shared

At the end, both shared tensors are freed so their memory can be reused:

.. code-block:: python

   self.free_shared(sa)
   self.free_shared(sb)

Why two synchronizations?
^^^^^^^^^^^^^^^^^^^^^^^^^

Both ``self.sync()`` calls are necessary because ``sa`` and ``sb`` are reused
across loop iterations:

- The **first sync** (after ``store_shared``) guarantees that every thread has
  finished writing its portion of the tile into shared memory before any thread
  tries to read from it via ``load_shared``.
- The **second sync** (after ``dot``) guarantees that the dot product has
  finished reading from shared memory before the next iteration overwrites it
  with new data.

Omitting either synchronization leads to a data race.

New instructions
----------------

This example introduces five new :class:`~tilus.Script` methods compared to
``MatmulV0``:

.. currentmodule:: tilus.Script

- :meth:`shared_tensor` -- allocate a shared-memory tensor with a given dtype
  and shape.
- :meth:`store_shared` -- copy a register tensor into a shared tensor.
- :meth:`load_shared` -- copy a shared tensor into a new register tensor.
- :meth:`free_shared` -- release the shared memory so it can be reused. Every
  allocation must be freed before the kernel ends.
- :meth:`sync` -- synchronize all threads in the thread block (equivalent to
  ``__syncthreads()`` in CUDA C).

Launching the kernel
--------------------

The launch code is identical in structure to the previous version -- create an
instance, prepare tensors, and call the script:

.. literalinclude:: ../../../../../examples/matmul/matmul_v1.py
   :language: python
   :lines: 120-151
   :caption: Benchmarking MatmulV1

Full source
-----------

The complete example is available at
:download:`examples/matmul/matmul_v1.py <../../../../../examples/matmul/matmul_v1.py>`.
