5. Split-K
==========

In previous examples each output tile of C is computed by a single thread
block that iterates over the entire K dimension. This works well when M and N
are large enough to saturate the GPU. However, for workloads with small M and
N but large K, there are not enough output tiles to keep all SMs busy.

**Split-K** addresses this by partitioning the K dimension into
``split_k_factor`` segments, assigning each segment to a separate thread block.
The partial results are then aggregated in-place using semaphore-based
synchronization.

New Instructions
~~~~~~~~~~~~~~~~

This example introduces three new tilus instructions:

* :meth:`~tilus.Script.global_tensor` -- allocates a global tensor managed by
  tilus. Here it stores one semaphore per output tile. The
  ``requires_clean=True`` flag guarantees the tensor is zero-initialized before
  each kernel launch.
* :meth:`~tilus.Script.lock_semaphore` -- spins until the semaphore reaches the
  expected ``value``, then proceeds. This ensures blocks aggregate in the
  correct order.
* :meth:`~tilus.Script.release_semaphore` -- sets the semaphore to a new value,
  unblocking the next waiting block.

Aggregation Protocol
~~~~~~~~~~~~~~~~~~~~

Suppose ``split_k_factor=4``, producing blocks 0, 1, 2, 3 for the same output
tile:

1. **Block 0** stores its partial result directly to C (no lock needed).
   It then releases the semaphore with value 1.
2. **Block 1** spins on :meth:`~tilus.Script.lock_semaphore` until the
   semaphore equals 1. It loads the partial C, adds its own contribution,
   stores the sum back, and releases with value 2.
3. **Block 2** and **Block 3** follow the same pattern.
4. **The last block** releases the semaphore with value 0, resetting it for
   ``requires_clean=True``.

Kernel Implementation
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../../examples/matmul/matmul_v5.py
   :language: python
   :lines: 53-187
   :emphasize-lines: 5,33-36,44-45,100,102,114-117,120-121,129,132-133

Key points in the code above:

* **Three-dimensional grid** (lines 85--89) -- the third grid dimension is
  ``split_k_factor``, and ``self.blockIdx.z`` identifies which K-segment a
  block processes.
* **Per-block K range** (lines 93--97) -- each block computes over
  ``[start_offset_k, end_offset_k)``, a contiguous slice of the K dimension
  rounded to multiples of ``block_k``.
* **Pipelined main loop** (lines 109--147) -- identical in structure to V4,
  with the loop bounds narrowed to the block's K-segment.
* **Layout change via shared memory** (lines 154--159) -- after accumulation,
  the result is cast to float16, written to shared memory, and reloaded. This
  changes the register tensor layout to one suitable for the global store and
  subsequent aggregation.
* **Semaphore-guarded aggregation** (lines 166--186):

  - When ``split_k_factor > 1``, a :meth:`~tilus.Script.global_tensor`
    allocates one int32 semaphore per output tile.
  - Block 0 stores directly; blocks 1+ call
    :meth:`~tilus.Script.lock_semaphore`, load the partial result, accumulate,
    and store.
  - Each block calls :meth:`~tilus.Script.release_semaphore` with the next
    expected value (wrapping to 0 for the last block).

Launch the Kernel
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../../examples/matmul/matmul_v5.py
   :language: python
   :lines: 198-228

.. note::

   The full source code for this example can be found at
   :download:`matmul_v5.py <../../../../../examples/matmul/matmul_v5.py>`.
