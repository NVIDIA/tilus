Thread Group
============

In a GPU kernel, all threads in a thread block execute the same code. But in practice, different
threads often need to do different things — one warp loads data while another computes, or a single
thread updates a barrier while the rest wait. **Thread groups** let you partition threads within a
block and assign different work to each partition.

.. code-block:: python

    def __call__(self, ...):
        self.attrs.warps = 4  # 128 threads total

        # All 128 threads execute this
        acc = self.register_tensor(dtype=float32, shape=[64, 64], init=0.0)

        with self.thread_group(thread_begin=0, num_threads=32):
            # Only threads 0–31 (one warp) execute this
            self.tma.global_to_shared(...)

        with self.thread_group(thread_begin=32, num_threads=32):
            # Only threads 32–63 execute this
            self.tcgen05.mma(...)

        # All 128 threads execute this again
        self.sync()


Why Thread Groups?
------------------

Thread groups serve three purposes:

1. **Hardware requirements** — Some instructions require a specific number of threads. TMA operations
   need a warp-aligned group (32 threads). WGMMA needs a full warp group (128 threads). Mbarrier
   arrive/expect-tx-multicast needs at least 16 threads. Semaphore operations need exactly one thread.

2. **Parallel pipelines** — In high-performance kernels, you often want one set of threads loading
   data while another set computes. Thread groups let you express this naturally.

3. **Avoiding redundant work** — Operations like barrier signaling or TMA setup only need one thread.
   Running them on all threads wastes cycles and can cause incorrect behavior (e.g., decrementing
   a barrier count 128 times instead of once).


The ``thread_group`` Context Manager
-------------------------------------

The core API is ``self.thread_group(thread_begin, num_threads)``:

.. code-block:: python

    with self.thread_group(thread_begin=0, num_threads=32):
        # Only threads 0–31 execute instructions here
        ...

- ``thread_begin``: index of the first thread (relative to the **parent** group, not the block).
- ``num_threads``: how many consecutive threads are in this group.

**Constraints:**

- ``thread_begin >= 0``
- ``thread_begin + num_threads <= parent group size``

Thread groups can be **nested** — each level partitions the parent group further:

.. code-block:: python

    with self.thread_group(thread_begin=0, num_threads=64):
        # Threads 0–63

        with self.thread_group(thread_begin=0, num_threads=32):
            # Threads 0–31 (relative to parent = absolute 0–31)
            ...

        with self.thread_group(thread_begin=32, num_threads=32):
            # Threads 32–63 (relative to parent = absolute 32–63)
            ...


Shortcuts
---------

Tilus provides convenience methods for common thread group patterns:

``self.single_thread(thread=-1)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute with exactly one thread. By default (``thread=-1``), the hardware picks any thread
using elect-any semantics, which generates efficient uniform-predicate code. Pass an explicit
index to pin execution to a specific thread.

.. code-block:: python

    with self.single_thread():
        # One thread signals the barrier
        self.mbarrier.arrive_and_expect_tx(barrier, transaction_bytes=tile_bytes)

``self.single_warp(warp=0)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute with one warp (32 threads). Equivalent to ``thread_group(warp * 32, 32)``.

.. code-block:: python

    with self.single_warp():
        # One warp issues the TMA copy
        self.tma.global_to_shared(src=ga, dst=sa, offsets=..., mbarrier=barrier)

``self.warp_group(warp_begin, num_warps)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute with multiple warps. Equivalent to ``thread_group(warp_begin * 32, num_warps * 32)``.
Commonly used with ``num_warps=4`` for WGMMA (which requires 128 threads).

.. code-block:: python

    with self.warp_group(warp_begin=0, num_warps=4):
        # 128 threads (warps 0–3) perform WGMMA
        self.wgmma.fence()
        self.wgmma.mma(sa, sb, acc)
        self.wgmma.commit_group()
        self.wgmma.wait_group(0)


Producer-Consumer Pipelines
----------------------------

The most common use of thread groups is building **producer-consumer pipelines** where one group
loads data and another group computes. Barriers synchronize between them:

.. code-block:: python

    def __call__(self, ...):
        self.attrs.warps = 2  # 64 threads

        barriers = self.mbarrier.alloc(num_stages)

        with self.thread_group(thread_begin=0, num_threads=32):
            # Producer warp: loads tiles via TMA
            for stage in self.range(num_stages):
                with self.single_thread():
                    self.mbarrier.arrive_and_expect_tx(barriers[stage], tile_bytes)
                self.tma.global_to_shared(src=g_a, dst=s_a[stage], ..., mbarrier=barriers[stage])

        with self.thread_group(thread_begin=32, num_threads=32):
            # Consumer warp: waits for data, then computes
            for stage in self.range(num_stages):
                self.mbarrier.wait(barriers[stage], phase=0)
                self.tcgen05.mma(s_a[stage], s_b[stage], acc)
                self.tcgen05.commit(mbarrier=...)

The producer and consumer warps run **in parallel** — the producer starts loading the next tile
while the consumer is still computing on the current one. Barriers prevent the consumer from
reading data that hasn't arrived yet.


Synchronization Scope
---------------------

``self.sync()`` synchronizes all threads in the **current thread group**, not the entire block:

.. code-block:: python

    with self.thread_group(thread_begin=0, num_threads=64):
        work_phase_1()
        self.sync()    # Syncs only threads 0–63
        work_phase_2()

    self.sync()        # Syncs all threads in the block

This means you can synchronize within a warp group without stalling threads in other groups.
