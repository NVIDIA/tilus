3. Async Copy
=============

On NVIDIA Ampere and newer architectures, hardware support for asynchronous
copy from global memory to shared memory was introduced. The key advantage is
that data moves directly from global to shared memory **without** using
registers as an intermediate buffer, freeing register resources and reducing
latency.

Tilus exposes this feature through two block-level instructions:

* :meth:`~tilus.Script.copy_async` -- issues an asynchronous copy from a global
  tensor to a shared tensor. The call returns immediately; the data transfer
  happens in the background.
* :meth:`~tilus.Script.copy_async_wait_all` -- blocks until **all** previously
  issued asynchronous copies have completed, guaranteeing that the data is
  available in shared memory.

Because :meth:`~tilus.Script.copy_async_wait_all` does not synchronize threads
within the block, a subsequent :meth:`~tilus.Script.sync` call is still
necessary before reading the shared memory data.

Kernel Implementation
~~~~~~~~~~~~~~~~~~~~~

The kernel below is structurally similar to V2 but replaces the
``load_global`` / ``store_shared`` pair with :meth:`~tilus.Script.copy_async`,
eliminating the register intermediate buffer:

.. literalinclude:: ../../../../../examples/matmul/matmul_v3.py
   :language: python
   :lines: 42-105
   :emphasize-lines: 45-50

Lines 86--90 (within the loop) are the core change compared to V2:

1. Two :meth:`~tilus.Script.copy_async` calls issue background copies for
   tiles of A and B.
2. :meth:`~tilus.Script.copy_async_wait_all` ensures both copies have landed
   in shared memory.
3. :meth:`~tilus.Script.sync` synchronizes all threads so every thread sees
   the updated shared data before the computation begins.

Launch the Kernel
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../../examples/matmul/matmul_v3.py
   :language: python
   :lines: 108-137

.. note::

   The full source code for this example can be found at
   :download:`matmul_v3.py <../../../../../examples/matmul/matmul_v3.py>`.
