Instructions
============

Tilus provides a set of instructions for working with tensors maintained by the thread block. These instructions allow
you to create, copy, and compute tensors in global, shared, and register memory.

Tensor Creation and Free
~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::
   :class: margin

   Please submit a feature request if your kernel requires additional instructions.

.. currentmodule:: tilus.Script

.. autosummary::

   global_view
   register_tensor
   shared_tensor
   global_tensor
   free_shared


Load and Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   load_global
   store_global
   load_shared
   store_shared


Asynchronous Copy (SM80+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   copy_async
   copy_async_commit_group
   copy_async_wait_group
   copy_async_wait_all


Bulk Asynchronous Copy (SM90+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   copy_async_bulk_global_to_shared
   copy_async_bulk_global_to_cluster_shared
   copy_async_bulk_shared_to_global


Linear Algebra
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   dot

Elementwise Arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   add
   exp
   exp2
   abs
   maximum
   round
   where

Transform
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   assign
   cast
   squeeze
   unsqueeze
   repeat
   repeat_interleave
   view
   transpose

Reduction
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   max
   min
   sum


Barrier
~~~~~~~

.. autosummary::

   init_barrier
   arrive_barrier
   arrive_remote_barrier
   wait_barrier

Atomic and Semaphore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    lock_semaphore
    release_semaphore

Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   sync
   cluster_sync


Miscellaneous
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   assume
   annotate_layout
   print_tensor
   printf
