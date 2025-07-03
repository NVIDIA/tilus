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


Load and Store Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   load_global
   store_global
   load_shared
   store_shared


Asynchronous Copy Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   copy_async
   copy_async_commit_group
   copy_async_wait_group
   copy_async_wait_all


Linear Algebra Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   dot

Elementwise Arithmetic Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   add
   exp
   exp2
   abs
   maximum
   round
   where

Transform Instructions
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

Reduction Instructions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   max
   min
   sum

Atomic and Semaphore Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    lock_semaphore
    release_semaphore

Synchronization Instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   sync


Miscellaneous Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   assume
   annotate_layout
   print_tensor
   printf
