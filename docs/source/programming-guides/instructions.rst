Instructions
============


Supported Instructions
-----------------------

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

Transform Instructions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   cast

Atomic and Semaphore Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    lock_semaphore
    release_semaphore

Synchronization Instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   sync

