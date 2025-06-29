.. _tilus-script:


tilus.Script
============


.. currentmodule:: tilus


.. class:: Script

    The :class:`Script` class represents a tilus script, which defines a GPU kernel through a sequence of block-level
    instructions.

.. currentmodule:: tilus.Script

Attributes and Variables
------------------------

.. autosummary::
   :toctree: generated

   attrs
   blockIdx
   gridDim


Instructions
------------

Tensor Creation and Free
~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::
   :class: margin

   Please submit an feature request if your kernel requires additional instructions.


.. autosummary::
   :toctree: generated

   global_view
   register_tensor
   shared_tensor
   global_tensor
   free_shared


Load and Store Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   load_global
   store_global
   load_shared
   store_shared


Asynchronous Copy Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated

   copy_async
   copy_async_commit_group
   copy_async_wait_group
   copy_async_wait_all


Linear Algebra Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   dot

Transform Instructions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   cast

Atomic and Semaphore Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated

    lock_semaphore
    release_semaphore

Synchronization Instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   sync
