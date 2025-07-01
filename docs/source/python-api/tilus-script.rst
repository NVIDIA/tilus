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

.. autosummary::
   :toctree: generated

   cast
   copy_async
   copy_async_commit_group
   copy_async_wait_all
   copy_async_wait_group
   dot
   free_shared
   global_tensor
   global_view
   load_global
   load_shared
   lock_semaphore
   register_tensor
   release_semaphore
   shared_tensor
   squeeze
   store_global
   store_shared
   sync
   unsqueeze
