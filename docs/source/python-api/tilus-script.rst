.. _tilus-script:


tilus.Script
============


.. currentmodule:: tilus


.. class:: Script

    The :class:`Script` class represents a tilus script, which defines a GPU kernel through a sequence of block-level
    instructions. See :doc:`../programming-guides/tilus-script` for an overview of the tilus script language.

    .. method:: __init__()

        Initializes the script. All subclass should call this __init__ method.
        In the ``__init__`` method of the subclass, it can be used to perform compilation-time setup, such as defining
        hyper-parameters or pre-computing values that will be used in the kernel code.

    .. method:: __call__(*args, **kwargs)

        Defines the kernel code that will be executed on the GPU. This method should contain the logic of the kernel,
        including memory accesses, computations, and any other operations that need to be performed.

.. currentmodule:: tilus.Script

Attributes and Variables
------------------------

.. autosummary::
   :toctree: generated

   attrs
   blockIdx
   gridDim

Language Constructs
-------------------

.. autosummary::
   :toctree: generated

   assume
   range


Instructions
------------

.. autosummary::
   :toctree: generated

   abs
   add
   annotate_layout
   arrive_barrier
   arrive_remote_barrier
   assign
   cast
   copy_async
   copy_async_bulk_global_to_shared
   copy_async_bulk_global_to_cluster_shared
   copy_async_bulk_shared_to_global
   copy_async_commit_group
   copy_async_wait_all
   copy_async_wait_group
   cluster_sync
   dot
   exp
   exp2
   free_shared
   global_tensor
   global_view
   init_barrier
   load_global
   load_shared
   lock_semaphore
   max
   maximum
   min
   print_tensor
   printf
   register_tensor
   release_semaphore
   repeat
   repeat_interleave
   round
   shared_tensor
   squeeze
   store_global
   store_shared
   sum
   sync
   transpose
   unsqueeze
   view
   wait_barrier
   where


Script Attributes
-----------------

.. currentmodule:: tilus.lang

.. class:: Attributes

.. currentmodule:: tilus.lang.Attributes

.. autosummary::
   :toctree: generated

   blocks
   warps
