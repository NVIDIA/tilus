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
   thread_group


Instructions
------------

.. autosummary::
   :toctree: generated

   abs
   add
   annotate_layout
   assign
   cast
   copy_async
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
   where


Instruction Groups
------------------

.. toctree::
   :maxdepth: 1

   instruction-groups/mbarrier
   instruction-groups/fence


Script Attributes
-----------------

.. currentmodule:: tilus.lang

.. class:: Attributes

   The ``attrs`` object on a :class:`~tilus.Script` instance is used to set kernel launch configuration.

   .. attribute:: blocks
      :type: list[int]

      The grid dimensions of the kernel launch. Set as a list of up to 3 integers, e.g., ``self.attrs.blocks = [grid_x, grid_y]``.

   .. attribute:: warps
      :type: int

      The number of warps per thread block. Must be a compile-time constant.
