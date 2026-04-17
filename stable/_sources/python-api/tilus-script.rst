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

   blockIdx
   gridDim
   current_num_threads
   current_thread_begin
   current_thread_end

Language Constructs
-------------------

.. autosummary::
   :toctree: generated

   assume
   range
   single_thread
   single_warp
   static_assert
   thread_group
   warp_group


Instructions
------------

.. autosummary::
   :toctree: generated

   abs
   add
   all
   annotate_layout
   any
   assign
   cast
   clip
   copy_async
   copy_async_commit_group
   copy_async_wait_all
   copy_async_wait_group
   dot
   exp
   exp2
   fast_divmod
   flatten
   free_shared
   global_tensor
   global_view
   load_global
   load_shared
   lock_semaphore
   log
   max
   maximum
   min
   print_tensor
   printf
   register_tensor
   release_semaphore
   repeat
   repeat_interleave
   reshape_shared
   round
   rsqrt
   shared_tensor
   sqrt
   square
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
   :hidden:

   instruction-groups/mbarrier
   instruction-groups/fence
   instruction-groups/tma
   instruction-groups/tcgen05
   instruction-groups/clc
   instruction-groups/cluster
   instruction-groups/wgmma

.. list-table::
   :widths: 20 80

   * - :doc:`mbarrier <instruction-groups/mbarrier>`
     - Memory barrier instructions for synchronizing async memory transactions.
   * - :doc:`fence <instruction-groups/fence>`
     - Fence instructions for memory ordering between proxies.
   * - :doc:`tma <instruction-groups/tma>`
     - Tensor Memory Accelerator (TMA) async copy instructions.
   * - :doc:`tcgen05 <instruction-groups/tcgen05>`
     - Tensor Core Generation 05 (Blackwell) instructions.
   * - :doc:`clc <instruction-groups/clc>`
     - Cluster Launch Control instructions.
   * - :doc:`cluster <instruction-groups/cluster>`
     - Block cluster synchronization and shared memory access.
   * - :doc:`wgmma <instruction-groups/wgmma>`
     - Warp Group Matrix Multiply-Accumulate (Hopper) instructions.


Script Attributes
-----------------

.. toctree::
   :hidden:

   attrs

.. list-table::
   :widths: 20 80

   * - :doc:`attrs <attrs>`
     - Kernel launch configuration (blocks, warps, cluster).
