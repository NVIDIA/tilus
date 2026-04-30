Script.tcgen05
==============

.. currentmodule:: tilus.lang.instructions.tcgen05

.. autoclass:: Tcgen05InstructionGroup
   :no-autosummary:
   :no-members:
   :exclude-members: __init__, __new__

.. figure:: figures/tmem_layout.svg
   :width: 90%
   :align: center

   Tensor Memory layout: 128 lanes x 512 columns, each cell 32 bits.

.. rubric:: Instructions

.. currentmodule:: tilus.lang.instructions.tcgen05.Tcgen05InstructionGroup

.. autosummary::
   :toctree: generated

   alloc
   dealloc
   slice
   view
   relinquish_alloc_permit
   load
   store
   wait_load
   wait_store
   commit
   copy
   mma
