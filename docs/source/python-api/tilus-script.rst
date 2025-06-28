.. _tilus-script:


tilus.Script
============


.. currentmodule:: tilus


.. class:: Script

    The :class:`Script` class represents a tilus script, which defines a GPU kernel through a sequence of block-level
    instructions.

.. currentmodule:: tilus.Script

Script Attributes and Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   attrs
   blockIdx
   gridDim

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
   free_shared


Load and Store Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   load_global
   store_global
   load_shared
   store_shared


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

Synchronization Instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   sync
