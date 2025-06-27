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


Tensor Creation
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   global_view
   register_tensor


Load and Store Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   load_global
   store_global


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
