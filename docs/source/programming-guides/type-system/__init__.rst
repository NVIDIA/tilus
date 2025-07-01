Type System
===========

This section provides an overview of the type system of Tilus Script. There are three categories of types in Tilus Script:

1. **Scalar Types**: represent basic data types such as integers, floats, and booleans.
2. **Pointer Types**: represent addresses of data in memory.
3. **Tensor Types**: represent multi-dimensional arrays of data in global, shared, or register memory.

A variable in Tilus Script will have a type in one of these categories. Each variable is owned by the entire thread block,
even if it is only a single scalar value. We have block-level instructions to operate on these variables (See :doc:`../instructions`).

For a complete list of types for each category, refer to the following sections:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   scalar-types
   pointer-types
   register-tensor
   shared-tensor
   global-tensor
   layout-inference
