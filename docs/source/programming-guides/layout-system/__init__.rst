Layout System
=============

.. toctree::
    :hidden:

    global-layout
    shared-layout
    register-layout
    layout-inference

The layout system in tilus provides us a flexible way to define how tensor elements are stored in different memory scopes, such as global memory, shared memory, and registers.

There are three kinds of tensors in tilus:

- :doc:`../type-system/global-tensor`
- :doc:`../type-system/shared-tensor`
- :doc:`../type-system/register-tensor`


Each of these tensors can have a corresponding layout that defines how the elements are organized in memory:

- :doc:`global-layout`
- :doc:`shared-layout`
- :doc:`register-layout`


We require the user to explicitly or implicitly define the layout of global tensors, since they are the interface between
the kernel and the host. As for shared and register tensors, the layout can be explicitly defined by the user, or
automatically inferred by the tilus compiler based on the usage of the tensor. We give a brief overview of the layout
inference algorithm adopted by tilus in :doc:`layout-inference`.
