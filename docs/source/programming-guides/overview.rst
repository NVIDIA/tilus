Overview
========

Tilus is a programming model designed to simplify the development of high-performance applications on modern hardware.
It provides a high-level abstraction for writing parallel programs that can run efficiently on GPUs, while also allowing
for fine-grained control over hardware resources like shared memory and thread mapping. This guide provides an overview
of the Tilus programming model and its key features.


Hello World
-----------

To write a kernel with Tilus Script, we can define a subclass of :py:class:`tilus.Script` and implement the ``__call__`` method.

.. code-block:: python

    import torch
    import tilus

    # define the kernel by subclassing `tilus.Script`
    class MyKernel(tilus.Script):
        def __call__(self):
            self.attrs.blocks = 1   # one thread block
            self.attrs.warps = 1    # one warp per thread block

            self.printf("Hello, World!\n")

    # instantiate the kernel
    kernel = MyKernel()

    # launch the kernel on GPU
    kernel()
    torch.cuda.synchronize()

Output:

.. code-block:: text

    Hello, World!

Dive Deeper
-----------

We also have detailed sections on different aspects of the Tilus programming model. To learn more about Tilus, you can
explore the following sections:

- :doc:`tilus-script`
- :doc:`type-system/__init__`
- :doc:`instructions`
- :doc:`control-flow`
- :doc:`cache`
- :doc:`autotuning`
- :doc:`low-precision-support`
