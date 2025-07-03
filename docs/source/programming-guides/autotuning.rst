Autotuning
==========

For the same tilus script, we might use different hyperparameters to achieve different performance. The optimal choice
of the hyperparameters depends on the target hardware and the specific input sizes. Both might not known at the time of
kernel development. To address this, tilus provides an autotuning mechanism that automatically finds the best hyperparameters
for a given tilus script on target hardware and input sizes. The core idea is simple: we compile the tilus script with
different configurations of the hyperparameters (we call them `schedules`), and then run the compiled kernel with
actual input data to measure the performance. The best schedule is selected based on the measured performance.

What's typical hyperparameters?
-------------------------------

The hyperparameters can be any parameters that affect the performance of the kernel but we can not determine the best
one at the time of kernel development. The commonly used hyperparameters include:

- **warps**: we typically use 4 or 8 warps per thread block, but the optimal number of warps may vary depending on the
  target hardware and input sizes.
- **tile sizes**: the tile sizes of the tensor computation assigned to each thread block. The optimal tile sizes depend on the
  target hardware and input sizes, and can be different for different dimensions of the tensor.
- **optimization knobs**: some optimizations configs might have different optimal choice. For example, we can
  use split-k optimization or not (see matrix multiplication tutorial). We might also use different number of stages
  for the software pipelining optimization.

Define tuning space
-------------------

If we have a tilus script that has some hyperparameters like

.. code-block:: python

    class MyScript(tilus.Script):
        def __init__(self, group_size, warps, tile_m, tile_n):
           ...

    # define a kernel with given hyperparameters
    kernel = MyScript(group_size=128, warps=8, tile_m=16, tile_n=16)

where ``group_size`` is a parameter that requires the user to specify and it is related to the functionality of the script,
while ``warps``, ``tile_m``, and ``tile_n`` are hyperparameters that we want to tune for performance. We can use the
:py:meth:`~tilus.autotune` function to define the tuning space for the hyperparameters:

.. code-block:: python

    @tilus.autotune('warps' [4, 8, 16])
    @tilus.autotune('tile_m, tile_n', [(16, 16), (16, 32), (32, 16)])
    class MyScript(tilus.Script):
        def __init__(self, group_size, warps, tile_m, tile_n):
            ...

    # define a kernel with group_size=128
    kernel = MyScript(group_size=128)

    # the kernel launch will trigger the autotuning process, and choose the best schedule
    # among the 9 combinations of hyperparameters: (warps, tile_m, tile_n)
    # (4, 16, 16), (4, 16, 32), (4, 32, 16)
    # (8, 16, 16), (8, 16, 32), (8, 32, 16)
    # (16, 16, 16), (16, 16, 32), (16, 32, 16)
    kernel(...)

We can use the :py:func:`~tilus.autotune` decorator to specify the hyperparameters we want to tune. We can use this
decorator many times to specify different hyperparameters. In one call to the decorator, we can specify one or more
hyperparameters to tune, and the values can be a list of values or a list of tuples for multiple hyperparameters.
The final tuning space is the Cartesian product of all the values specified in the decorator calls.
We can not annotate the same hyperparameter multiple times.

When we launch the kernel, tilus will automatically compile the kernel with all the combinations of the hyperparameters
The kernels will be compiled in parallel when we first call the kernel with a specific input size triggering the JIT
compilation (:doc:`tilus-script`). We can use :py:func:`tilus.option.parallel_workers` to control the number of
parallel workers to compile the kernels.

