Thread Group
============

In Tilus Script, you can partition the threads in a thread block into smaller **thread groups** and define instructions that execute only within specific thread groups. This provides fine-grained control over thread execution and enables efficient parallel programming patterns.

Overview
--------

At the root level of a kernel, there is one thread group that includes all threads in the thread block. Using the ``thread_group`` context manager, you can:

- Partition threads into multiple sub-groups
- Execute instructions on specific thread groups only
- Create nested thread group hierarchies
- Synchronize threads within a group

The ``thread_group`` Context Manager
------------------------------------

**Syntax:**

.. code-block:: python

    with self.thread_group(group_index, *, num_groups=None, group_size=None):
        # Instructions executed by threads in the specified thread group
        ...

**Parameters:**

- ``group_index`` (int): The index of the thread group to create. Must be in range [0, num_groups).
- ``num_groups`` (int, optional): The number of thread groups to partition the current thread group into.
- ``group_size`` (int, optional): The number of threads in each thread group.

**Constraints:**

- Either ``num_groups`` or ``group_size`` must be specified (or both).
- If both are specified, they must satisfy: ``num_groups * group_size == parent_group_size``
- If only one is specified, the other is automatically inferred.

How Thread Partitioning Works
------------------------------

Thread groups partition the current thread group based on the relationship:

.. code-block:: text

    num_groups * group_size = parent_group_size

**Examples:**

- If you have 128 threads and specify ``group_size=32``, you get ``num_groups=4``
- If you have 128 threads and specify ``num_groups=8``, you get ``group_size=16``
- If you specify both ``num_groups=4`` and ``group_size=32``, they must multiply to 128

Basic Usage Examples
--------------------

**Example 1: Simple Thread Group Partitioning**

.. code-block:: python

    class MyScript(tilus.Script):
        def __init__(self):
            super().__init__()

        def __call__(self, ...):
            # Specify 4 warps = 128 threads total
            self.attrs.warps = 4

            # All threads execute this
            data = self.register_tensor(dtype=f32, shape=[16])

            # Only threads in group 0 execute this block (threads 0-31)
            with self.thread_group(0, num_groups=4):
                # Instructions for first quarter of threads
                result = self.load_global(src, offsets=[0, 0])

            # Only threads in group 1 execute this block (threads 32-63)
            with self.thread_group(1, num_groups=4):
                # Instructions for second quarter of threads
                result = self.load_global(src, offsets=[16, 0])

            # All threads execute this again
            self.sync()

**Example 2: Using Group Size Instead of Number of Groups**

.. code-block:: python

    class MyScript(tilus.Script):
        def __init__(self):
            super().__init__()

        def __call__(self, ...):
            # Specify 4 warps = 128 threads total
            self.attrs.warps = 4

            # Only threads in group 0 execute this (32 threads: 0-31)
            with self.thread_group(0, group_size=32):
                # First group of 32 threads processes first chunk
                data = self.load_global(src, offsets=[0, 0])

            # Only threads in group 1 execute this (32 threads: 32-63)
            with self.thread_group(1, group_size=32):
                # Second group of 32 threads processes second chunk
                data = self.load_global(src, offsets=[32, 0])

**Example 3: Nested Thread Groups**

.. code-block:: python

    class MyScript(tilus.Script):
        def __init__(self):
            super().__init__()

        def __call__(self, ...):
            # Specify 4 warps = 128 threads total
            self.attrs.warps = 4

            # First level: split into 4 groups of 32 threads each
            with self.thread_group(0, num_groups=4):
                # Only first group (threads 0-31) enters here

                # Second level: further split into 2 sub-groups of 16 threads each
                with self.thread_group(0, num_groups=2):
                    # Only threads 0-15 execute this
                    fine_grained_work()

                with self.thread_group(1, num_groups=2):
                    # Only threads 16-31 execute this
                    different_fine_grained_work()

                # Back to first level - all threads 0-31 execute this
                self.sync()

Synchronization Within Thread Groups
------------------------------------

The ``self.sync()`` instruction synchronizes all threads in the **current thread group**, not the entire thread block.

.. code-block:: python

    class MyScript(tilus.Script):
        def __init__(self):
            super().__init__()

        def __call__(self, ...):
            # Specify 4 warps = 128 threads total
            self.attrs.warps = 4

            with self.thread_group(0, num_groups=2):
                # Some work by first half of threads (threads 0-63)
                work_part_1()

                # Synchronize only threads in group 0
                self.sync()  # Only waits for threads 0-63

                # Continue with synchronized work
                work_part_2()

            # Synchronize all threads in the thread block
            self.sync()
