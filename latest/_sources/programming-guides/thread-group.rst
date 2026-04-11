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

    with self.thread_group(thread_begin, num_threads):
        # Instructions executed by threads in the specified thread group
        ...

**Parameters:**

- ``thread_begin`` (int): The index of the first thread in the thread group (relative to the parent thread group).
- ``num_threads`` (int): The number of threads in this thread group.

**Constraints:**

- ``thread_begin`` must be non-negative and within the parent thread group's range.
- ``thread_begin + num_threads`` must not exceed the parent thread group's size.
- ``num_threads`` must divide evenly into the parent thread group's size.

How Thread Partitioning Works
------------------------------

Thread groups partition the current thread group by specifying:

1. **thread_begin**: The starting thread index (relative to the parent group)
2. **num_threads**: How many consecutive threads to include

This allows you to create non-overlapping thread groups that cover different ranges of threads within the parent group.

**Examples:**

- If you have 128 threads and specify ``thread_begin=0, num_threads=32``, threads 0-31 execute
- If you have 128 threads and specify ``thread_begin=32, num_threads=32``, threads 32-63 execute
- If you have 128 threads and specify ``thread_begin=64, num_threads=64``, threads 64-127 execute

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

            # Only threads 0-31 execute this block
            with self.thread_group(thread_begin=0, num_threads=32):
                # Instructions for first 32 threads
                result = self.load_global(src, offsets=[0, 0])

            # Only threads 32-63 execute this block
            with self.thread_group(thread_begin=32, num_threads=32):
                # Instructions for second 32 threads
                result = self.load_global(src, offsets=[16, 0])

            # All threads execute this again
            self.sync()

**Example 2: Using Different Thread Ranges**

.. code-block:: python

    class MyScript(tilus.Script):
        def __init__(self):
            super().__init__()

        def __call__(self, ...):
            # Specify 4 warps = 128 threads total
            self.attrs.warps = 4

            # Only threads 0-31 execute this
            with self.thread_group(thread_begin=0, num_threads=32):
                # First group of 32 threads processes first chunk
                data = self.load_global(src, offsets=[0, 0])

            # Only threads 32-63 execute this
            with self.thread_group(thread_begin=32, num_threads=32):
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

            # First level: first 32 threads (threads 0-31)
            with self.thread_group(thread_begin=0, num_threads=32):
                # Only threads 0-31 enter here

                # Second level: further split into threads 0-15
                with self.thread_group(thread_begin=0, num_threads=16):
                    # Only threads 0-15 execute this
                    fine_grained_work()

                # Second level: threads 16-31
                with self.thread_group(thread_begin=16, num_threads=16):
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

            # First 64 threads
            with self.thread_group(thread_begin=0, num_threads=64):
                # Some work by first 64 threads (threads 0-63)
                work_part_1()

                # Synchronize only threads 0-63
                self.sync()  # Only waits for threads 0-63

                # Continue with synchronized work
                work_part_2()

            # Synchronize all threads in the thread block
            self.sync()
