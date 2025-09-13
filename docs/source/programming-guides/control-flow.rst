Control Flow
============

In Tilus Script, we can control the flow of our program using various control flow statements.
These statements allow us to execute different parts of the code based on certain conditions or to repeat certain actions.

Currently, Tilus Script supports the following control flow statements:

- **if** statement: to execute a block of code conditionally.
- **for** statement: to iterate over a sequence of values.
- **while** statement: to repeat a block of code as long as a condition is true.
- **break** statement: to exit a loop prematurely.
- **continue** statement: to skip the current iteration of a loop and continue with the next iteration.

``if`` statement
----------------

.. code-block:: python
    :caption: Syntax

    if <condition>:
        <block>
    elif <condition>:
        <block>
    elif <condition>:
        <block>
    else:
        <block>

The ``if`` statement allows us to execute a block of code if a specified condition is true.
We can also use `elif` to check additional conditions and `else` to provide a default block of code if none of the
conditions are met. The conditions can be any expression that evaluates to a boolean value.

The ``elif`` and ``else`` parts are optional, allowing us to create simple or complex conditional structures.


``for`` statement
-----------------

.. code-block:: python
    :caption: Syntax

    for <variable> in range(<end>):
        <block>

    for <variable> in range(<start>, <end>):
        <block>

    for <variable> in range(<start>, <end>, <step>):
        <block>

    for <variable> in self.range(<end>, unroll=<unroll>):
        <block>

    for <variable> in self.range(<start>, <end>, unroll=<unroll>):
        <block>

    for <variable> in self.range(<start>, <end>, <step>, unroll=<unroll>):
        <block>

The ``for`` statement allows us to iterate over a sequence of values. We can use the built-in `range` function to
generate a sequence of integers. The `range` function can take one, two, or three arguments:

- ``range(<end>)``: generates a sequence from 0 to `<end> - 1`.
- ``range(<start>, <end>)``: generates a sequence from `<start>` to `<end> - 1`.
- ``range(<start>, <end>, <step>)``: generates a sequence from `<start>` to `<end> - 1` with a step of `<step>`.

We can also use the `self.range` method to generate a sequence of integers, which is similar to the built-in `range` function but allows for an optional `unroll` parameter:

- ``unroll`` parameter: specify the unrolling factor for the loop. It can be

  - ``"all"``: unroll the loop completely.
  - ``<n>``: a positive integer representing the unrolling factor, which indicates how many iterations to unroll.


``while`` statement
-------------------

.. code-block:: python
    :caption: Syntax

    while <condition>:
        <block>

The ``while`` statement allows us to repeat a block of code as long as a specified condition is true.
The condition can be any expression that evaluates to a boolean value. The block of code will be executed repeatedly
until the condition becomes false.

``break`` and ``continue`` statements
-------------------------------------

In addition to the control flow statements mentioned above, Tilus Script also supports `break` and `continue` statements:

- **break**: This statement can be used to exit a loop prematurely.
  When encountered, it will immediately terminate the innermost loop and continue execution after the loop.
- **continue**: This statement can be used to skip the current iteration of a loop and continue with the next iteration.
  When encountered, it will immediately jump to the next iteration of the innermost loop.

