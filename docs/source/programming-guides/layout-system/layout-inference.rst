Automatic Layout Inference
==========================

Tilus supports automatic layout inference for shared and register tensors. The users could explicitly specify layouts
for some of the tensors, and tilus will try to infer the layouts for the rest of the tensors based on their usage in the program.
It's possible to specify the layout of all tensors explicitly, or specify none of them and let tilus infer all layouts.

Layout Inference Rules
----------------------

We have a pre-defined set of rules to infer the layout of shared and register tensors. Each rule is designed to infer
the layout of operands and/or results given the existing layouts of other operands/results. It's possible that only
one operand is inferred while other operands still have unknown layouts after the inference. We can have multiple
inference rules for a single instruction.

All inference rules are ordered by their priority.

Layout Inference Order
----------------------

We start to apply the rules for the instructions with some layouts already specified. Afterwards, we apply the rules
to the instructions with none layouts specified. Besides, each rule has a priority, and we apply the rules in the order of
their priority after the first round of ranking.


Layout Inference Algorithm
--------------------------

We repeat applying the inference rules to the instructions with missing layouts, until no more layouts can be inferred.
Once we successfully infer a layout, we will immediately repeat the inference process from the beginning.
Once no more layouts can be inferred, we check if there are any instructions with missing layouts. If so, we will raise an error
indicating that the layout inference failed. If all layouts are successfully inferred, we will proceed to the layout
validation phase. Each instruction has a pre-defined validation rule that checks if the inferred layout is valid.
If we inferred all layouts successfully and all of them are valid, we will proceed to subsequent steps of compilation.

If you are interested in the details, feel free to check the source code of the layout inference algorithm in
:py:mod:`tilus.ir.layout.inference`.
