.. _the-standard-library:

========================================
Colang Standard Library (CSL)
========================================

The Colang Standard Library (CSL) provides an abstraction from the underlying event and action layer and offers a semantic interface to design interaction patterns between the bot and the user. Currently, there are the following library files available under ``nemoguardrails/colang/v2_x/library/`` (`Github link <../../../nemoguardrails/colang/v2_x/library>`_):

.. toctree::
   :maxdepth: 1

   csl/core.rst
   csl/timing.rst
   csl/lmm.rst
   csl/avatars.rst
   csl/guardrails.rst
   csl/attention.rst

To use the flows defined in these libraries you have two options:

1) [Recommended] Import the standard library files using the import statement: e.g. ``import llm``
2) Copy the corresponding `*.co` file directly inside your Colang script directory.

Note that the ``import <library>`` statement will import all available flows of the corresponding library.
