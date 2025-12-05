.. _colang_2_getting_started_hello_world:

=============
Hello World
=============

This section introduces a "Hello World" Colang example.

Flows
-----

A Colang script is a ``.co`` file and is composed of one or more flow definitions. A flow is a sequence of statements describing the desired interaction between the user and the bot.

The entry point for a Colang script is the ``main`` flow. In the example below, the ``main`` flow is waiting for the user to say "hi" and instructs the bot to respond with "Hello World!".

.. code-block:: colang
  :caption: examples/v2_x/tutorial/hello_world_1/main.co
  :linenos:

  import core

  flow main
    user said "hi"
    bot say "Hello World!"

.. note::
  You can find the full example from this guide `here <../../../examples/v2_x/tutorial/hello_world_1>`_.

The achieve this, the ``main`` flow uses two pre-defined flows:

- ``user said``: this flow is triggered when the user said something.
- ``bot say``: this flow instructs the bot to say a specific message.

The two flows are located in the ``core`` module, included in the Colang Standard Library, which is available by default (similarly to the Python Standard Library). The ``import`` statement at the beginning, imports all the flows from the ``core`` module.

.. note::

  For more details, check out the :ref:`the-standard-library`.

Testing
-------

To test the above script, you can use the NeMo Guardrails CLI:

.. code-block:: text

  $ nemoguardrails chat --config=examples/v2_x/tutorial/hello_world_1

  > hi

  Hello World!

  > something else is ignored

  >

.. note::

  The above example does not use the LLM. To get a response from the bot, you have to send the exact "hi" text, otherwise no response is returned. Extending this with LLM support is covered in the next section.


Congratulations, you've just created your first Colang script.

The :ref:`next example<colang_2_getting_started_dialog_rails>` will teach you how to create *dialog rails* by adding additional flows and enabling the LLM integration.
