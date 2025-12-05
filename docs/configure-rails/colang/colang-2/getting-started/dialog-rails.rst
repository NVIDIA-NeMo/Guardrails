.. _colang_2_getting_started_dialog_rails:

=============
Dialog Rails
=============

This section explains how to create dialog rails using Colang.

Definition
----------

*Dialog Rails* are a type of rails enforcing the path that the dialog between the user and the bot should take. Typically, they involve three components:

1. The definition of user messages, which includes the canonical forms, e.g., ``user expressed greeting``, and potential utterances.
2. The definition of bot messages, which includes the canonical forms, e.g., ``bot express greeting``, and potential utterances.
3. The definition of flows "connecting" user messages and the bot messages.

.. note::

  The definitions of user and bot messages are themselves flows that use other pre-defined flows, e.g., ``user said`` and ``bot say``.

The example below extends the :ref:`Hello World example<colang_2_getting_started_hello_world>` by creating the ``user expressed greeting`` and ``bot express greeting`` messages.

.. code-block:: colang
  :caption: examples/v2_x/tutorial/hello_world_2/main.co
  :linenos:

  import core

  flow main
    user expressed greeting
    bot express greeting

  flow user expressed greeting
    user said "hi" or user said "hello"

  flow bot express greeting
    bot say "Hello world!"

.. note::

  The recommended practice is to use past tense for matching external actions, like the user saying something, and present for bot actions that must be executed. See :ref:`flow-naming-conventions` for more details.


LLM Integration
---------------

While the example above has more structure, it is still rigid in the sense that it only works with the exact inputs "hi" and "hello".

To enable the use of the LLM to drive the interaction for inputs that are not matched exactly by flows, you have to *activate* the ``llm continuation`` flow, which is part of the ``llm`` module in the :ref:`the-standard-library`.

.. code-block:: colang
  :caption: examples/v2_x/tutorial/hello_world_3/main.co
  :linenos:

  import core
  import llm

  flow main
    activate llm continuation
    activate greeting

  flow greeting
    user expressed greeting
    bot express greeting

  flow user expressed greeting
    user said "hi" or user said "hello"

  flow bot express greeting
    bot say "Hello world!"

*Flow activation* is a core mechanism in Colang 2.0. In the above example, the ``greeting`` dialog rail is also encapsulated as a flow which is activated in the ``main`` flow. If a flow is not activated (or called explicitly by another flow), it will not be used.

.. note::

  When a flow is **activated** it will start to monitor the stream of events and drive the interaction whenever there is a match.


Testing
-------

.. code-block:: text

  $ nemoguardrails chat --config=examples/v2_x/tutorial/hello_world_3

  > hello there!

  Hello world!

  > how are you?

  I am an AI, so I don't have feelings like humans do. But thank you for asking! Is there something specific you would like to know or talk about?

First, you can see how the user utterance "hello there!" is matched to the flow ``user expressed greeting`` based on its similarity to the expected user answers. Secondly, any unexpected user utterance like "how are you?" will trigger the LLM to generate a suitable response. This is all automatically handled and taken care of by the flow ``llm continuation``. To have more explicit control over the interaction loop checkout the :ref:`colang_2_getting_started_interaction_loop` example.

.. hint::

  To get a better understand of what's happening under the hood, you can use the ``--verbose`` flag when launching the NeMo Guardrails CLI to show all the precessed events:

  .. code-block:: text

    $ nemoguardrails chat --config=examples/v2_x/tutorial/hello_world_3 --verbose

The :ref:`next example<colang_2_getting_started_multimodal_rails>` will show you how to describe multimodal rails using Colang.
