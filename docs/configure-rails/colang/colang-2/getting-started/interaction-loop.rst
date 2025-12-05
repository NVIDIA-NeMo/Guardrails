.. _colang_2_getting_started_interaction_loop:

================
Interaction Loop
================

This section explains how to create an interaction loop in Colang 2.0.

Usage
-----

In various LLM-based application, there is a need for the LLM to keep interacting with the user in a continuous interaction loop. The example below shows how a simple interaction loop can be implemented using the ``while`` construct and how the bot can be proactive when the user is silent.

.. code-block:: colang
  :linenos:
  :caption: examples/v2_x/tutorial/interaction_loop/main.co
  :emphasize-lines: 3-4, 8, 11, 14

  import core
  import llm
  import avatars
  import timing

  flow main
    activate automating intent detection
    activate generating user intent for unhandled user utterance

    while True
      when unhandled user intent
        $response = ..."Response to what user said."
        bot say $response
      or when user was silent 12.0
        bot inform about service
      or when user expressed greeting
        bot say "Hi there!"
      or when user expressed goodbye
        bot inform "That was fun. Goodbye"

  flow user expressed greeting
    user said "hi"
      or user said "hello"

  flow user expressed goodbye
    user said "goodbye"
      or user said "I am done"
      or user said "I have to go"

  flow bot inform about service
    bot say "You can ask me anything!"
      or bot say "Just ask me something!"


The ``main`` flow above activates the ``generating user intent for unhandled user utterance`` flow from the ``avatars`` module which uses the LLM to generate the canonical form for a user message (a.k.a., the user intent). Also, when the LLM generates an intent that is not handled by the Colang script, the ``unhandled user intent`` flow is triggered (line 11).

Line 14 in the example above shows how to use the pre-defined ``user silent`` event to model time-driven interaction.

This example also uses the ``when`` / ``or when`` syntax, which is a mechanism for branching a flow on multiple paths. When a flow reaches a branching point, it will start monitoring all the branches and continue the interaction as soon as a branch is matched.

Testing
-------

.. code-block:: text

  $ nemoguardrails chat --config=examples/v2_x/tutorial/interaction_loop

  > hi

  Hi there!

  << pause for 12 seconds >>

  You can ask me anything!

  > how are you?

  I am doing well, thank you for asking! How can I assist you today?

The :ref:`next example<colang_2_getting_started_llm_flows>` will show you how to create LLM-driven flows.
