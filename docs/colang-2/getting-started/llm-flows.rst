.. _colang_2_getting_started_llm_flows:

=============
LLM Flows
=============

This section explains how to create LLM-driven flows in Colang 2.0.

Using Colang, you can describe complex patterns of interaction. However, as a developer, you will never be able to describe all the potential paths an interaction can take. And this is where an LLM can help, by generating *LLM-driven continuations* at runtime.

The :ref:`colang_2_getting_started_dialog_rails` and the :ref:`colang_2_getting_started_input_rails` examples, show how to use the LLM to generate continuations dynamically. The example below is similar to the dialog rails example, but it instructs the LLM to generate directly the bot response. Note, the quality of the response depends on the configured LLM model and can vary.


.. code-block:: colang
  :linenos:
  :caption: examples/v2_x/tutorial/llm_flows/main.co

  import core
  import llm

  flow main
    """You are an assistant that should talk to the user about cars.
    Politely decline to talk about anything else.

    Last user question is: "{{ question }}"
    Generate the output in the following format:

    bot say "<<the response>>"
    """
    $question = await user said something
    ...

The ``main`` flow above waits for the ``user said something`` to match a user utterance, stores the result in the ``$question`` local variable and then invokes the LLM, through the ``...`` (generation operator) to generate the continuation of the flow.

.. note::

  Context variables can be included in the NLD (Natural Language Description) of a flow (a.k.a., docstrings in Python) using double curly braces (the Jinja2 syntax).

Testing
-------

.. code-block:: text

  $ nemoguardrails chat --config=examples/v2_x/other/llm_flow

  > hi

  Hello! How can I assist you with cars today?

  > what can yo udo?

  I am an assistant that can talk to you about cars. Is there anything specific you would like to know?

This section concludes the Colang 2.0 getting started guide. Check out the :ref:`colang_2_getting_started_recommended_next_steps` for the recommended way to continue learning about Colang 2.0.
