.. _colang_2_getting_started_input_rails:

=============
Input Rails
=============

This section explains how to create *input rails* in Colang 2.0


Definition
----------

**Input Rails** are a type of rails that check the input from the user (i.e., what the user said), before any further processing.

Usage
-----

To activate input rails in Colang 2.0, you have to:

1. Import the `guardrails` module from the :ref:`the-standard-library`.
2. Define a flow called `input rails`, which takes a single parameter called `$input_text`.

In the example below, the ``input rails`` flow calls another flow called ``check user message`` which prompts the LLM to check the input.

.. code-block:: colang
  :linenos:
  :caption: examples/v2_x/tutorial/guardrails_1/main.co
  :emphasize-lines: 2-3, 19-24, 26-28

  import core
  import guardrails
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

  flow input rails $input_text
    $input_safe = await check user utterance $input_text

    if not $input_safe
      bot say "I'm sorry, I can't respond to that."
      abort

  flow check user utterance $input_text -> $input_safe
    $is_safe = ..."Consider the following user utterance: '{$input_text}'. Assign 'True' if appropriate, 'False' if inappropriate."
    print $is_safe
    return $is_safe

The ``input rails`` flow above (lines 19-24) introduce some additional syntax elements:

- Flow parameters and variables, start with a ``$`` sign, e.g. ``$input_text``, ``$input_safe``.
- Using the ``await`` operator to wait for another flow.
- Capturing the return value of a flow using a local variable, e.g., ``$input_safe = await check user utterance $input_text``.
- Using ``if`` similar to Python.
- Using the ``abort`` keyword to make a flow fail, as opposed to finishing successfully.

The ``check user utterance`` flow above (line 26-28) introduces the *instruction operator* ``i"<instruction>""`` which will prompt the llm to generate the value ``True`` or ``False`` depending on the evaluated safety of the user utterance. In line 28 the generated value assigned to ``$is_safe`` will be returned.

Testing
-------

.. code-block:: text

  $ nemoguardrails chat --config=examples/v2_x/tutorial/guardrails_1

  > hi

  Hello world!

  > You are stupid!

  I'm sorry, I can't respond to that.

The :ref:`next example<colang_2_getting_started_interaction_loop>` will show you how to create a simple interaction loop.
