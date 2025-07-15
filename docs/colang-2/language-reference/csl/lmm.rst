------------------------------------------------------------------------------------------------------------------------------------------------------------------
LLM Flows (`llm.co <../../../nemoguardrails/colang/v2_x/library/llm.co>`_)
------------------------------------------------------------------------------------------------------------------------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LLM Enabled Bot Actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: bot say something like $text

    Trigger a bot utterance similar to given text

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_bot_say_something_like
        :end-before: # COLANG_END: test_bot_say_something_like
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_bot_say_something_like
        :end-before: # USAGE_END: test_bot_say_something_like
        :dedent:


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LLM Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: polling llm request response $interval=1.0

    Start response polling for all LLM related calls to receive the LLM responses and act on that

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_polling_llm_request_response
        :end-before: # COLANG_END: test_polling_llm_request_response
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_polling_llm_request_response
        :end-before: # USAGE_END: test_polling_llm_request_response
        :dedent:


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Interaction Continuation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Flow that will continue the current interaction for unhandled user actions/intents or undefined flows.


.. py:function:: llm continuation

    Activate all LLM based interaction continuations

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_llm_continuation
        :end-before: # COLANG_END: test_llm_continuation
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_llm_continuation
        :end-before: # USAGE_END: test_llm_continuation
        :dedent:

.. py:function:: generating user intent for unhandled user utterance

    Generate a user intent event (finish flow event) for unhandled user utterance

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_generating_user_intent_for_unhandled_user_utterance
        :end-before: # COLANG_END: test_generating_user_intent_for_unhandled_user_utterance
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_generating_user_intent_for_unhandled_user_utterance
        :end-before: # USAGE_END: test_generating_user_intent_for_unhandled_user_utterance
        :dedent:

.. py:function:: unhandled user intent -> $intent

    Wait for the end of an user intent flow

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_unhandled_user_intent
        :end-before: # COLANG_END: test_unhandled_user_intent
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_unhandled_user_intent
        :end-before: # USAGE_END: test_unhandled_user_intent
        :dedent:

.. py:function:: continuation on unhandled user intent

    Generate and start new flow to continue the interaction for an unhandled user intent

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_continuation_on_unhandled_user_intent
        :end-before: # COLANG_END: test_continuation_on_unhandled_user_intent
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_continuation_on_unhandled_user_intent
        :end-before: # USAGE_END: test_continuation_on_unhandled_user_intent
        :dedent:

.. py:function:: continuation on undefined flow

    Generate and start a new flow to continue the interaction for the start of an undefined flow

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_continuation_on_undefined_flow
        :end-before: # COLANG_END: test_continuation_on_undefined_flow
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_continuation_on_undefined_flow
        :end-before: # USAGE_END: test_continuation_on_undefined_flow
        :dedent:

.. py:function:: llm continue interaction

    Generate and continue with a suitable interaction

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_llm_continue_interaction
        :end-before: # COLANG_END: test_llm_continue_interaction
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_llm_continue_interaction
        :end-before: # USAGE_END: test_llm_continue_interaction
        :dedent:


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
More Advanced Flows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This section describes more advanced flows defined in the ``llm.co`` library. When you get started with Colang you most
likely will not need to directly use these flows. These flows exist to support more advanced use cases.

**Advanced Interaction Continuation**

Flows with more advanced LLM based continuations

.. code-block:: colang

    # Generate a flow that continues the current interaction
    flow llm generate interaction continuation flow -> $flow_name


**Interaction History Logging**

Flows to log interaction history to created required context for LLM prompts.

.. code-block:: colang

    # Activate all automated user and bot intent flows logging based on flow naming
    flow automating intent detection

    # Marking user intent flows using only naming convention
    flow marking user intent flows

    # Generate user intent logging for marked flows that finish by themselves
    flow logging marked user intent flows

    # Marking bot intent flows using only naming convention
    flow marking bot intent flows

    # Generate user intent logging for marked flows that finish by themselves
    flow logging marked bot intent flows

**State Tracking Flows**

These are flows that track bot and user states in global variables.

.. code-block:: colang

    # Track most recent unhandled user intent state in global variable $user_intent_state
    flow tracking unhandled user intent state
