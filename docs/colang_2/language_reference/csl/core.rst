------------------------------------------------------------------------------------------------------------------------------------------------------------------
Fundamental Core Flows (`core.co <../../../nemoguardrails/colang/v2_x/library/core.co>`_)
------------------------------------------------------------------------------------------------------------------------------------------------------------------

The core library that contains all relevant flows related to user and bot utterance events and actions.

^^^^^^^^^^^^^^^^
User Event Flows
^^^^^^^^^^^^^^^^
.. py:function:: user said $text -> $transcript

    Wait for a user to have said the provided text using an exact match.

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_user_said
        :end-before: # COLANG_END: test_user_said
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_user_said
        :end-before: # USAGE_END: test_user_said
        :dedent:


.. py:function:: user said something -> $transcript

    Wait for a user to have said something matching any transcript.

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_user_said_something
        :end-before: # COLANG_END: test_user_said_something
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_user_said_something
        :end-before: # USAGE_END: test_user_said_something
        :dedent:


.. py:function:: user saying $text -> $transcript

    Wait for a user to say the given text while talking (this matches the partial transcript of the user
    utterance even if the utterance is not finished yet).

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_user_saying
        :end-before: # COLANG_END: test_user_saying
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_user_saying
        :end-before: # USAGE_END: test_user_saying
        :dedent:


.. py:function:: user saying something -> $transcript

    Wait for any ongoing user utterance (partial transcripts).

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_user_saying_something
        :end-before: # COLANG_END: test_user_saying_something
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_user_saying_something
        :end-before: # USAGE_END: test_user_saying_something
        :dedent:

.. py:function:: user started saying something

    Wait for start of user utterance

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_user_started_saying_something
        :end-before: # COLANG_END: test_user_started_saying_something
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_user_started_saying_something
        :end-before: # USAGE_END: test_user_started_saying_something
        :dedent:

.. py:function:: user said something unexpected -> $transcript

    Wait for a user to have said something unexpected (no active match statement for the user utterance that
    matches the incoming event). This is a rather technical flow. If you are looking for a way to react to a
    wide variety of user messages check out the flows in ``llm.co``.

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_user_said_something_unexpected
        :end-before: # COLANG_END: test_user_said_something_unexpected
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_user_said_something_unexpected
        :end-before: # USAGE_END: test_user_said_something_unexpected
        :dedent:


^^^^^^^^^^^^^^^^
Bot Action Flows
^^^^^^^^^^^^^^^^

.. py:function:: bot say $text

    Execute a bot utterance with the provided text and wait until the utterance is completed (e.g. for a voice bot this
    flow will finish once the bot audio has finished).

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_bot_say
        :end-before: # COLANG_END: test_bot_say
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_bot_say
        :end-before: # USAGE_END: test_bot_say
        :dedent:



    **Semantic variants**

    For more expressive interaction histories and more advance use cases the ``core.co`` library provides several
    semantic wrappers for ``bot say``. You can use them anywhere instead of a ``bot say`` to annotated the
    purpose of the bot utterance.


    .. code-block:: colang

        # Trigger the bot to inform about something
        flow bot inform $text

        # Trigger the bot to ask something
        flow bot ask $text

        # Trigger the bot to express something
        flow bot express $text

        # Trigger the bot to respond with given text
        flow bot respond $text

        # Trigger the bot to clarify something
        flow bot clarify $text

        # Trigger the bot to suggest something
        flow bot suggest $text


^^^^^^^^^^^^^^^^
Bot Event Flows
^^^^^^^^^^^^^^^^

.. py:function:: bot started saying $text

    Wait for the bot starting with the given utterance

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_bot_started_saying_example
        :end-before: # COLANG_END: test_bot_started_saying_example
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_bot_started_saying_example
        :end-before: # USAGE_END: test_bot_started_saying_example
        :dedent:

.. py:function:: bot started saying something

    Wait for the bot starting with any utterance

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_bot_started_saying_something
        :end-before: # COLANG_END: test_bot_started_saying_something
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_bot_started_saying_something
        :end-before: # USAGE_END: test_bot_started_saying_something
        :dedent:

.. py:function:: bot said $text

    Wait for the bot to finish saying given utterance

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_bot_said
        :end-before: # COLANG_END: test_bot_said
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_bot_said
        :end-before: # USAGE_END: test_bot_said
        :dedent:


.. py:function:: bot said something -> $text

    Wait for the bot to finish with any utterance

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_bot_started_saying_something
        :end-before: # COLANG_END: test_bot_started_saying_something
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_bot_started_saying_something
        :end-before: # USAGE_END: test_bot_started_saying_something
        :dedent:

    **Semantic variants**

    You may react to specific semantic wrappers for ``bot say`` that are defined in the ``core.co`` library


    .. code-block:: colang

        # Wait for the bot to finish informing about something
        flow bot informed something -> $text

        # Wait for the bot to finish asking about something
        flow bot asked something -> $text

        # Wait for the bot to finish expressing something
        flow bot expressed something -> $text

        # Wait for the bot to finish responding something
        flow bot responded something -> $text

        # Wait for the bot to finish clarifying something
        flow bot clarified something -> $text

        # Wait for the bot to finish suggesting something
        flow bot suggested something -> $text

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: wait indefinitely

    Helper flow to wait indefinitely. This is often used at the end of the ``main`` flow to make sure the interaction
    is not restarted.

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_wait_indefinitely
        :end-before: # COLANG_END: test_wait_indefinitely
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_wait_indefinitely
        :end-before: # USAGE_END: test_wait_indefinitely
        :dedent:


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
State Tracking Flows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These are flows that track bot and user states in global variables.


.. py:function:: tracking bot talking state

    Track bot talking state in global variable ``$bot_talking_state``, ``$last_bot_script``.

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_tracking_bot_talking_state
        :end-before: # COLANG_END: test_tracking_bot_talking_state
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_tracking_bot_talking_state
        :end-before: # USAGE_END: test_tracking_bot_talking_state
        :dedent:

.. py:function:: tracking user talking state

    Track user utterance state in global variables: ``$user_talking_state``, ``$last_user_transcript``.

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_tracking_user_talking_state
        :end-before: # COLANG_END: test_tracking_user_talking_state
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_tracking_user_talking_state
        :end-before: # USAGE_END: test_tracking_user_talking_state
        :dedent:


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Development Helper Flows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: notification of colang errors

    A flow to notify about any runtime Colang errors

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_notification_of_colang_errors
        :end-before: # COLANG_END: test_notification_of_colang_errors
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_notification_of_colang_errors
        :end-before: # USAGE_END: test_notification_of_colang_errors
        :dedent:

.. py:function:: notification of undefined flow start

    A flow to notify about the start of an undefined flow

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_notification_of_undefined_flow_start
        :end-before: # COLANG_END: test_notification_of_undefined_flow_start
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_notification_of_undefined_flow_start
        :end-before: # USAGE_END: test_notification_of_undefined_flow_start
        :dedent:

.. py:function:: notification of unexpected user utterance

    A flow to notify about an unhandled user utterance

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_notification_of_unexpected_user_utterance
        :end-before: # COLANG_END: test_notification_of_unexpected_user_utterance
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_notification_of_unexpected_user_utterance
        :end-before: # USAGE_END: test_notification_of_unexpected_user_utterance
        :dedent:
