---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Timing Flows (`timing.co <../../../nemoguardrails/colang/v2_x/library/timing.co>`_)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Flows related to timing and reacting to periods of silence.

.. py:function:: wait $time_s $timer_id="wait_timer_{uid()}"

    Wait the specified number of seconds before continuing

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_wait_time
        :end-before: # COLANG_END: test_wait_time
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_wait_time
        :end-before: # USAGE_END: test_wait_time
        :dedent:


.. py:function:: repeating timer $timer_id $interval_s

    Start a repeating timer

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_repeating_timer
        :end-before: # COLANG_END: test_repeating_timer
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_repeating_timer
        :end-before: # USAGE_END: test_repeating_timer
        :dedent:

.. py:function:: user was silent $time_s

    Wait for when the user was silent for $time_s seconds

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_user_was_silent
        :end-before: # COLANG_END: test_user_was_silent
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_user_was_silent
        :end-before: # USAGE_END: test_user_was_silent
        :dedent:

.. py:function:: user didnt respond $time_s

    Wait for when the user was silent for $time_s seconds while the bot was silent

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_user_didnt_respond
        :end-before: # COLANG_END: test_user_didnt_respond
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_user_didnt_respond
        :end-before: # USAGE_END: test_user_didnt_respond
        :dedent:

.. py:function:: bot was silent $time_s

    Wait for the bot to be silent (no utterance) for given time

    Example:

    .. literalinclude:: ../../examples/test_csl.py
        :language: colang
        :start-after: # COLANG_START: test_bot_was_silent
        :end-before: # COLANG_END: test_bot_was_silent
        :dedent:


    .. literalinclude:: ../../examples/test_csl.py
        :language: text
        :start-after: # USAGE_START: test_bot_was_silent
        :end-before: # USAGE_END: test_bot_was_silent
        :dedent:
