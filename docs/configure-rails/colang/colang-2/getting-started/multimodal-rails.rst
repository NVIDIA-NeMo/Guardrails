.. _colang_2_getting_started_multimodal_rails:

================
Multimodal Rails
================

This section explains how to create multimodal rails in Colang 2.0.

Definition
----------

**Multimodal rails** are a type of rails that take into account multiple types of input/output modalities (e.g., text, voice, gestures, posture, image).

Usage
-----
The example below shows how you can control the greeting behavior of an interactive avatar.

.. note::

  The :ref:`the-standard-library` includes an `avatars` module with flows for multimodal events and actions to implement interactive avatars use cases.


.. code-block:: colang
  :caption: examples/v2_x/tutorial/multi_modal/main.co
  :linenos:
  :emphasize-lines: 2, 10, 18

  import core
  import avatars

  flow main
    user expressed greeting
    bot express greeting

  flow user expressed greeting
    user expressed verbal greeting
      or user gestured "Greeting gesture"

  flow user expressed verbal greeting
    user said "hi"
      or user said "hello"

  flow bot express greeting
    bot express verbal greeting
      and bot gesture "Smile and wave with one hand."

  flow bot express verbal greeting
    bot say "Hi there!"
      or bot say "Welcome!"
      or bot say "Hello!"

In the flow above, lines 9 and 17 use the pre-defined flows ``user gestured`` and ``bot gesture`` which match user gestures and control bot gestures.

Under the Hood
--------------

Under the hood, the interactive systems that uses the Colang script above would need to generate ``GestureUserActionFinished`` events (which is what the ``user gestured`` flow is waiting for) and know how to handle ``StartGestureBotAction`` events (which is what the ``bot gesture`` flow triggers).

Testing
-------

To test the above logic using the NeMo Guardrails CLI you can manually send an event by starting the message with a ``/``:

.. code-block:: text

  $ nemoguardrails chat --config=examples/v2_x/tutorial/guardrails_1

  > hi

  Welcome!

  Gesture: Smile and wave with one hand.

  > /GestureUserActionFinished(gesture="Greeting gesture")

  Hi there!

  Gesture: Smile and wave with one hand.

The :ref:`next example<colang_2_getting_started_input_rails>` will show you how define input rails.
