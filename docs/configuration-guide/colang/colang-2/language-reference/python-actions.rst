.. _python-actions:

========================================
Python Actions
========================================

.. .. note::
..     Feedbacks & TODOs:

In the section :ref:`working-with-actions` you have seen how actions can be used in the context of UMIM events. Additionally, you can also use the action concept to call custom python functions decorated as actions in the file ``actions.py`` or in any python file in the subfolder ``action``. This can be particularly useful if you need more complex functionality that cannot be done in Colang. Note, that all the python actions will run in the context of the Colang interpreter.

Here is an example of such a Python action definition:

.. code-block:: python

    from nemoguardrails.actions import action

    @action(name="CustomTestAction")
    async def custom_test(value: int):
        # Complicated calculation based on parameter value
        return result

And here is how you can call it from a Colang flow:

.. code-block:: colang

    flow main
        $result = await CustomTestAction(value=5)
        bot say "The result is: {$result}"

Be aware that Python actions are blocking per default. That means if the action implements a long running task (e.g. an REST API request or) you will want to make the Python Action asynchronous. You can do this by adding the parameter ``execute_async=True`` to the function decorator :

.. code-block:: python

    from nemoguardrails.actions import action

    @action(name="CustomAsyncTestAction", execute_async=True)
    async def custom_test(value: int):
        # Something that takes time, e.g. a REST API request
        return value

And here is how you can call it from a Colang flow:

.. code-block:: colang

    flow main
        # Option 1 start the action and let your flow continue until you really need the result from the action
        start CustomTestAction(value=5) as $action_ref
        # Some other statements ...
        match $action_ref.Finished() as $event_ref
        bot say "The result is: {$event_ref.return_value}" # Access the function return value via the event reference

        # Option 2: You can still use async Python actions like you would use any other action (the same as for non async Python actions)
        $result = await CustomTestAction(value=5)
        bot say "The result is: {$result}"

.. note::

    All Python action names need to end with ``Action``.

In addition to all the custom user defined parameters, the parameters listed below are available in a Python action. To make use of these parameters in your Python action implementation add the parameter to your function signature.

.. code-block:: python

    events: list # Recent history of events
    context: dict # Contains all global variables and can be updated via ContextUpdate event
    config: dict # All configurations from the `config.yml` file
    llm_task_manager: LLMTaskManager # The llm task manage object of type LLMTaskManager
    state: State # The state machine state object
