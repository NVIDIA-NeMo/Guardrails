.. _defining-flows:

========================================
Defining Flows
========================================

.. .. note::
..     Feedbacks & TODOs:

..     .. - SK: maybe we could have a section that first introduces concurrent pattern matching without any conflicts? And then only later introduce conflict handling? I think parallel flows are interesting by itself.
..     .. - SK: maybe first introduce flow groups and then talk about starting and awaiting them?
..     .. - SK: I think it would be nice if we had a definition (like in other programming languages) of what a statement (line in code) can be. Proposal: A flow consists of n statements where any statement is either a match statement, send statement, a control flow statement or expression statement. (we can optimize the names, but the important thing would be that we stick to the same basic naming convention)

----------------------------------------
Introduction
----------------------------------------

So far you have seen only one flow, the main flow. But in Colang we can define many different flows, like functions in other programming languages. A flow defines a specific interaction pattern made of a sequence of statements. The name of a flow consists of lowercase letters, numbers, underline and whitespace characters. Additionally, a flow definition can include input and output parameters (or short: in and out parameters) with optional default values.

.. important::
    Flow syntax definition:

    .. code-block::

        flow <name of flow>[ $<in_param_name>[=<default_value>]...] [-> <out_param_name>[=<default_value>][, <out_param_name>[=<default_value>]]...]
            ["""<flow summary>"""]
            <interaction pattern>

    Examples:

    .. code-block:: colang

        flow bot say $text $intensity=1.0
            """Bot says given text."""
            # ...

        flow user said $text
            """User said given text."""
            # ...

        flow user said something -> $transcript
            """User said something."""
            # ...

    The choice of allowing whitespace characters in flow names comes with some limitations:

    * The keywords ``and``, ``or`` and ``as`` cannot be used in flow names and would need to be escaped with a leading underline character (e.g., ``this _and that``). But often, rather than using e.g. the word 'and', you can use the word 'then' to combine to actions, e.g ``bot greet then smile`` to describe the sequential dependency. Or alternatively write it as ``bot greet smiling`` if it happens concurrently.
    * As shown in chapter :ref:`Working with Variables & Expressions <working-with-variables-and-expressions>` variables will always start with a ``$`` character.

Like an action, a flow can be started and waited for to finish using the keywords ``start``, ``await`` and ``match``:

.. code-block:: colang
    :caption: flows/call_a_flow/main.co

    flow main
        # Start and wait for a flow in two steps using a flow reference
        start bot express greeting as $flow_ref
        match $flow_ref.Finished()

        # Start and wait for a flow to finish
        await bot express greeting

        # Or without the optional await keyword
        bot express greeting

        match RestartEvent()

    flow bot express greeting
        await UtteranceBotAction(script="Hi")

Note, that starting a flow will immediately process and trigger all initial statements of the flow, up to the first statement that waits for an event:

.. code-block:: colang
    :caption: flows/start_flow/main.co

    flow main
        start bot handle user welcoming
        match RestartEvent() # <- This statement is only processed once the previous flow has started

    flow bot handle user welcoming
        start UtteranceBotAction(script="Hi")
        start GestureBotAction(gesture="Wave") as $action_ref
        match $action_ref.Finished() # <- At this point the flow is considered to have started
        match UtteranceUserAction().Finished()
        start UtteranceBotAction(script="How are you?")

.. important::
    Starting a flow will immediately process and trigger all initial statements of the flow, up to the first statement that waits for an event.


------------
Flow events
------------

Similar to actions, flows themselves can generate different events that relate to a flow's status or lifetime. These flow events have priority over other events (see :ref:`Internal Events<internal-events-defining-flows>`):

.. code-block:: colang

    FlowStarted(flow_id: str, flow_instance_uid: str, source_flow_instance_uid: str) # When a flow has started
    FlowFinished(flow_id: str, flow_instance_uid: str, source_flow_instance_uid: str) # When the interaction pattern of a flow has successfully finished
    FlowFailed(flow_id: str, flow_instance_uid: str, source_flow_instance_uid: str) # When the interaction pattern of a flow has failed

The events can also be accessed like an object method of the flow:

.. code-block:: colang

    Started(flow_id: str, flow_instance_uid: str, source_flow_instance_uid: str) # When a flow has started
    Finished(flow_id: str, flow_instance_uid: str, source_flow_instance_uid: str) # When the interaction pattern of a flow has successfully finished
    Failed(flow_id: str, flow_instance_uid: str, source_flow_instance_uid: str) # When the interaction pattern of a flow has failed


These events can be matched via a flow reference or the flow name itself:

.. code-block:: colang

    # Match to flow event with flow reference
    match $flow_ref.Finished()

    # Match to flow event based on flow name
    match (bot express greeting).Finished()

The main difference is, that matching to a flow event with a reference will be specific to the actual referenced flow instance, whereas matching via the flow name will succeed for any flow instance of that flow.

Here is an example of a flow with parameters:

.. code-block:: colang
    :caption: flows/flow_parameters/main.co

    flow main
        # Say 'Hi' with the default volume of 1.0
        bot say "Hi"

    flow bot say $text $volume=1.0
        await UtteranceBotAction(script=$text, intensity=$volume)

Note how we can abstract and simplify the action handling with flows using a simpler name. This allows us to wrap most actions and events into flows that are made readily available through the :ref:`the-standard-library`. See also section :ref:`Internal Events <internal-events-defining-flows>` where the underlying flow event mechanics are explained in more detail.

----------------------------------------
Flow and Action Lifetime
----------------------------------------

Starting a flow within another flow will implicitly create a hierarchy of flows where the '`main`' flow is the root flow of all of them. Like actions, the lifetime of a flow is limited by the lifetime of its parent flow. In other words, a flow will be stopped as soon as the flow that started it has finished or was stopped itself:

.. code-block:: colang

    flow main
        match UserReadyEvent()
        bot express greeting

    flow bot express greeting
        start bot say "Hi!" as $flow_ref
        start bot gesture "wave with one hand"
        match $flow_ref.Finished()

    flow bot say $text
        await UtteranceBotAction(script=$text)

    flow bot gesture $gesture
        await GestureBotAction(gesture=$gesture)

We see that the '`main`' flow starts and waits for the flow '`bot express greeting`', which starts the two flows '`bot say`' and '`bot gesture`'. But the flow '`bot express greeting`' will only wait for '`bot say`' to finish and automatically stop '`bot gesture`' if it is still active. Now with our simple chat CLI this is a bit difficult to simulate, since both the `UtteranceBotAction` and `GestureBotAction` have no duration and will finish immediately. In an interactive system, where the bot actually speaks and uses e.g. animations for the gesture action this would take some time to finish. But we can also simulate this effect by using the `TimerBotAction` that will just introduce a specified delay:

.. code-block:: colang
    :caption: flows/flow_hierarchy/main.co

    flow main
        match UserReadyEvent()
        bot express greeting

    flow bot express greeting
        start bot say "Hi!" as $flow_ref
        start bot gesture "wave with one hand"
        match $flow_ref.Finished()

    flow bot say $text
        await TimerBotAction(timer_name="utterance_timer", duration=2.0)
        await UtteranceBotAction(script=$text)

    flow bot gesture $gesture
        await TimerBotAction(timer_name="gesture_timer", duration=5.0)
        await GestureBotAction(gesture=$gesture)

Running this now shows the desired behavior:

.. code-block:: text

    > /UserReadyEvent

    Hi

If you want you can also change the duration of the gesture timer to be smaller than the utterance timer to see that the gesture can finish successfully:

.. code-block:: text

    /UserReadyEvent

    Gesture: wave with on hand

    Hi!

The end of a flow (finished or failed) will also stop all remaining active actions. Like flows, the lifetime of actions that were started within a flow are limited by the lifetime of the parent flow. This helps to limit unintended side effects and makes the interaction design more robust.

.. important::
    The lifetime of any started flow or action is limited by the lifetime of the parent flow.

.. _defining-flows-concurrent-pattern-matching:

----------------------------------------
Concurrent Pattern Matching
----------------------------------------

Flows are more than just functions as known from other programming languages. Flows are interaction patterns that can match and progress concurrently:

.. code-block:: colang
    :caption: flows/concurrent_flows_basics/main.co

    flow main
        start pattern a as $flow_ref_a
        start pattern b as $flow_ref_b
        match $flow_ref_a.Finished() and $flow_ref_b.Finished()
        await UtteranceBotAction(script="End")
        match RestartEvent()

    flow pattern a
        match UtteranceUserAction.Finished(final_transcript="Bye")
        await UtteranceBotAction(script="Goodbye") as $action_ref

    flow pattern b
        match UtteranceUserAction.Finished(final_transcript="Hi")
        await UtteranceBotAction(script="Hello")
        match UtteranceUserAction.Finished(final_transcript="Bye")
        await UtteranceBotAction(script="Goodbye") as $action_ref

.. code-block:: text

    > Hi

    Hello

    > Bye

    Goodbye

    End

The two flows '`pattern a`' and '`pattern b`' get immediately started from '`main`', waiting for a first user utterance action. After the user interaction you see how both the flows finish, since they matched the interaction pattern. Note, that the last bot action, saying "Goodbye", is the same in both flows and will therefore only be triggered once. Therefore, the ``$action_ref`` will actually point to the same action object. As we have seen before, an action will be stopped if the parent flow has finished. For an action that is shared in two concurrent flows this still holds, but it will only be forced to stop when both flows have finished.

We can make the same example using wrapper flows to abstract the actions and it will work exactly the same. Remember, that we don't have to write the ``await`` keyword since it is the default:

.. code-block:: colang
    :caption: flows/concurrent_flows_basics_wrapper/main.co

    flow main
        start pattern a as $flow_ref_a
        start pattern b as $flow_ref_b
        match $flow_ref_a.Finished() and $flow_ref_b.Finished()
        bot say "End"
        match RestartEvent()

    flow pattern a
        user said "Bye"
        bot say "Goodbye"

    flow pattern b
        user said "Hi"
        bot say "Hello"
        user said "Bye"
        bot say "Goodbye"

    flow user said $text
        match UtteranceUserAction.Finished(final_transcript=$text)

    flow bot say $text
        await UtteranceBotAction(script=$text)

This example will work identically when flow `'a'` uses a less specific match statement:

.. code-block:: colang

    # ...

    flow pattern a
        user said something
        bot say "Goodbye"

    # ...

    flow user said something
        match UtteranceUserAction.Finished()

Now, let's see what happens if two matching flows disagree on an action by differing in the two last statements:

.. code-block:: colang
    :caption: flows/action_conflict_resolution/main.co

    flow main
        start pattern a
        start pattern b
        match RestartEvent()

    flow pattern a
        user said something
        bot say "Hi"
        user said "How are you?"
        bot say "Great!"

    flow pattern b
        user said something
        bot say "Hi"
        user said something
        bot say "Bad!

    # ...


.. code-block:: text

    > Hello

    Hi

    > How are you?

    Great!

    > /RestartEvent
    > Welcome

    Hi

    > How are you doing?

    Bad!

We can see from this, that as long as the two flows agree they both will progress with their statements. This is also true at the third statement where flow '`pattern a`' is waiting for a specific user utterance, versus '`pattern b`' that is waiting for any user utterance. Where it gets interesting is at the last statement which is triggering a different action for each of these two flows that results in the generation of two different events. The concurrent generation of two different events conflicts by default in Colang and needs to be resolved. Only one can be generated, but which one? The resolution of conflicting event generation is done based on the specificity of the current pattern matching. The specificity is calculated as a matching score that depends on the number of parameters that are matching compared to all available parameters in the corresponding event. The matching score will be the highest if we have a match for all available event parameters. Since in the first run the user asked 'How are you?' and the third event matching statement in flow '`pattern a`' was the better match, flow '`pattern a`' will succeed triggering its action. Flow '`pattern b`' on the other hand will fail due to the conflict resolution. In the second run this is different and only '`pattern b`' will match and therefore progress.


.. pattern matching before. The specificity is calculated as a matching score for each match statement that depends on the number of parameters that are matching compared to all available parameters in the corresponding event (see section :ref:`More on Flows <more-on-flows-flow-conflict-resolution-prioritization>` for a more detailed discussion). The matching score will be the highest if we have a match for all available event parameters. Since in the first run the user asked 'How are you?' and the third event matching statement in flow '`pattern a`' was the better match, flow '`pattern a`' will succeed triggering its action. Flow '`pattern b`' on the other hand will fail due to the conflict resolution. In the second run this is different and only '`pattern b`' will match and therefore progress.

.. important::
    The concurrent generation of different events conflicts and will be resolved depending on the specificity (matching score) of the pattern matching. If the matching score is exactly the same, the event will be chosen at random.

When resolving an event generation conflict we only take into account the current event matching statements that lead to the event generation and ignore earlier pattern matches in the flows.

.. When resolving an event generation conflict all previous matches are taken into account to figure out which pattern matches better:

.. .. code-block:: colang
..     :caption: flows/action_conflict_resolution/main.co

..     flow main
..         start pattern a
..         start pattern b
..         match RestartEvent()

..     flow pattern a
..         user said "Hello"
..         bot say "Hi"
..         user said "How are you?"
..         bot say "Great and you?"
..         user said something
..         bot say "Thanks for sharing"

..     flow pattern b
..         user said something
..         user said something
..         user said "Bad"
..         bot say "What is bad?"

..     # Action wrapper flows
..     # ...

.. .. code-block:: text

..     > Hello

..     Hi

..     > How are you?

..     Great and you?

..     > Bad

..     Thanks for sharing

.. Note how the order of the matches does not matter, but only the accumulated matching score over all the matches. Pattern `'a'` matches better, even if the last match statement had a higher matching score in flow `'b'`.

----------------------------------------
Finished/Failed Flows
----------------------------------------

The interaction pattern of a flow can only end in two different ways. Either by successfully matching and triggering all events of the pattern (``Finished``) or by failing earlier (``Failed``).

An interaction pattern is considered to have successfully finished in one of the following cases:

A) All statements of the pattern were successfully processed and the flow reached its end.
B) A ``return`` statement is reached as part of the pattern that indicates that the pattern defined by the flow has successfully matched against the interaction (see section :ref:`Flow Control<flow-control-return-abort>`)
C) The pattern defined by the flow is considered to be successfully matched based on an internal event form another flow (see section :ref:`Internal Events<internal-events-defining-flows>`).


.. note::
    Remember: The ``Finished`` event of a flow is matched implicitly in the ``await`` statement that combines the start of the flow and then waits for it to finish.


If an interaction pattern in a flow fails, the flow itself is considered to fail, generating the ``Failed`` event. An interaction pattern can fail for one of the following reasons:

A) An action trigger statement (e.g. ``UtteranceBotAction(script="Yes")``) in the pattern conflicted with the action trigger statement of another concurrent pattern (e.g. ``UtteranceBotAction(script="No")``) with an action and was **less specific** than the other.
B) The current match statement of the pattern is waiting for an **impossible event** (e.g. waiting for a flow to finish that has failed).
C) An ``abort`` statement is reached as part of the pattern that indicates that the pattern cannot be matched (and therefore failed) against the interaction (see section :ref:`Flow Control<flow-control-return-abort>`).
D) The pattern fails due to an internal event that was generated by another flow (see section :ref:`Internal Events<internal-events-defining-flows>`).

In the context of flow hierarchies case B) plays a particularly important role. Let's see an example to understand this better:

.. code-block:: colang
    :caption: flows/flows_failing/main.co

    flow main
        start pattern a as $ref
        start pattern c
        match $ref.Failed()
        bot say "Pattern a failed"
        match RestartEvent()

    flow pattern a
        await pattern b

    flow pattern b
        user said something
        bot say "Hi"

    flow pattern c
        user said "Hello"
        bot say "Hello"

The user input "Hello" will result in the failure of flow `'pattern a'`:

.. code-block:: text

    > Hello

    Hello

    Pattern a failed

The reason for that lies in the way the flows fail:

1) The user utterance event "Hello" matches and advances `'pattern c'` and `'pattern b'` concurrently
2) Flow pattern `'pattern c'` and `'pattern b'` conflict due to their different actions  and `'pattern b'` fails since it is less specific
3) The failure of `'pattern b'` makes it impossible for flow `'pattern a'` to ever finish since it is waiting for flow `'pattern b'` to successfully finish, therefore `'pattern a'` fails as well (see case B)

A failing flow does not always need to result in the parent flow to fail as well, either by starting the flow asynchronously with the keyword ``start`` or by using the ``when/or when`` flow control construct (see section :ref:`Flow Control<flow-control-event-branching>`)

These are all the cases where a pattern can fail due to an impossible event:

- Event matching statement that waits for the ``FlowFinished`` event of a specific flow, but the flow fails.
- Event matching statement that waits for the ``FlowFailed`` event of a specific flow, but the flow finishes successfully.
- Event matching statement that waits for the ``FlowStarted`` event of a specific flow, but the flow finishes or fails.

.. - Event matching statement that waits for a event of an action or flow that has already finished

.. _defining-flows-flow-grouping:

----------------------------------------
Flow Grouping
----------------------------------------

Like for actions, we can use ``start`` and ``await`` on a flow group that is build using the grouping operators ``and`` and ``or``. Let's take a closer look at how this works based on the following four cases using the two placeholder flows `'a'` and `'b'`:

.. code-block:: colang

    # A) Starts both flows sequentially without waiting for them to finish
    start a and b
    # Equivalent representation:
    start a
    start b

    # B) Starts both flows concurrently without waiting for them to finish
    start a or b
    # No other representation

    # C) Starts both flows sequentially and waits for both flows to finish
    await a and b
    # Equivalent representation:
    start a as $ref_a and b as $ref_b
    match $ref_a.Finished() and $ref_b.Finished()

    # D) Starts both flows concurrently and waits for the first (earlier) to finish
    await a or b
    # Equivalent representation:
    start a as $ref_a or b as $ref_b
    match $ref_a.Finished() or $ref_b.Finished()

Cases A and C don't need much more explanation and should be pretty intuitive to understand. Cases B and D though, use the concept of concurrency that we have already seen in the pattern matching section before. If two flows get started concurrently they will progress together and potentially result in conflicting actions. The resolution of such conflicts is handled exactly the same. Let's see this with two concrete flow examples:

.. code-block:: colang

    flow main
        # A) Starts both bot actions sequentially without waiting for them to finish
        start bot say "Hi" and bot gesture "Wave with one hand"

        # B) Starts only one of the bot actions at random since they conflict in the two concurrently started flows
        start bot say "Hi" or bot gesture "Wave with one hand"

        # C) Starts both bot actions sequentially and waits for both of them to finish
        await bot say "Hi" and bot gesture "Wave with one hand"

        # D) Starts only one of the bot actions at random and waits for it to finish
        await bot say "Hi" or bot gesture "Wave with one hand"

    flow bot say $text
        await UtteranceBotAction(script=$text)

    flow bot gesture $gesture
        await GestureBotAction(gesture=$gesture)

.. code-block:: colang

    flow main
        # A) Starts both flows sequentially that will both wait for their user action event match
        start user said "Hi" and user gestured "Waving with one hand"

        # B) Starts both flows concurrently that will both wait for their user action event match
        start user said "Hi" or user gestured "Waving with one hand"

        # C) Wait for both user action events (order does not matter)
        await user said "Hi" and user gestured "Waving with one hand"

        # D) Waits for one of the user action events only
        await user said "Hi" or user gestured "Waving with one hand"

    flow user said $text
        match UtteranceUserAction.Finished(final_transcript=$text)

    flow user gestured $gesture
        match GestureUserAction.Finished(gesture=$gesture)

Note how:

- Case B of the first example also explains the underlying mechanics with an event generation or-group (see section :ref:`Event Generation - Event Grouping<event-generation-and-matching-event-grouping>`). The random selection is a result of the event conflict resolution and no special case.
- Case B in the second example with the user actions which has the same effect as case A. This might be a bit unexpected from a semantic point of view but is consistent with the underlying mechanics.

----------------------------------------
Mixing Flow, Action and Event Grouping
----------------------------------------

So far we have looked at event, action and flow grouping in separated contexts. But they can actually all be mixed in groups depending on the statement keyword.

- ``match``: Accepts only groups of events
- ``start``: Accepts groups of actions and flows but now events
- ``await``: Accepts groups of actions and flows but now events

.. code-block:: colang

    # Wait for either a flow or action to finish
    match (bot say "Hi").Finished() or UtteranceUserAction.Finished(final_transcript="Hello")

    # Combining the start of a flow and an action
    start bot say "Hi" and GestureBotAction(gesture="Wave with one hand")

    # Same as before but with additional reference assignment
    start bot say "Hi" as $bot_say_ref
        and GestureBotAction(gesture="Wave with one hand") as $gesture_action_ref

    # Combining awaiting (start and wait for them to finish) two flows and a bot action
    await bot say "Hi" or GestureBotAction(gesture="Wave with one hand") or user said "hi"

While this offers a lot of flexibility in how to design interaction patterns, it is considered "good design" to wrap all actions and events into flows before using them in the main interaction pattern designs.

.. _flow-naming-conventions:

--------------------------------
Flow Naming Convention
--------------------------------

You might have spotted by now the deliberate use of tenses in the naming of flows. While there are no binding rules on how you name your flows we do suggest to follow these conventions:

- Begin with flow names with a subject like ``bot`` or ``user`` if the flow is related to a system event/action that represents a bot or user action/intent.
- Use the imperative form of a verb to describe a bot action that should be executed, e.g. ``bot say $text``.
- Use the past form of a verb to describe an action that has happened, e.g. ``user said something`` or ``bot said something``
- Use the form ``<subject> started <verb continuous form> ...`` to describe an action that has started, e.g. ``bot started saying something`` or ``user started saying something``
- Start with the noun or gerund form of an activity for flows that should be activated and that wait for a certain interaction pattern to react to, e.g. ``reaction to user greeting``, ``handling user leaving`` or ``tracking bot talking state``.


.. _action-like-and-intent-like-flows:

----------------------------------------
Action-like and Intent-like Flows
----------------------------------------

We have already seen some examples of user and bot action-like flows:

.. code-block:: colang

    flow bot say $text
        await UtteranceBotAction(script=$text)

    flow bot gesture $gesture
        await GestureBotAction(gesture=$gesture)

    flow user said $text
        match UtteranceUserAction.Finished(final_transcript=$text)

    flow user gestured $gesture
        match GestureUserAction.Finished(gesture=$gesture)

With the help of these flows we can construct another abstraction, flows that represent bot or user intents:

.. code-block:: colang

    # A bot intent flow
    flow bot greet
        (bot say "Hi"
            or bot say "Hello"
            or bot say "Welcome")
            and bot gesture "Raise one hand in a greeting gesture"

    # A user intent flow
    flow user expressed confirmation
        user said "Yes"
            or user said "Ok"
            or user said "Sure"
            or user gestured "Thumbs up"

Note how the bot action-like flow will randomly combine one of the three utterances with the greeting gesture, whereas the user action-like flow will only finish if one of the specified user utterances or the user gesture was received. With the help of more examples or regular expressions those bot and user intent flows can be made more flexible. But they will never cover all the cases and in the section about :ref:`Making Use of Large Language Models<make-use-of-llms>` we will see how we can tackle that.

.. important::
    All the examples of a bot or user intent must be defined in a single statement in the flow using ``and`` or ``or`` to combine them. Flows containing multiple statements (comments excluded) will not be interpreted as intent-like flows.

.. _internal-events-defining-flows:

----------------------------------------
Internal Events
----------------------------------------

Besides all the events read and written to the event channel of the system, there is a special set of internal events that have priority over the system events and will not show up on the event channel:

.. code-block:: colang

    # Starts a new flow instance with the name flow_id and an unique instance identifier flow_instance_uid
    StartFlow(flow_id: str, flow_instance_uid: str, **more_variables)

    # Flow will be finished successfully either by flow_id or flow_instance_uid
    FinishFlow(flow_id: str, flow_instance_uid: str, **more_variables)

    # Flows will be stopped and failed either by flow_id or flow_instance_uid
    StopFlow(flow_id: str, flow_instance_uid: str, **more_variables)

    # Flow has started (reached first match statement or end)
    FlowStarted(flow_id: str, flow_instance_uid: str, **all_flow_variables, **more_variables)

    # Flow with name flow_id has finished successfully (containing all flow instance variables)
    FlowFinished(flow_id: str, flow_instance_uid: str, **all_flow_variables, **more_variables)

    # Flow with name flow_id has failed (containing all flow instance variables)
    FlowFailed(flow_id: str,  flow_instance_uid: str, **all_flow_variables, **more_variables)

    # Any unhandled (unmatched) event will generate a 'UnhandledEvent' event,
    # including all the corresponding interaction loop ids and original event parameters
    UnhandledEvent(event: str, loop_ids: Set[str], **all_event_parameters)

Note, that the parameter ``flow_id`` contains the name of the flow and the parameter ``flow_instance_uid`` the actual instance identifier, since the same flow can be started multiple times. Furthermore, for the second half of the internal events (including ``**all_flow_variables``), all flow parameters and variables will be returned.

Under the hood, all interaction patterns are based on these internal events. Have a look at the underlying mechanics of e.g. the ``await`` keyword:

.. code-block:: colang

    # Start of a flow ...
    await pattern a

    # is equivalent to
    start pattern a as $ref
    match $ref.Finished()

    # which is equivalent to
    $uid = "{uid()}"
    send StartFlow(flow_id="pattern a", flow_instance_uid=$uid)
    match FlowStarted(flow_instance_uid=$uid) as $ref
    match FlowFinished(flow_instance_uid=$ref.flow.uid)

Internal events can be matched to and generated like system events, but will be processed with priority to any next system event. This allows us to create more advance flows like e.g. a pattern that triggers when an undefined flow is called:

.. code-block:: colang
    :caption: flows/undefined_flow/main.co

    flow main
        activate notification of undefined flow start
        bot solve all your problems
        match RestartEvent()

    flow notification of undefined flow start
        match UnhandledEvent(event="StartFlow") as $event
        bot say "Cannot start the undefined flow: '{$event.flow_id}'!"
        # We need to abort the flow that sent the FlowStart event since it might be waiting for it
        send StopFlow(flow_instance_uid=$event.source_flow_instance_uid)

In the flow `'notification of undefined flow start'` we wait for an ``UnhandledEvent`` event that was triggered by a ``StartFlow`` event and will warn the user about the attempt to start an undefined flow.

Next, we will see more about how to work with :ref:`working-with-variables-and-expressions`.
