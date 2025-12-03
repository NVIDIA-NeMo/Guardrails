.. _working-with-actions:

========================================
Working with Actions
========================================

.. .. note::
..     Feedbacks & TODOs:

..     .. - CS: Add explanation about implicit action state update by events.

----------------------------------------
Introduction
----------------------------------------

You have seen in section :ref:`event-generation-and-matching` how to work with events. We will now introduce actions that will enable you to better manage events and their temporal dependencies.

.. important::
    Action definition:

    .. code-block:: text

        <ActionName>[(param=<value>[, param=<value>]…)]

    Examples:

    .. code-block:: colang

        # Bot utterance action
        UtteranceBotAction(script="Hello", intensity=1.0)

        # Bot gesture action
        GestureBotAction(gesture="Wave with one hand")

    An action uses the Pascal case style naming like events, but ends with ``Action``.

Different from events, actions have a lifetime that depends on the underlying events. All actions contain the following main events, where we use X as a placeholder for the action name:

.. code-block:: colang

    StartXAction(**action_arguments) # Start of an action with given action specific arguments
    StopXAction(action_uid: str) # Stop of an action with the related uid
    XActionStarted(action_uid: str) # Event to signal the action has started
    XActionFinished(action_uid: str, is_success: bool, was_stopped: bool, **action_arguments) # Event to signal that the action has ended

In Colang an action is like an object in Python and the events can be accessed like an object method of the action.

.. code-block:: colang

    XAction(**action_arguments).Start()
    XAction.Stop(action_uid: str)
    XAction.Started(action_uid: str)
    XAction.Finished(action_uid: str, is_success: bool, was_stopped: bool, **action_arguments)

Here are two common life cycles of an action that depend on the events:

.. code-block:: text

    XAction.Start() -> Action state: `Starting`
    XAction.Started() -> Action state: `Running`
    XAction.Finished() -> Action state: `Finished`

.. code-block:: text

    XAction.Start() -> Action state: `Starting`
    XAction.Started() -> Action state: `Running`
    XAction.Stop() -> Action state: `Stopping`
    XAction.Finished() -> Action state: `Finished`

In the :ref:`UMIM <UMIM intro>` reference documentation you will find many predefined actions that should cover the most common use cases. Let's have a look at one of the most prominent actions called `UtteranceBotAction`. This action represents the main channel to communicate with the user, e.g. through speech or text (depending on the specific system). This action relates to events that can be grouped into output and input events from the perspective of a bot:

Output events:

.. code-block:: colang

    UtteranceBotAction(script: str).Start()
    UtteranceBotAction.Stop(action_uid: str)
    UtteranceBotAction.Change(action_uid: str, intensity: float)

Input events:

.. code-block:: colang

    UtteranceBotAction.Started(action_uid: str, action_started_at: str, ...)
    UtteranceBotActionScript.Updated(action_uid: str, interim_transcript: str, ...)
    UtteranceBotAction.Finished(action_uid: str, final_transcript: str, ...)

.. note::

    Note, that Colang is not limited to work only from a bot perspective but could also be used to simulate the user side. In this case the output and input categorization would flip.

While there are no binding rules on how we work with these events using the ``send`` or ``match`` keywords, in most cases we will generate output events and match to input events. E.g. if we want the bot to finish saying something before making a gesture we could create the following interaction pattern:

.. code-block:: colang
    :caption: actions/action_events/main.co

    flow main
        match StartEvent()
        send UtteranceBotAction(script="Hello").Start() as $event_ref
        match UtteranceBotActionFinished(action_uid=$event_ref.action_uid)
        send GestureBotAction(gesture="Wave").Start()

.. code-block:: text

    > /StartEvent

    Hello

    Gesture: Wave

We see that by using the `action_uid` property of the `Start` action event reference, we can match to the `Finished` event of the specific action. This is important, since there could be other `UtteranceBotAction` finishing before.

----------------------------------------
`Start` an Action
----------------------------------------

Colang supports several features based on the action concept that make designing interaction patterns more convenient. The first feature we introduce is the  ``start`` statement.

.. important::
    Start statement definition:

    .. code-block:: text

        start <Action> [as $<action_ref_name>] [and|or <Action> [as $<action_ref_name>]…)]

    Example:

    .. code-block:: colang

        start UtteranceBotAction(script="Hello") as $bot_action_ref

.. code-block:: colang
    :caption: actions/start_keyword/main.co

    flow main
        start UtteranceBotAction(script="Hello") as $ref_action
        match $ref_action.Finished()
        start GestureBotAction(gesture="Wave")
        match RestartEvent()

.. code-block:: text

    Hello

    Gesture: Wave

The keyword ``start`` creates an action object and then generates an action specific `Start` action event. A reference to this action object can be stored using ``as $ref_name``. With this action reference we can now conveniently match to the `Finished` event and no longer need to use the `action_uid` parameter to identify the specific action.

Let's look now at an example of the common user action `UtteranceUserAction`. This action represents the counterpart to the `UtteranceBotAction` and is the main channel of the user to expressing herself, e.g. by talking, writing or any other way depending on the actual system. A user action is usually not started by Colang but by the user (system). These are the most important action events and parameters:

.. code-block:: colang

    UtteranceUserActionStarted(action_uid: str)
    UtteranceUserActionTranscriptUpdated(action_uid: str, interim_transcript: str)
    UtteranceUserActionIntensityUpdated(action_uid: str, intensity: float)
    UtteranceUserActionFinished(action_uid: str, final_transcript: str)

With this, let's now build a little dialog pattern:

.. code-block:: colang
    :caption: actions/dialog_pattern/main.co

    flow main
        match UtteranceUserAction.Finished(final_transcript="Hi")
        start UtteranceBotAction(script="Hi there! How are you?") as $ref_action_1
        match $ref_action_1.Finished()
        match UtteranceUserAction.Finished(final_transcript="Good and you?")
        start UtteranceBotAction(script="Great! Thanks") as $ref_action_2
        start GestureBotAction(gesture="Thumbs up") as $ref_action_3
        match $ref_action_2.Finished() and $ref_action_3.Finished()

.. code-block:: text

    > Hi

    Hi there! How are you?

    > Good and you?

    Great! Thanks

    Gesture: Thumbs up

As you might have already noticed, this is very similar to the example we saw in the introduction example `introduction/interaction_sequence/main.co`. The difference is that we have more temporal control and will only start matching user input when the bot has completed the utterance. Note also, how the interaction pattern is completed only once the second bot utterance action and the bot gesture action have both finished.

----------------------------------------
`Await` an Action
----------------------------------------

Let's introduce the ``await`` statement to further simplify the previous example:

.. code-block:: colang
    :caption: actions/await_keyword/main.co

    flow main
        match UtteranceUserAction.Finished(final_transcript="Hi")
        await UtteranceBotAction(script="Hi there! How are you?")
        match UtteranceUserAction.Finished(final_transcript="Good and you?")
        # ...


``await`` is a shortcut notation for starting an action and waiting for the action to be finished (i.e matching the ``.Finished()`` event.)

.. important::
    Await statement definition:

    .. code-block:: text

        await <Action> [as $<action_ref_name>] [and|or <Action> [as $<action_ref_name>]…)]

    Example:

    .. code-block:: colang

        await UtteranceBotAction(script="Hello") as $bot_action_ref

Unfortunately, we cannot simplify the second part of the example with this... Or can we though? Actually yes! We can make use of action grouping using the ``and`` keyword to simplify it like this:

.. code-block:: colang
    :caption: actions/action_grouping/main.co

    flow main
        match UtteranceUserAction.Finished(final_transcript="Hi")
        await UtteranceBotAction(script="Hi there! How are you?")
        match UtteranceUserAction.Finished(final_transcript="Good and you?")
        await UtteranceBotAction(script="Great! Thanks") and GestureBotAction(gesture="Thumbs up")

Action grouping is identical to event grouping using the keywords ``start`` and ``await`` instead of ``send`` and ``match``.

.. important::
    Note, that this:

    .. code-block:: colang

        await Action1() or Action2()

    will randomly start only one of the actions (not both!) and wait for it to finish

To simplify it even more, we can actually omit all the ``await`` keywords completely, since it is the default statement keyword.

.. code-block:: colang
    :caption: actions/omit_wait_keyword/main.co

    flow main
        match UtteranceUserAction.Finished(final_transcript="Hi")
        UtteranceBotAction(script="Hi there! How are you?")
        match UtteranceUserAction.Finished(final_transcript="Good and you?")
        UtteranceBotAction(script="Great! Thanks") and GestureBotAction(gesture="Thumbs up")

.. important::
    ``await`` is the default statement keyword and can be omitted.

If we would like to start two actions and only wait until either one of them has finished, we can achieve this like this:

.. code-block:: colang
    :caption: actions/wait_for_first_action_only/main.co

    flow main
        match StartEvent()
        start UtteranceBotAction(script="Great! Thanks") as $ref_action_1
            and GestureBotAction(gesture="Thumbs up") as $ref_action_2
        match $ref_action_1.Finished() or $ref_action_2.Finished()

----------------------------------------
More about Actions
----------------------------------------

If needed, we can also stop an action using its reference like this:

.. code-block:: colang
    :caption: actions/stop_action/main.co

    flow main
        match StartEvent()
        start UtteranceBotAction(script="Great! Thanks") as $ref_action
        send $ref_action.Stop()

Unfortunately, with the simple chat CLI we will not see any effect of the `Stop` event since the utterance is done immediately. But in any real system, the utterance will take some time to finish and can be stopped like this if needed.

Another detail to point out is the difference between matching to an action event accessed via an action reference versus matching directly by an action:

.. code-block:: colang

    # Case 1) Wait for Finished event of the specific action
    start UtteranceBotAction(script="hi") as $action_ref
    match $action_ref.Finished()

    # Case 2) Wait for the Finished event of any UtteranceBotAction
    match UtteranceBotAction.Finished()

In the first case the match is on the specific action reference (same action_uid parameter) and will not match to any other `Finished` event of another `UtteranceBotAction`. The second case is more general and will match to any `Finished` event from any `UtteranceBotAction`.

.. Furthermore, actions that were started are implicitly updated by relevant events, even if there is no related matching statement:

With this, we are now prepared to learn more about the concept of flows in the next chapter :ref:`defining-flows`.
