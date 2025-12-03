.. _flow-control:

========================================
Flow control
========================================

.. .. note::
..     Feedbacks & TODOs:

Flow control is an essential tool in all programming languages and Colang supports this as well. It enables branching and repetition of interaction patterns in a single flow.

----------------------------------------
Conditional Branching (``if/elif/else``)
----------------------------------------

.. important::
    Syntax definition of conditional branching:

    .. code-block:: text

        if <condition1>
            <interaction pattern sequence 1>
        [elif <condition2>
            <interaction pattern sequence 2>]
        .
        .
        .
        [else
            <interaction pattern else sequence>]

The conditional branching is a well known concept and works identical to Python:

.. code-block:: colang
    :caption: control_flow_tools/conditional_branching/main.co

    flow main
        $number_of_users = 1
        if $number_of_users == 0
            await user became present
        elif $number_of_users > 1
            bot say "I am sorry, I can only interact with a single user!"
        else
            bot say "Welcome! Nice to meet you!"
        match RestartEvent()

In this example the bot's reaction depends on the state of the variable ``$number_of_users`` that would contain the number of available users.

.. _flow-control-event-branching:

----------------------------------------
Event Branching (``when/or when/else``)
----------------------------------------

Event branching is a new Colang based concept that enables a branching based on expected events.

.. important::
    Syntax definition of event based branching:

    .. code-block:: text

        when <MixedGroup1>
            <interaction pattern sequence 1>
        [or when <MixedGroup2>
            <interaction pattern sequence 2>]
        .
        .
        .
        [else
            <interaction pattern else sequence>]


    - The `<MixedGroup>` stands for a mixed grouping of flows, actions and events
    - All actions and flows in the ``when/or when`` statements will be started concurrently
    - If neither the ``or when`` nor the ``else`` statement is used, the ``when`` construct could be replace with either just an ``await`` or ``match`` statement

With the concurrent pattern matching mechanism we have already seen one way to design a branching interaction pattern based on the users input:

.. code-block:: colang
    :caption: control_flow_tools/concurrent_patterns/main.co

    flow main
        bot say "How are you?"
        bot react to user feeling good or bot react to user feeling bad

    flow bot react to user feeling good
        user said "Good" or user said "Great"
        bot say "Great"

    flow bot react to user feeling bad
        user said "Bad" or user said "Terrible"
        bot say "Sorry to hear"

Depending on the user's answer we will get a different bot reaction. Although this concurrent flow mechanism is very powerful, it is sometimes better to have everything in a single flow with the help of the ``when`` construct:

.. code-block:: colang
    :caption: control_flow_tools/event_branching/main.co

    flow main
        bot say "How are you?"
        bot react to user wellbeing

    flow bot react to user wellbeing
        when user said "Good" or user said "Great"
            bot say "Great"
        or when user said "Bad" or user said "Terrible"
            bot say "Sorry to hear"

The number of cases can easily be extended by adding more ``or when`` statements. The ``else`` statement will only trigger if all of the ``when/or when`` statements have failed.

From the definition we see that ``when/or when`` statements support mixed groups that can contain events, actions and flows. For events this works like a ``match`` statement, whereas for actions and flows it behaves like an ``await`` statement. Therefore, actions and flows will be started and then matched with their ``Finished`` event. Note, that all flows and actions will be started concurrently in the different ``when/or when`` statements and stopped as soon as the first case succeeds.

.. important::
    All started flows and actions in all the ``when/or when`` statements will be stopped as soon as one of the cases succeeded.

We can also use this construct to easily create a branching for a flow that either finishes or fails:

.. code-block:: colang
    :caption: control_flow_tools/catch_failing_flow/main.co

    flow main
        start pattern a
        when pattern b
            bot say "Pattern b has finished"
        else
            bot say "Pattern b has failed"

    flow pattern a
        user said "Hello"
        bot say "Hello"

    flow pattern b
        user said something
        bot say "Hi"

Due to the event generation conflict resolution `'pattern b'` will fail for the user input "Hello", but successfully finish for the user input "Hi":

.. code-block:: text

    > Hello

    Hello

    Pattern b has failed

    > Hi

    Hi

    Pattern b has finished


It is considered "bad design" when used with action-like flows that start with an action:

.. code-block:: colang

    flow bot greet then comment
        when bot say "Hi there!"
            bot say "I am done talking first"
        or when bot gesture "Wave with one hand"
            bot say "I am done gesturing first"

    flow bot say $text
        await UtteranceBotAction(script=$text)

    flow bot gesture $gesture
        await GestureBotAction(gesture=$gesture)

This example will not work correctly because only one of the two actions will be started due to the action conflict between ``UtteranceBotAction`` and ``GestureBotAction``. Note, that such cases can be easily detected when following a proper flow :ref:`naming convention<flow-naming-conventions>` since ``when bot say "Hi there!"`` is grammatically incorrect. The example above would need to be implemented like this:

.. code-block:: colang

    flow bot greet then comment
        start bot say "Hi there!" as $action_1_ref
            and bot gesture "Wave with one hand" as $action_2_ref
        when $action_1_ref.Finished()
            bot say "I am done talking first"
        or when $action_2_ref.Finished()
            bot say "I am done gesturing first"

.. important::
    The ``when/or when/else`` branching should only be used with intent-like flows.

----------------------------------------
Loop (``while``)
----------------------------------------

.. important::
    Syntax definition of a loop:

    .. code-block:: text

        while <condition>
            <interaction pattern sequence>

In this example the bot will count from one to ten:

.. code-block:: colang
    :caption: control_flow_tools/loop/main.co

    flow main
        bot count to 10

    flow bot count to $number
        $current_number = 1
        while $current_number < $number
            bot say "{$current_number}"
            $current_number = $current_number + 1

In order to abort the loop early or to skip the rest of the current loop iteration the keywords ``break`` and ``continue`` can be used, respectively:

.. code-block:: colang

    flow bot count to $number
        $current_number = 0 # Initialized it with 0
        while True # Endless loop
            bot say "{$current_number}"
            $current_number = $current_number + 1
            if $current_number == 0
                continue # Skip the number 0
            if $current_number > $number
                break # Break out of loop when target number was reached

.. _flow-control-return-abort:

------------------------------------------
Finish or abort a flow (``return/abort``)
------------------------------------------

Flows can be finished or failed at any point from within the flow using the keywords ``return`` and ``abort``, respectively:

.. code-block:: colang

    flow main
        user greeted then expressed feeling unwell

    flow user greeted then expressed feeling unwell
        match user greeted
        when user expressed feeling unwell
            return
        or when user said something
            abort
        # We never reach this, except if both cases fail

Additionally, ``return`` takes an optional value such that you can use a flow like a common function:

.. code-block:: colang

    flow main
        $result = await multiply 3 4
        bot say "{$result}"

    flow multiply $number_1 $number_2
        return $number_1 * $number_2

If no return value is provided you ``None`` is passed by default.

.. note::
    When assigning the return value of a flow to a variable ``await`` is not optional before the flow name.

----------------------------------------
No-op operation (``pass``)
----------------------------------------

Sometimes, it is useful to have a no-operation keyword ``pass``, e.g. as placeholder to make e.g. the syntax valid:

.. code-block:: colang

    flow main
        user greeted then expressed feeling unwell

    flow user greeted then expressed feeling unwell
        match user greeted
        when user expressed feeling unwell
            pass # Just continue with the flow
        or when user said something
            abort
        # The flow will successfully finish here
