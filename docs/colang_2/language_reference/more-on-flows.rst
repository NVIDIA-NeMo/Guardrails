.. _more-on-flows:

========================================
More on Flows
========================================

.. .. note::
..     Feedbacks & TODOs:

..     .. - CS: Add section about one time flow activation for specific instance (i.e. same parameters)
..     .. - CS: Add section about override decorator

In section :ref:`Defining Flows<defining-flows>` we learned the core mechanisms of flows. In this section will look at more advanced topics that are related to flows.

.. _more-on-flows-activate-a-flow:

----------------------------------------
Activate a Flow
----------------------------------------

We already have seen the ``start`` and ``await`` keywords to trigger a flow. We are now introducing the third keyword ``activate`` that can start a flow. The difference to ``start`` lies in the behavior of the flow when it has finished or failed. If a flow was activated it will always automatically restart a new instance of the flow as soon as it has ended. Furthermore, a specific flow configuration (with identical flow parameters) can only be activated once and will not start new instances, even if activated multiple times.

.. important::
    Flow activation statement syntax definition:

    .. code-block:: text

        activate <Flow> [and <Flow>]…

    - Reference assignments for activated flows is not supported since the instance will change after a restart
    - Only and-groups are supported and not or-groups

    Examples:

    .. code-block:: colang

        # Activate a single flow
        activate handling user presents

        # Activate two different instances of the same flow with parameters
        activate handling user said "Hi"
        activate handling user said "Bye"

        # Activate a group of flows
        activate handling user presents and handling question repetition 5.0


.. code-block:: colang
    :caption: more_on_flows/activate_flow/main.co

    import core

    flow main
        activate managing user greeting
        bot say "Welcome"
        user said "Bye"
        bot say "Goodbye"
        match RestartEvent()

    flow managing user greeting
        user said "Hi"
        bot say "Hello again"

By running this example you will see the bot responding with "Hello again" as long as you keep greeting with "Hi":

.. code-block:: text

    Welcome

    > Hi

    Hello again

    > Hi

    Hello again

    > Bye

    Goodbye

    > Hi

    Hello again

    > Bye
    >

In contrast, you can only say "Bye" once before you restart the story.

Activating a flow enables you to keep matching the interaction event sequence against the pattern defined in the flow, even if the pattern previously successfully matched the interaction event sequence (finished) or failed. Since the same flow configuration can only be activated once, you can use the flow activation directly wherever you require the flow's functionality. This `on demand pattern` is better than activating it once in the beginning before you actually know if it is needed.

.. important::
    Activating a flow will start a flow and automatically restart it when it has ended (finished or failed) to match to reoccurring interaction patterns.

Alternatively, you can use the ``@active`` decorator notation to activate a flow at the start as a child of the main flow:

.. code-block:: colang

    import core

    flow main
        bot say "Welcome"
        user said "Bye"
        bot say "Goodbye"
        match RestartEvent()

    @active
    flow managing user greeting
        user said "Hi"
        bot say "Hello again"

If you use the ``@active`` decorator for flows that were defined in a separate Colang library module, they will get automatically activated when the library is imported. But we advice you to use the ``activate`` statement if possible, since it is more explicit and result in better readability.

.. important::
    The main flow behaves also like an activated flow. As soon as it reaches the end it will restart automatically.

There is one exception though from this rule! If a flow does not contain any statement that waits for an event and immediately finishes, it will run only once when activated and it will stay activated since otherwise you would get an infinite loop.

.. code-block:: colang
    :caption: more_on_flows/non-repeating-flows/main.co

    import core

    flow main
        activate managing user greeting
        # No additional match statement need to keep this flow activated without repeating

    flow managing user greeting
        user said "Hi"
        bot say "Hello again"

.. code-block:: text

    > Hi

    Hello again

    > Hi

    Hello again

See, how the main flow does not require any match statement at the end and will continue to be activated without repeating, even though it reached the end.

.. important::
    An activated flow that immediately finished (does not wait for any event) will only be run once and will stay activated.

.. _more-on-flows-start-a-new-flow-instance:

----------------------------------------
Start a new Flow Instance
----------------------------------------

In some cases it is not enough to restart a flow only once it has finished since this can miss certain pattern repetitions:

.. code-block:: colang
    :caption: more_on_flows/restart_flow_instance/main.co

    import core

    flow main
        activate managing user presence
        bot say "Welcome"
        match RestartEvent()

    flow managing user presence
        user said "Hi"
        bot say "Hello again"
        user said "Bye"
        bot say "Goodbye"

In the following interaction we see that the second "Hi" of the user does not trigger anything since the flow already advanced to the next statement ``user said "Bye"`` but did not yet start an new instance due to its activation:

.. code-block:: text

    Welcome

    > Hi

    Hello again

    > Hi
    > Bye

    Goodbye

    > Hi

    Hello again

    >

If we want to start an new instance before the end of the current instance, we can achieve that by adding a label called ``start_new_flow_instance`` at the corresponding position in the interaction sequence:

.. code-block:: colang
    :caption: more_on_flows/start_new_flow_instance/main.co

    # ...

    flow managing user presence
        user said "Hi"

        start_new_flow_instance: # Start a new instance of the flow and continue with this one

        bot say "Hello again"
        user said "Bye"
        bot say "Goodbye"

We now see the correct behavior:

.. code-block:: text

    Welcome

    > Hi

    Hello again

    > Hi

    Hello again

    > Bye

    Goodbye

    > Bye
    >

Note, that as soon as the second instance advances to the next match statement, a third instance is started, waiting for the next user input "Hi". The other two instances will advance in parallel.
Since the first instance already started a new instance (second one) it will not start another one such that we don't get a growing number of instances when progressing. Note how the second "Bye" will not trigger anything since the first and second instance have already finished and the third instance is still at the first statement waiting for a "Hi".

.. note::
    You can think of the ``start_new_flow_instance`` label being at the end of each activated flow. Defining it in a different position will move it up from the default position at the end.

.. _more-on-flows-deactivate-a-flow:

----------------------------------------
Deactivate a Flow
----------------------------------------

An activated flow will usually stay alive since it always restarts when it finishes or fails. To deactivate an activated flow you can use the `deactivate` keyword:

.. important::
    Flow deactivation statement syntax definition:

    .. code-block:: text

        deactivate <Flow>

    Examples:

    .. code-block:: colang

        # Deactivate a single flow
        deactivate handling user presents

        # Deactivate two different instances of the same flow with different parameters
        deactivate handling user said "Hi"
        deactivate handling user said "Bye"

Under the hood the `deactivate` keyword will abort the flow and disable the restart. It is a shortcut for this statement:

.. code-block:: colang

    send StopFlow(flow_id="flow name", deactivate=True)


.. _more-on-flows-override-flows:

---------------
Override Flows
---------------

A flow can be overridden by another flow with the same name by using the override decorator:

.. code-block:: colang

    flow bot greet
        bot say "Hi"

    @override
    flow bot greet
        bot say "Hello"

In this example the second '`bot greet`' flow will override the first one. This is particularly useful when working with imported Colang modules from a library to override, e.g. the '`bot say`' flow from the core module of the standard library to include an additional log statement:

.. code-block:: colang

    import core

    flow main
        bot say "Hi"

    @override
    flow bot say $text
        log "bot say {$text}"
        await UtteranceBotAction(script=$text) as $action

At the moment the definition order of flows does not make a difference and therefore only two flows with the same name can be defined where one must have the override decorator.

.. note::

    If two flows have the same name, one must be prioritized by the override decorator.

.. _more-on-flows-interaction_loops:

-------------------
Interaction Loops
-------------------

So far, any concurrently progressing flows that resulted in different event generations created a conflict that needed to be resolved. While this makes sense in many cases, sometimes one would like to allow different actions to happen at the same time. In particular, when these actions are on different modalities. We can achieve this by defining different interaction loops using the a decorator style syntax on flows:

.. important::
    Interaction loop syntax definition:

    .. code-block:: colang

        @loop([id=]"<loop_name>"[,[priority=]<integer_number>])
        flow <name of flow> ...

    Hint: To generate a new loop name for each flow call use the loop name "NEW"

By default, any flow without an explicit interaction loop inherits the interaction loop of its parent flow and has priority level 0. Let's see now an example of a second interaction loop to design flows that augment the main interaction rather than compete with it:

.. code-block:: colang
    :caption: more_on_flows/interaction_loops/main.co

    import core
    import avatars

    flow main
        activate handling bot gesture reaction
        while True # Keep reacting to user inputs
            when user said "Hi"
                bot say "Hi"
            or when user said something
                bot say "Thanks for sharing"
            or when user said "Bye"
                bot say "Goodbye"

    @loop("bot gesture reaction")
    flow handling bot gesture reaction # Just a grouping flow for different bot reactions
        activate reaction of bot to user greeting
        activate reaction of bot to user leaving

    flow reaction of bot to user greeting
        user said "Hi"
        bot gesture "smile"

    flow reaction of bot to user leaving
        user said "Bye"
        bot gesture "frown"


The example implements two bot reaction flows that listen to the user saying "Hi" or "Bye". Whenever one of the two events happen the bot will show the corresponding gesture "smile" or "frown", respectively. Note how these flows inherit their interaction loop id from the parent flow `'handling bot gesture reaction'` that is different from the main flow. Therefore, bot gesture actions will never compete with the bot say actions from the main interaction flow and will be triggered in parallel:

.. code-block:: text

    > Hi
    Gesture: smile

    Hi

    > I am feeling great today

    Thanks for sharing

    > I am looking forward to my birthday

    Thanks for sharing

    > Bye
    Gesture: frown

    Goodbye

By default, parallel flows in different interaction loops advance in order of their start or activation. This might be an important detail if e.g a global variable is set in one flow and read in another. If the order is wrong, the global variable will not be set yet when read by the other flow. In order to enforce the processing order independent of the start or activation order, you can define the interaction loop priority level using an integer. By default, any interaction loop has priority 0. A higher number defines a higher priority, and lower (negative) number a lower processing priority.


.. _more-on-flows-flow-conflict-resolution-prioritization:

----------------------------------------
Flow Conflict Resolution Prioritization
----------------------------------------

In section :ref:`Defining Flows<defining-flows-concurrent-pattern-matching>` we have already learned a bit about the mechanics of resolving an action conflict between flows. We will now look at this in more detail.

For every successful match statement a matching score is computed that is greater than :math:`0.0` (no match) and smaller or equal to :math:`1.0` (perfect match). A perfect match is when all parameters of the expected event match with all the parameters from the actual event. If the actual event has more parameters than the expected event the matching score will be decreased by multiplying it by a factor of :math:`0.9` for every missing parameter. So let's say we have a matching event containing five parameters, but we only specified two of them, the score would be :math:`0.9^{5-2} = 0.729`. Since a system event can trigger a chain of internal events we need to take into account all the generated matching scores in that sequence. Let's use the following example to better illustrate that:

.. code-block:: colang

    flow main
        activate pattern a and pattern b

    flow pattern a
        user said "Hi"
        bot say "Hello"

    flow pattern b
        user said something
        bot say "Sure"

    flow user said $text
        match UtteranceUserActionFinished(final_transcript=$text)

    flow user said something
        match UtteranceUserActionFinished()

    flow bot say $text
        await UtteranceBotAction(script=$text)

After starting the main flow, the two flows `'pattern a'` and `'pattern b'` will be active and waiting for the user to say something. Let's look at the two event generation chains triggered by the event ``UtteranceUserActionFinished(final_transcript="Hi")``:

.. code-block:: colang

    1) UtteranceUserActionFinished(final_transcript="Hi") -> send FlowFinished(flow_id="user said", text="Hi") -> send StartFlow(flow_id="bot say", text="Hello") -> send StartUtteranceBotAction(text="Hello")
    2) UtteranceUserActionFinished(final_transcript="Hi") -> send FlowFinished(flow_id="user said something") -> send StartFlow(flow_id="bot say", text="Sure") -> send StartUtteranceBotAction(text="Sure")

Because the resulting action events at the end of these chains are different, there will be a conflict that needs to be resolved. Let's look at the corresponding match statements in these chains:

.. code-block:: colang

    1) match UtteranceUserActionFinished(final_transcript="Hi") -> match FlowFinished(flow_id="user said", text="Hi") -> match StartFlow(flow_id="bot say", text="Hello")
    2) match UtteranceUserActionFinished() -> match FlowFinished(flow_id="user said something") -> match StartFlow(flow_id="bot say", text="Sure")

Comparing these match statements to the events will result in the following matching scores:

.. code-block:: colang

    1) 1.0 -> 1.0 -> 1.0
    2) 0.9 -> 1.0 -> 1.0

In order to find the best event matching sequence we will compare each matching score from the different chains from left to right and determine the winner as soon as one score is higher than the other. You see that the first match in the second chain is not perfect and resulted in a value of :math:`0.9`. Therefore, the first chain is the winner and the second will fail, resulting in the following output:

.. code-block:: text

    > Hi

    Hello

In some cases you might want to influence the matching score of some matches to change the conflict resolution outcome. You can do this by specifying a flow priority with the statement ``priority <float_value>`` where the value is between :math:`0.0` and :math:`1.0`. Each match in the flow will then be multiplied by the current flow priority. Since this approach currently can only reduce the matching score you cannot use it to increase the priority of a match. A work around that can sometimes be employed is to improve the matching score of a non-perfect match by adding missing parameters using a regular expression that matches any value like that ``regex(".*")``:

.. code-block:: colang

    # ...

    flow user said something
        match UtteranceUserActionFinished(final_transcript=regex(".*"))

    # ...

In this example the conflict resolution between the actions will happen at random since all the matching scores are equal.
