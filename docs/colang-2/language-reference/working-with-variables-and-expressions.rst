.. _working-with-variables-and-expressions:

========================================
Working with Variables & Expressions
========================================

.. .. note::
..     Feedbacks & TODOs:

..     .. - CS: Add section about expression evaluation (e.g. with ($var_1 + $var_2))
..     .. - CS: Add section about variable context update through Context update event

----------------------------------------
Introduction
----------------------------------------

Like Python, Colang supports these fundamental data types: `string`, `int`, `float`, `bool`, `list`, `set`, `dict`, in addition to the Colang specific event, action and flow references. We have already seen how to assign event, action and flow references to a variable using the ``... as $ref`` notation. Here ``$ref`` is also just a variable. To enable whitespace characters in flow names Colang variables must always start with the ``$`` character and cannot contain whitespace characters themselves.

.. important::
    Variable naming convention:

    - Has to start with an alphabetic character
    - Can contain alphanumeric characters including ``_``
    - Is case sensitive

    .. code-block:: text

        Regular expression: \$[^\W\d]\w*

Variables in Colang behave exactly like in Python. As a consequence, mutable data types like a list will not be copied for a new variable assignment but only referenced by the new variable.

.. code-block:: colang
    :caption: variables/assignment/main.co

    flow main
        $value = "Hi"
        $value_copy = $value # Copy of the value
        await UtteranceBotAction(script=$value_copy)
        match AnEvent()

Note, that Colang references are also mutable and therefore point to the same underlying object:

.. code-block:: colang
    :caption: variables/references/main.co

    flow main
        match UtteranceUserAction.Finished() as $ref
        $ref_copy = $ref # Both references are pointing to same event object
        await UtteranceBotAction(script=$ref_copy.final_transcript)
        match AnEvent()

Currently, the creation of references of other variables is not supported. References can only point to event, action or flow objects and are created in ``send``, ``match``, ``start`` or ``await`` statements using the ``as`` keyword.

.. important::

    - Assignment to variables will always create a copy of the value in memory
    - Copy of event, action or flow references keep pointing to the same object
    - Flow parameters are passed by value

Here are some variable assignment examples:

.. code-block:: colang

    $string_value = "Hi"
    $integer_value = 42
    $float_value = 3.14159
    $list_of_strings = ["one", "two", "three"]
    $list_of_integers = [1, 2, 3, 4]
    $set_of_floats = {0.1, 0.2, 0.3}
    $dictionary = {"value_a":1, "value_b": 2}

----------------------------------------
Expression Evaluation
----------------------------------------

Colang supports evaluation of common Python expressions for simple and compound data types (see `Simple Eval <https://github.com/danthedeckie/simpleeval>`_):

.. code-block:: colang

    # Arithmetic expressions
    21 + 21
    21 + 19 / 7 + (8 % 3) ** 9

    # Supported operators
    +, -, *, / # standard arithmetic operators
    ** # to the power of: 2 ** 10 -> 1024
    % # modulus
    ==, <, >, <=, >= # comparison operators
    and, or, not # logical operators
    in # is something contained within something else
    >>, <<, ^, |, &, ~ # Bitwise operators

    # Conditional expressions
    "equal" if x == y else "not equal"
    "a" if 1 == 2 else "b" if 2 == 3 else "c"

    # Compound data types
    list_variable[0] # Access item by index
    dict_variable[key] # Access item by key
    object.attribute # Access object attribute

    # Supported custom functions
    len(obj: Any) -> int # Return number of items of a compound variable
    regex(pattern: str) -> Pattern # Creates a regex pattern that can be compared to
    search(pattern: str, string: str) -> bool # Check for regex pattern in string
    findall(pattern: str, string: str) -> List[str] # Return all matches of regex pattern with string
    uid() -> str # Create new universal unique identifier
    int(string: str) -> int # Convert the number in the string to an int
    float(string: str) -> float # Convert the number in the string to a float
    str(x: Any) -> str # Convert x to a string
    pretty_str(x: Any) -> str # Convert x to a formatted string
    int(x: Any) -> int # Convert x to a int
    float(x: Any) -> float # Convert x to a float
    escape(string: str) -> str # Escape a string and expressions inside the string
    is_bool(x: Any) -> bool # Check if x is a bool
    is_int(x: Any) -> bool # Check if x is an int
    is_float(x: Any) -> bool # Check if x is a float
    is_str(x: Any) -> bool # Check if x is a str
    is_regex(x: Any) -> bool # Check if x is a regex pattern
    type(x: Any) -> str # Returns type as string of object x
    list(x: Iterable[T]) -> list[T] # Converts an iterable object to a list
    rand() -> float # Return a random float between 0 and 1
    randint(x: int) -> int # Return a random int below x
    flows_info() -> dict # Returns a dictionary that contains more information about the current flow

Here is how expression can be used withing Colang:

.. code-block:: colang

    # Expression in an assignment
    $dict = {"value": 2 + 3}

    # Expression as standalone statement
    ($dict.update({"value": 4}))

    # Expression as a flow parameter
    bot count to ($dict["value"])
    bot count to (int("3"))

You see how expressions can be used in different context and need to be wrapped in parentheses if used as a *standalone statement* or as a *flow parameter*.


----------------------------------------
Flow Variable Access
----------------------------------------

By default variables defined in a flow have a local scope and are not accessible from outside the flow. One way to enable access to them is by declaring them as flow attributes using the notation shown in :ref:`defining-flows` in the flow definition:

.. code-block:: colang
    :caption: variables/flow_attributes/main.co

    flow main
        await user said something as $ref
        await UtteranceBotAction(script=$ref.transcript)
        match AnEvent()

    flow user said something -> $transcript
        match UtteranceUserAction.Finished() as $event_ref
        $transcript = $event_ref.final_transcript

With this we can e.g. access the user transcript and use it to repeat it with a bot utterance action.

Another way to share information between flows using variables is to make it global by using the keyword ``global``.

.. code-block:: colang
    :caption: variables/global_variables/main.co

    flow main
        global $transcript
        await bot said something
        await UtteranceBotAction(script=$transcript)
        match AnEvent()

    flow bot said something
        global $transcript
        match UtteranceUserAction.Finished() as $event_ref
        $transcript = $event_ref.final_transcript

As you can see from the example, we need to define in each flow that the variable ``$transcript`` is global in order to get access to the global instance. Otherwise, it would be a local variable hiding the global instance. But please think twice about using global variables as it can be an indication of a non-optimal Colang design.

----------------------------------------
Expressions in Strings
----------------------------------------

As in Python's formatted string literals we can use braces to evaluate an expression inside a string ``"{$variable}"``:

.. code-block:: colang
    :caption: variables/string_expression_evaluation/main.co

    flow main
        $user_name = "John"
        await UtteranceBotAction(script="Hi {$user_name}!")
        match AnEvent()

If you need to include a brace character in the literal text, it can be escaped by doubling: ``{{`` and ``}}``.

----------------------------------------
Flow Member Variables
----------------------------------------

To access a flow instance's member variables you can use a reference or the reserved variable ``$self`` from within the flow itself:

.. code-block:: colang

    $ref.uid: str # The unique id of the flow instance
    $ref.flow_id: str # The name of the flow
    $ref.status.value: str # Name of the low state ("waiting", "starting", "started", "stopping", "stopped", "finished")
    $ref.loop_id: Optional[str] # The interaction loop id of the flow
    $ref.parent_uid: Optional[str] # The unique id of the parent flow instance
    $ref.child_flow_uids: List[str] # All unique ids of the child flow instances
    $ref.context: dict # The variable context that contains all user defined variables in the flow
    $ref.priority: float # Priority of the flow (range: [0.0-1.0], default: 1.0)
    $ref.arguments: dict # All arguments of the flow
    $ref.flow_instance_uid: str # Flow instance specific uid
    $ref.source_flow_instance_uid: str # The parent flow uid of the flow
    $ref.activate: bool # True if the flow was activated and will therefore restart immediately when finished
    $ref.new_instance_started: bool # True if new instance was started of an activated flow

You should not change those values if you are not sure what you are doing since this can have side effects on the further execution of the flow!


----------------------------------------
Action Member Variables
----------------------------------------

To access the member variables of an action you can use an action reference:

.. code-block:: colang

    $ref.uid: str # The unique id of the action instance
    $ref.name: str # The name of the action
    $ref.flow_uid: str # The flow that started the action
    $ref.status.value: str # The action status ("initialized", "starting", "started", "stopping", "finished")
    $ref.context: dict # Contains all the action event parameters
    $ref.start_event_arguments: dict # Contains all action start arguments

----------------------------------------
Event Member Variables
----------------------------------------

To access the member variables of an event you can use an event reference:

.. code-block:: colang

    $ref.name: str # The name of the event
    $ref.arguments: dict # A dictionary with all the event arguments

    # Only for flow events
    $ref.flow: FlowReference # A reference to the flow of the event

----------------------------------------
System and Bot Configuration Values
----------------------------------------

To access system or bot specific configuration variables you can use ``$system``:

.. code-block:: colang

    $system.config # Current bot configuration object (YAML)
    $system.state # The bots current runtime state object

As an example, if you defined a new boolean value `streaming` in the yaml bot configuration:

.. code-block:: yaml

    streaming: True

you can access and print it like that:

.. code-block:: colang

    print $system.config.streaming


Next we learn how to use :ref:`flow-control` to create branching or looping interaction patterns.
