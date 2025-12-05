.. _whats-changed:

==============
What's Changed
==============

This guide provides a non-comprehensive overview of the most important changes in Colang 2.0.

Terminology
-----------

To limit the learning curve, Colang 2.0 borrows as much as possible the terminology from Python:

- Every bit of Colang code is called a *script*.
- A single ``.co`` file is called a *module*.
- A folder of Colang files, potentially with sub-folders is called a *package*.
- Modules and packages can be imported.

Syntax Changes
--------------

- Dropping the ``define``, ``execute`` keywords.
- Adding the ``flow``, ``match``, ``send``, ``start``, ``await``, ``activate`` keywords.
- Support for decorators.
- ``when``/``or when`` instead of ``when``/``else when``.
- Dropped subflows.

Defining User and Bot Intents
-----------------------------

The special ``define user ...`` and ``define bot ...`` syntax is no longer supported. When defining example utterances, in Colang 2.0 you have to use a more explicit syntax:

.. code-block:: colang

    flow user expressed greeting
      user said "hi" or user said "hello"

.. code-block:: colang

    flow user expressed greeting
      user said "hi"
        or user said "hello"
        or user said "Good evening!"
        or user said "Good afternoon!"

Why? For example, now you can mix other types of events and modalities, e.g., ``user gesture "wave"``.

Similarly, for bot intents:

.. code-block:: colang

    flow bot express greeting
      bot say "Hello world!"
        or bot say "Hi there!"

Flow naming conventions
-----------------------

The flows modeling the events from "outside of the system" are named using the **past tense**, e.g., ``user said``, ``user expressed greeting``, etc. On the bot side, they represent actions that need to be taken and the **imperative form** is used, e.g., ``bot say``, ``bot express greeting``, ``bot refuse to respond``, etc. For more details see :ref:`flow-naming-conventions`.

The Generation Operator
-----------------------

Colang 2.0 introduces the ``...`` operator, a.k.a. the "generation" operator. This can be used whenever a part of a Colang script needs to be generated dynamically, at runtime. Typically, this is done using an LLM.

The ``...`` operator enables the use of *natural language flows* where the docstring of the flow is used to generate the content of the flow.

Active Flows
------------

In Colang 1.0, all the flows are active by default.
In Colang 2.0, flows must be activated explicitly. There is also now a ``main`` flow which is activated by default and is the entry point.

Entry Point
-----------

In Colang 1.0, there was no clear entry point for a Colang script. In Colang 2.0, the ``main`` flow is the entry point. The ``main`` flows triggers the activation of all other flows used in the Colang package.

Import Mechanism
----------------

Colang 2.0 adds an import mechanism similar to python.
Any Colang module or package that is exists in the ``COLANGPATH`` can be imported using the ``import`` statement. Unlike Python, currently, Colang 2.0 only offers module/package level import, i.e., you can't import only a specific flow. This will be added in a future version.

Standard Library
----------------

Colang 2.0 now has a standard library:

- ``core``: a set of core flows related to user and bot utterances, e.g., ``user said``, ``bot say``.
- ``llm``: the flows related to driving the interaction using an LLM.
- ``timing``: timing dependent flows, e.g. ``wait``, ``user was silent $time_s``.
- ``guardrails``: support for adding guardrails, i.e., check the user input, bot output, etc.
- ``avatars``: support for controlling interactive avatars.
- ``utils``: a small set of utility flows.

Asynchronous Actions
--------------------

In Colang 1.0, actions could only be executed synchronously, blocking a flow. Also, there was no way to start two actions in parallel. This was particularly important, for example, if you wanted multiple input rails to run in parallel.

In Colang 2.0, the ``execute`` keyword has been replaced with ``await``, similar to Python. Also, you can use ``start`` to start an action without blocking the flow.

Naming Conventions
------------------

Colang 2.0 uses the following naming conventions:
- Flow names: lower case, can have spaces, should read naturally.
- Action names: camel case, must end with "Action".
- Event names: camel case.

There are certain conventions for the events that mark the start and finish of an action:
``Start...Action``, ``...ActionStarted``, ``...ActionFinished``.

Multi-modal
-----------

Colang 2.0 supports modeling multi-modal interaction not just text-based interaction (e.g., ``user gesture``, ``bot gesture``, ``bot posture``, etc.)

Variables
---------

In Colang 1.0 all variables are global by default. In Colang 2.0, all variables are local by default. To make a variable global, you can use the ``global`` keyword.

There are no default global variables in Colang 2.0.

String formatting
-----------------

The inline ``"Hello there, $name!"`` is no longer supported. You must always wrap variables within curly braces, similar to python ``"Hello there, {$name}!"``.

LLM invocation
--------------

In Colang 1.0, as soon as you defined a user intent, the dialog rails would be automatically activated and the LLM would be used. In Colang 2.0, to use the LLM, you have to activate the mechanism explicitly:

.. code-block:: colang

  flow main
    activate llm continuation

Python API
----------

Colang 2.0 adds support for an explicit "state object". For interactions that span multiple turns/events, a state object is returned after each processing and needs to be passed back on the next processing cycle.


.. _whats-changed-alpha-to-beta:

Breaking changes from alpha to beta version
--------------------------------------------

* Metatags
    * ``# meta: user intent`` -> ``@meta(user_intent=True)`` (also ``user_action``, ``bot_intent``, ``bot_action``)
    * ``# meta: exclude from llm`` -> ``@meta(exclude_from_llm=True)``

* Interaction loop id
    * ``# meta: loop_id=<loop_id>`` -> ``@loop("<loop_id>")``
* Or when statement:
    * ``orwhen`` -> ``or when``
* NLD instructions
    * ``"""<NLD>"""`` -> ``..."<NLD>"``
* Internal event parameter renaming:
    * ``flow_start_uid`` -> ``flow_instance_uid``
* Regular expression
    * ``r"<regex>"`` -> ``regex("<regex>")``
* Expressions in strings
    * ``"{{<expression>}}"`` -> ``"{<expression>}"``
* Colang function name changes
    * ``findall`` -> ``find_all``
* Bot specific copies of the Colang Core Library
    * ccl_*.co files are deprecated and should be removed from the bot folders. It is replaced by the Colang Standard Libraries that are included in NeMo Guardrails and can be imported (e.g. ``import core`` or ``import llm`` ). See next the new name mapping of standard library flows.
* Standard library flow name changes
    * ``catch colang errors`` -> ``notification of colang errors`` (core.co)
    * ``catch undefined flows`` -> ``notification of undefined flow start`` (core.co)
    * ``catch unexpected user utterance`` -> ``notification of unexpected user utterance`` (core.co)
    * ``poll llm request response`` -> ``polling llm request response`` (llm.co)
    * ``trigger user intent for unhandled user utterance`` -> ``generating user intent for unhandled user utterance`` (llm.co)
    * ``generate then continue interaction`` -> ``llm continue interaction`` (llm.co)
    * ``track bot talking state`` -> ``tracking bot talking state`` (core.co)
    * ``track user talking state`` -> ``tracking user talking state`` (core.co)
    * ``track unhandled user intent state`` -> ``tracking unhandled user intent state`` (llm.co)
    * ``track visual choice selection state`` -> ``track visual choice selection state`` (avatars.co)
    * ``track user utterance state`` -> ``tracking user talking state`` (core.co)
    * ``track bot utterance state`` -> ``tracking bot talking state`` (core.co)
    * ``interruption handling bot talking`` -> ``handling bot talking interruption`` (avatars.co)
    * ``generate then continue interaction`` -> ``llm continue interaction`` (llm.co)
    * ``respond to unhandled user intent`` -> ``continuation on unhandled user intent`` (llm.co)
    * ``manage listening posture`` -> ``managing listening posture`` (avatars.co)
    * ``manage talking posture`` -> ``managing talking posture`` (avatars.co)
    * ``manage thinking posture`` -> ``managing thinking posture`` (avatars.co)
    * ``manage attentive posture`` -> No replacement (copy to your bot script if needed)
    * ``manage bot postures`` -> ``managing bot postures`` (avatars.co)
    * ``track user presence state`` -> No replacement (copy to your bot script if needed)
    * ``user became no longer present`` -> No replacement (copy to your bot script if needed)
