.. _development-and-debugging:

========================================
Development and Debugging
========================================

.. .. note::
..     Feedbacks & TODOs:

Currently, the Colang story designing and building environment is fairly limited but will be improved on in the future.

-----------------------------------------
Integrated Development Environment (IDE)
-----------------------------------------

- We suggest using `Visual Studio Code <https://code.visualstudio.com/>`_ and the Colang highlighting extension (`Github link <https://github.com/NVIDIA/NeMo-Guardrails/tree/main/vscode_extension>`_) to help with highlighting Colang code.
- You can use the Visual Studio Code launch setting to run a Colang story with nemoguardrails in a python environment by pressing F5 (`launch.json <https://github.com/NVIDIA/NeMo-Guardrails/blob/main/.vscode/launch.json>`_)
- You can show generated and received events by adding the ``--verbose`` flag when starting nemoguardrails, that will also show all generated LLM prompts and responses
- To see even more details and show the internal run-time logs use ``--debug-level=INFO`` or set it equal to ``DEBUG``

-------------------------
Log and Print Statements
-------------------------

To help debugging your Colang flows you can use the print statement ``print <expression>`` to print something to the standard output.

.. code-block:: colang
    :caption: development_debugging/print_statement/main.co

    flow main
        user said something as $flow
        print $flow.transcript

.. code-block:: console

    > Hi

    Hi

Alternatively, use the log statement ``log <expression>`` to append to the logging shown in the verbose mode, which will appear as "Colang Log <flow_instance_uid> :: <expression>".

Furthermore, the Colang function ``flows_info`` can be used to return more information about a flow instance:

.. code-block:: colang
    :caption: development_debugging/flows_info/main.co

    flow main
        await a
        match RestartEvent()

    flow a
        $info = flows_info($self.uid)
        print pretty_str($info)


.. code-block:: json

    {
        "flow_instance_uid": "(a)2c62...",
        "flow_id": "a",
        "loop_id": "(main)8f3d...",
        "status": "starting",
        "flow_hierarchy": [
            "(main)8170..."
        ],
        "active_statement_at_lines": [
            6
        ],
        "meta_tags": []
    }

Where ``pretty_str`` converts the returned dictionary object to a nicely formatted string. If no parameter is provided to the function it will return a dictionary containing all the currently active flow instances.

-------------------------
CLI Debugging Commands
-------------------------

The NeMo Guardrail CLI provides a couple of additional debugging commands that always start with the ``!`` character:

.. code-block:: console

    > !flows
    ┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ ID ┃ Flow Name                                 ┃ Loop (Priority | Type | Id)   ┃ Flow Instances    ┃ Source                                    ┃
    ┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ 1  │ tracking bot talking state                │ 10 │ Named │ 'state_tracking' │ 4d0cb,3ea17,b3b49 │ /colang/v2_x/library/core.co              │
    │ 2  │ tracking user talking state               │ 10 │ Named │ 'state_tracking' │                   │ /colang/v2_x/library/core.co              │
    │ 3  │ tracking visual choice selection state    │ 10 │ Named │ 'state_tracking' │ fa0f8,196ad       │ /colang/v2_x/library/avatars.co           │
    │ 4  │ _bot_say                                  │ 0 │ parent                    │ b6036             │ /colang/v2_x/library/core.co              │
    │ 5  │ _user_said                                │ 0 │ parent                    │ 303c3,565f7,cca80 │ /colang/v2_x/library/core.co              │
    │ 6  │ _user_said_something_unexpected           │ 0 │ parent                    │ 1b5f7,41f02       │ /colang/v2_x/library/core.co              │
    │ 7  │ _user_saying                              │ 0 │ parent                    │ a65e9             │ /colang/v2_x/library/core.co              │
    │ 8  │ automating intent detection               │ 0 │ parent                    │ c7ead             │ /colang/v2_x/library/llm.co               │
    │ 9  │ await_flow_by_name                        │ 0 │ parent                    │ 5f7ef             │ /colang/v2_x/library/core.co              │
    │ 10 │ bot answer question about france          │ 0 │ parent                    │                   │ llm_example_flows.co                      │
    │ 11 │ bot ask                                   │ 0 │ parent                    │                   │ /colang/v2_x/library/core.co              │
    │ 12 │ bot ask how are you                       │ 0 │ parent                    │                   │ demo.co                                   │
    │ 13 │ bot ask user for age                      │ 0 │ parent                    │                   │ llm_example_flows.co                      │
    │ 14 │ bot ask user to pick a color              │ 0 │ parent                    │                   │ llm_example_flows.co                      │
    │ 15 │ bot asked something                       │ 0 │ parent                    │                   │ /colang/v2_x/library/core.co              │
    │ 16 │ bot asks email address                    │ 0 │ parent                    │                   │ show_case_back_channelling_interaction.co │
    │ 17 │ bot asks user how the day was             │ 0 │ parent                    │                   │ show_case_back_channelling_interaction.co │
    │ 18 │ bot attract user                          │ 0 │ parent                    │                   │ llm_example_flows.co                      │
    │ 19 │ bot clarified something                   │ 0 │ parent                    │                   │ /colang/v2_x/library/core.co              │
    │ 20 │ bot clarify                               │ 0 │ parent                    │                   │ /colang/v2_x/library/core.co              │
    │ 21 │ bot count from a number to another number │ 0 │ parent                    │                   │ llm_example_flows.co                      │
    │ 22 │ bot express                               │ 0 │ parent                    │                   │ /colang/v2_x/library/core.co              │
    │ 23 │ bot express feeling bad                   │ 0 │ parent                    │                   │ demo.co                                   │
    │ 24 │ bot express feeling well                  │ 0 │ parent                    │                   │ demo.co                                   │
    │ 25 │ bot express greeting                      │ 0 │ parent                    │                   │ demo.co                                   │
    │ 26 │ bot expressed something                   │ 0 │ parent                    │                   │ /colang/v2_x/library/core.co              │
    │ 27 │ bot gesture                               │ 0 │ parent                    │ a2528             │ /colang/v2_x/library/avatars.co           │
    │ 28 │ bot gesture with delay                    │ 0 │ parent                    │                   │ /colang/v2_x/library/avatars.co           │
    │ 29 │ bot inform                                │ 0 │ parent                    │ da186             │ /colang/v2_x/library/core.co              │
    │ 30 │ bot inform about service                  │ 0 │ parent                    │                   │ demo.co                                   │
    │ 31 │ bot informed something                    │ 0 │ parent                    │                   │ /colang/v2_x/library/core.co              │
    │ 32 │ bot make long pause                       │ 0 │ parent                    │                   │ demo.co                                   │
    │ 33 │ bot make short pause                      │ 0 │ parent                    │                   │ demo.co                                   │
    └────┴───────────────────────────────────────────┴───┴───────────────────────────┴───────────────────┴───────────────────────────────────────────┴

.. code-block:: console

    > !tree
    main
    ├── notification of undefined flow start `Excuse me, what did you say?` (fdb... ,started)
    ├── notification of colang errors `Excuse me, what did you say?` (69c... ,started)
    ├── automating intent detection (c7e... ,started)
    │   ├── logging marked user intent flows (d84... ,started)
    │   └── logging marked bot intent flows (462... ,started)
    ├── showcase selector (742... ,started)
    │   ├── handling bot talking interruption `inform` `stop|cancel` (dea... ,started)
    │   │   └── > user interrupted bot talking `15` `stop|cancel` (109... ,started)
    │   │       └── > bot started saying something (561... ,started)
    │   ├── > user was silent `15.0` (40f... ,started)
    │   │   └── > wait `15.0` `wait_timer_1f499f52-9634-4925-b18a-579fef485d5e` (ae4... ,started)
    │   ├── > user picked number guessing game showcase (6bc... ,started)
    │   │   ├── > user has selected choice `game` (b88... ,started)
    │   │   ├── > user said `I want to play the number guessing game` (448... ,started)
    │   │   │   └── > _user_said `I want to play the number guessing game` (303... ,started)
    │   │   ├── > user said `Show me the game` (d32... ,started)
    │   │   │   └── > _user_said `Show me the game` (565... ,started)
    │   │   ├── > user said `showcase A` (3fd... ,started)
    │   │   │   └── > _user_said `showcase A` (cca... ,started)
    │   │   ├── > user said `First showcase` (013... ,started)
    │   │   │   └── > _user_said `First showcase` (9d4... ,started)
    │   │   └── > user said `re.compile('(?i)guessing game', re.IGNORECASE)` (af9... ,started)
    │   │       └── > _user_said `re.compile('(?i)guessing game', re.IGNORECASE)` (966... ,started)
    │   ├── > user picked multimodality showcase (e2d... ,started)
    │   │   ├── > user has selected choice `multimodality` (f69... ,started)
    │   │   ├── > user said `Show me the multimodality showcase` (9be... ,started)
    │   │   │   └── > _user_said `Show me the multimodality showcase` (33f... ,started)
    │   │   ├── > user said `multimodality` (dfe... ,started)
    │   │   │   └── > _user_said `multimodality` (18f... ,started)
    │   │   ├── > user said `showcase B` (aa6... ,started)
    │   │   │   └── > _user_said `showcase B` (4c5... ,started)
    │   │   ├── > user said `Second showcase` (205... ,started)
    │   │   │   └── > _user_said `Second showcase` (92a... ,started)
    │   │   └── > user said `re.compile('(?i)multimodality', re.IGNORECASE)` (b10... ,started)
    │   │       └── > _user_said `re.compile('(?i)multimodality', re.IGNORECASE)` (d86... ,started)


.. code-block:: colang
    :caption: All CLI debugging commands

    flows [--all] [--order_by_name] # Shows all (active) flows in a table in order of their interaction loop priority and name
    tree # Shows the flow hierarchy tree of all (active) flows. A flow is waiting on a child flow to finish if it's indicated by the character `>`.
    flow [<flow_name>|<flow_instance_uid>] # Show flow or flow instance details
    pause # Pause timer event processing such that interaction does not continue on its own
    resume # Resume timer event processing, including the ones trigger during pause
    restart # Reset interaction and restart the Colang script


-------------
Useful Flows
-------------

In the Colang Standard Library there are a couple of helpful flows that help in the development process of a Colang story:

**notification of undefined flow start**.
Sometimes it is easy to misspell a flow name. By default Colang will just generate a ``StartFlow(flow_id=<name_of_flow>)`` internal event. But if no flow with that name exists, nothing will happen. This can be confusing and difficult to find. To get help with that you can activate the flow ``notification of undefined flow start`` which will detect these cases and print and log a corresponding message and optionally lets the bot respond.

**notification of unexpected user utterance**.
This flow helps you to detect if certain ``UtteranceUserAction.Finished()`` events are not matched to and therefore will create no bot response, which is probably almost never the case.

**notification of colang errors**.
When running a Colang story there can be runtime errors due to invalid attribute access or LLM related parsing errors. This error will generate a ``ColangError(error_type: str, error: str)`` event. Usually nothing will match to this event and the bot will remain silent. While debugging a this can be confusing and the flow ``notification of colang errors`` from the standard library will help make this more transparent by printing and logging a corresponding message and optionally letting the bot respond.
