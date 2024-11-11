---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Interactive Avatar Modality Flows (`avatars.co <../../../nemoguardrails/colang/v2_x/library/avatars.co>`_)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**User Event Flows**

.. code-block:: colang

    # Wait for a UI selection
    flow user selected choice $choice_id -> $choice

    # Wait for a UI selection to have happened (considering also choices that happened right before)
    flow user has selected choice $choice_id

    # Wait for user entering keystrokes in UI text field
    flow user typing $text -> $inputs

    # Wait for user to make a gesture
    flow user gestured $gesture -> $final_gesture

    # Wait for user to be detected as present (e.g. camera ROI)
    flow user became present -> $user_id

    # Wait for when the user talked while bot is speaking
    flow user interrupted bot talking $sentence_length=15


**Bot Action Flows**

.. code-block:: colang

    # Trigger a specific bot gesture
    flow bot gesture $gesture

    # Trigger a specific bot gesture delayed
    flow bot gesture with delay $gesture $delay

    # Trigger a specific bot posture
    flow bot posture $posture

    # Show a 2D UI with some options to select from
    flow scene show choice $prompt $options

    # Show a 2D UI with detailed information
    flow scene show textual information $title $text $header_image

    # Show a 2D UI with a short information
    flow scene show short information $info

    # Show a 2D UI with some input fields to be filled in
    flow scene show form $prompt $inputs

**Bot Event Flows**

.. code-block:: colang

    # Wait for the bot to start with the given gesture
    flow bot started gesture $gesture

    # Wait for the bot to start with any gesture
    flow bot started a gesture -> $gesture

    # Wait for the bot to start with the given posture
    flow bot started posture $posture

    # Wait for the bot to start with any posture
    flow bot started a posture -> $posture

    # Wait for the bot to start with any action
    flow bot started an action -> $action

**State Tracking Flows**

These are flows that track bot and user states in global variables.

.. code-block:: colang

    # Track most recent visual choice selection state in global variable $choice_selection_state
    flow tracking visual choice selection state

**Helper & Utility Flows**

These are some useful helper and utility flows:

.. code-block:: colang

    # Stops all the current bot actions
    flow finish all bot actions

    # Stops all the current scene actions
    flow finish all scene actions

    # Handling the bot talking interruption reaction
    flow handling bot talking interruption $mode="inform"

**Posture Management Flows**

.. code-block:: colang

    # Activates all the posture management
    flow managing bot postures

    # Start and stop listening posture
    flow managing listening posture

    # Start and stop talking posture
    flow managing talking posture

    # Start and stop thinking posture
    flow managing thinking posture

    # Start and stop idle posture
    flow managing idle posture
