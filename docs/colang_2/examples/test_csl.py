import pathlib
import sys

import pytest

pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))
print(sys.path)

from utils import compare_interaction_with_script

########################################################################################################################
# CORE
########################################################################################################################

## User event flows


@pytest.mark.asyncio
async def test_user_said():
    colang_code = """
# COLANG_START: test_user_said
import core

flow main
    # Only matches exactly "hello"
    user said "hello"
    bot say "hi"
# COLANG_END: test_user_said
    """

    test_script = """
# USAGE_START: test_user_said
> hi
> hello
hi
# USAGE_END: test_user_said
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_user_said_something():
    colang_code = """
# COLANG_START: test_user_said_something
import core

flow main
    $transcript = await user said something
    bot say "You said: {$transcript}"
# COLANG_END: test_user_said_something
    """

    test_script = """
# USAGE_START: test_user_said_something
> I can say whatever I want
You said: I can say whatever I want
# USAGE_END: test_user_said_something
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_user_saying():
    colang_code = """
# COLANG_START: test_user_saying
import core

flow main
    # Provide verbal feedback while the user is writing / speaking
    while True
        when user saying (regex("sad|lonely"))
            bot say "oooh"
        or when user saying (regex("great|awesome"))
            bot say "nice!"
# COLANG_END: test_user_saying
    """

    test_script = """
# USAGE_START: test_user_saying
> /UtteranceUserAction.TranscriptUpdated(interim_transcript="this is a ")
> /UtteranceUserAction.TranscriptUpdated(interim_transcript="this is a sad story")
oooh
# USAGE_END: test_user_saying
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_user_saying_something():
    colang_code = """
# COLANG_START: test_user_saying_something
import core
import avatars

flow main
    user saying something
    bot gesture "nod"
# COLANG_END: test_user_saying_something
    """

    test_script = """
# USAGE_START: test_user_saying_something
> /UtteranceUserAction.TranscriptUpdated(interim_transcript="anything")
Gesture: nod
# USAGE_END: test_user_saying_something
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_user_started_saying_something():
    colang_code = """
# COLANG_START: test_user_started_saying_something
import core
import avatars

flow main
    # Start a bot posture as soon as the user starts talking
    user started saying something
    start bot posture "listening" as $ref

    # Stop the posture when the user is done talking
    user said something
    send $ref.Stop()
# COLANG_END: test_user_started_saying_something
    """

    test_script = """
# USAGE_START: test_user_started_saying_something
> /UtteranceUserAction.Started()
Posture: listening
> /UtteranceUserAction.TranscriptUpdated(interim_transcript="I am starting to talk")
> /UtteranceUserAction.Finished(final_transcript="anything")
bot posture (stop)
# USAGE_END: test_user_started_saying_something
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_user_said_something_unexpected():
    colang_code = """
# COLANG_START: test_user_said_something_unexpected
import core

flow handling welcome
    user said "hi" or user said "hello"
    bot say "hello"

flow main
    activate handling welcome

    # If the user utterance is anything else except "hi" and "hello" this will advance
    user said something unexpected
    bot say "you said something unexpected"
# COLANG_END: test_user_said_something_unexpected
    """

    test_script = """
# USAGE_START: test_user_said_something_unexpected
> hi
hello
> how are you
you said something unexpected
# USAGE_END: test_user_said_something_unexpected
        """

    await compare_interaction_with_script(test_script, colang_code)


# Bot Action Flows
@pytest.mark.asyncio
async def test_bot_say():
    colang_code = """
# COLANG_START: test_bot_say
import core

flow main
    user said something
    bot say "Hello world!"
# COLANG_END: test_bot_say
    """

    test_script = """
# USAGE_START: test_bot_say
> anything
Hello world!
# USAGE_END: test_bot_say
        """

    await compare_interaction_with_script(test_script, colang_code)


# Bot Event Flows
@pytest.mark.asyncio
async def test_bot_started_saying_example():
    colang_code = """
# COLANG_START: test_bot_started_saying_example
import core

flow reacting to bot utterances
    bot started saying "hi"
    send CustomEvent()

flow main
    activate reacting to bot utterances

    user said something
    bot say "hi"
# COLANG_END: test_bot_started_saying_example
    """

    test_script = """
# USAGE_START: test_bot_started_saying_example
> hello
hi
Event: CustomEvent
# USAGE_END: test_bot_started_saying_example
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_bot_started_saying_something():
    colang_code = """
# COLANG_START: test_bot_started_saying_something
import core
import avatars

flow handling talking posture
    bot started saying something
    bot posture "talking"
    bot said something

flow main
    activate handling talking posture

    user said something
    bot say "hi"
# COLANG_END: test_bot_started_saying_something
    """

    test_script = """
# USAGE_START: test_bot_started_saying_something
> something
hi
Posture: talking
bot posture (stop)
# USAGE_END: test_bot_started_saying_something
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_bot_said():
    colang_code = """
# COLANG_START: test_bot_said
import core
import avatars

flow creating gestures
    when bot said "yes"
        bot gesture "thumbs up"
    or when bot said "no"
        bot gesture "shake head"

flow answering cat dog questions
    when user said "Do you like cats?"
        bot say "yes"
    or when user said "Do you like dogs?"
        bot say "no"

flow main
    activate creating gestures
    activate answering cat dog questions

    wait indefinitely

# COLANG_END: test_bot_said
    """

    test_script = """
# USAGE_START: test_bot_said
> Do you like cats?
yes
Gesture: thumbs up
> Do you like dogs?
no
Gesture: shake head
# USAGE_END: test_bot_said
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_tracking_bot_talking_state():
    colang_code = """
# COLANG_START: test_tracking_bot_talking_state
import core

flow main
    global $bot_talking_state
    activate tracking bot talking state

    user said something
    if $bot_talking_state
        bot gesture "show ignorance to user speech"
    else
        bot say "responding to user question"

# COLANG_END: test_tracking_bot_talking_state
    """

    test_script = """
# USAGE_START: test_tracking_bot_talking_state
> hello there
responding to user question
# USAGE_END: test_tracking_bot_talking_state
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_tracking_user_talking_state():
    colang_code = """
# COLANG_START: test_tracking_user_talking_state
import core

flow main
    global $last_user_transcript
    activate tracking user talking state

    user said something
    bot say "I remembered {$last_user_transcript}"

# COLANG_END: test_tracking_user_talking_state
    """

    test_script = """
# USAGE_START: test_tracking_user_talking_state
> my favorite color is red
I remembered my favorite color is red
# USAGE_END: test_tracking_user_talking_state
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_notification_of_colang_errors():
    colang_code = """
# COLANG_START: test_notification_of_colang_errors
import core

# We need to create an artificial error.
# We need to create this in a separate flow as otherwise the main flow will fail upon the error.
flow creating an error
    user said something
    $number = 3
    print $number.error

flow main
    activate notification of colang errors

    creating an error
    wait indefinitely


# COLANG_END: test_notification_of_colang_errors
    """

    test_script = """
# USAGE_START: test_notification_of_colang_errors
> test
Excuse me, there was an internal Colang error.
# USAGE_END: test_notification_of_colang_errors
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_notification_of_undefined_flow_start():
    colang_code = """
# COLANG_START: test_notification_of_undefined_flow_start
import core

flow main
    activate notification of undefined flow start

    # We are misspelling the `bot say` flow to trigger a undefined flow start.
    user said something
    bot sayy "hello"

# COLANG_END: test_notification_of_undefined_flow_start
    """

    test_script = """
# USAGE_START: test_notification_of_undefined_flow_start
> test
Failed to start an undefined flow!
# USAGE_END: test_notification_of_undefined_flow_start
        """

    await compare_interaction_with_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_notification_of_unexpected_user_utterance():
    colang_code = """
# COLANG_START: test_notification_of_unexpected_user_utterance
import core

flow reacting to user requests
    user said "hi" or user said "hello"
    bot say "hi there"

flow main
    activate notification of unexpected user utterance
    activate reacting to user requests

# COLANG_END: test_notification_of_unexpected_user_utterance
    """

    test_script = """
# USAGE_START: test_notification_of_unexpected_user_utterance
> hello
hi there
> what is your name
I don't know how to respond to that!
# USAGE_END: test_notification_of_unexpected_user_utterance
        """

    await compare_interaction_with_script(test_script, colang_code)
