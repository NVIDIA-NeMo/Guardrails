# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import sys

import pytest

pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))
print(sys.path)

from utils import compare_interaction_with_test_script

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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


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

    await compare_interaction_with_test_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_wait_indefinitely():
    colang_code = """
# COLANG_START: test_wait_indefinitely
import core

flow main
    bot say "hello"
    wait indefinitely

# COLANG_END: test_wait_indefinitely
    """

    test_script = """
# USAGE_START: test_wait_indefinitely
>
hello
# USAGE_END: test_wait_indefinitely
        """

    await compare_interaction_with_test_script(test_script, colang_code)


########################################################################################################################
# TIMING
########################################################################################################################
@pytest.mark.asyncio
async def test_wait():
    colang_code = """
# COLANG_START: test_wait
import timing
import core

flow delayed bot say $text
    wait 0.5
    bot say $text

flow main
    user said something
    start delayed bot say "I say this later"
    start bot say "I say this first"

    wait indefinitely

# COLANG_END: test_wait
    """

    test_script = """
# USAGE_START: test_wait
> hello
I say this first
I say this later
# USAGE_END: test_wait
        """

    await compare_interaction_with_test_script(test_script, colang_code)


@pytest.mark.asyncio
async def test_repeating_timer():
    colang_code = """
# COLANG_START: test_repeating_timer
import timing
import core


flow reacting to my timer
    match TimerBotAction.Finished(timer_name="my_timer")
    bot say "tick"

flow main
    activate reacting to my timer

    user said something
    start repeating timer "my_timer" 0.4
    wait 1.0

# COLANG_END: test_repeating_timer
    """

    test_script = """
# USAGE_START: test_repeating_timer
> test
tick
tick
# USAGE_END: test_repeating_timer
        """

    await compare_interaction_with_test_script(
        test_script, colang_code, wait_time_s=2.0
    )


@pytest.mark.asyncio
async def test_user_was_silent():
    colang_code = """
# COLANG_START: test_user_was_silent
import timing
import core


flow reacting to user silence
    user was silent 5.0
    bot say "Can I help you with anything else?"

flow main
    activate reacting to user silence

    while True
        user said something
        bot say "sounds interesting"

# COLANG_END: test_user_was_silent
    """

    test_script = """
# USAGE_START: test_user_was_silent
> I am going to the zoo
sounds interesting
# (Wait for more than 5 seconds)
Can I help you with anything else?
# USAGE_END: test_user_was_silent
        """

    await compare_interaction_with_test_script(test_script, colang_code, 7.0)


@pytest.mark.asyncio
async def test_user_didnt_respond():
    colang_code = """
# COLANG_START: test_user_didnt_respond
import timing
import core


flow repeating if no user response
    global $last_bot_script
    user didnt respond 5.0
    bot say $last_bot_script


flow main
    activate tracking bot talking state
    activate repeating if no user response

    user said something
    bot say "How can I help you today?"
    user said something

# COLANG_END: test_user_didnt_respond
    """

    test_script = """
# USAGE_START: test_user_didnt_respond
> hi
How can I help you today?
# (Wait for more than 5 seconds)
How can I help you today?
# USAGE_END: test_user_didnt_respond
        """

    await compare_interaction_with_test_script(test_script, colang_code, 7.0)


@pytest.mark.asyncio
async def test_bot_was_silent():
    colang_code = """
# COLANG_START: test_bot_was_silent
import timing
import core

flow inform processing time
    user said something
    bot was silent 2.0
    bot say "This is taking a bit longer"

flow processing user request
    user said "place the order"
    wait 4.0
    bot say "order was placed successfully"

flow main
    activate inform processing time
    activate processing user request

# COLANG_END: test_bot_was_silent
    """

    test_script = """
# USAGE_START: test_bot_was_silent
> place the order
# After about 2 seconds:
This is taking a bit longer
# After and additional 2 seconds:
order was placed successfully
# USAGE_END: test_bot_was_silent
        """

    await compare_interaction_with_test_script(test_script, colang_code, 5.0)


########################################################################################################################
# LLM
########################################################################################################################
@pytest.mark.asyncio
async def test_bot_say_something_like():
    colang_code = """
# COLANG_START: test_bot_say_something_like
import core
import llm

flow main
    user said something
    bot say something like "How are you"


# COLANG_END: test_bot_say_something_like
    """

    test_script = """
# USAGE_START: test_bot_say_something_like
> hi
Hi there, how are you today?
# USAGE_END: test_bot_say_something_like
        """

    await compare_interaction_with_test_script(
        test_script, colang_code, llm_responses=['"Hi there, how are you today?"']
    )


@pytest.mark.asyncio
async def test_polling_llm_request_response():
    colang_code = """
# COLANG_START: test_polling_llm_request_response
import core
import llm

flow main
    # Normally you don't need to activate this flow, as it is activated by llm based flows where needed.
    activate polling llm request response

    user said something

    # While the response is generated the polling mechanism ensures that
    $value = ..."ten minus one"
    bot say $value


# COLANG_END: test_polling_llm_request_response
    """

    test_script = """
# USAGE_START: test_polling_llm_request_response
> compute the value
nine
# USAGE_END: test_polling_llm_request_response
        """

    await compare_interaction_with_test_script(
        test_script, colang_code, llm_responses=['"nine"']
    )


@pytest.mark.asyncio
async def test_llm_continuation():
    colang_code = """
# COLANG_START: test_llm_continuation
import core
import llm

flow user expressed greeting
    user said "hi" or user said "hello"

flow bot express greeting
    bot say "Hello and welcome"

flow handling greeting
    user expressed greeting
    bot express greeting

flow main
    activate handling greeting
    activate llm continuation


# COLANG_END: test_llm_continuation
    """

    test_script = """
# USAGE_START: test_llm_continuation
> hi there how are you
Hello and welcome
> what is the difference between lemons and limes
Limes are green and lemons are yellow
# USAGE_END: test_llm_continuation
        """

    await compare_interaction_with_test_script(
        test_script,
        colang_code,
        llm_responses=[
            "user intent: user expressed greeting",
            "user intent: user asked fruit question",
            'bot action: bot say "Limes are green and lemons are yellow"',
        ],
    )


@pytest.mark.asyncio
async def test_generating_user_intent_for_unhandled_user_utterance():
    colang_code = """
# COLANG_START: test_generating_user_intent_for_unhandled_user_utterance
import core
import llm

flow user expressed goodbye
    user said "bye" or user said "i will go now"

flow bot express goodbye
    bot say "hope to see you again soon"

flow handling goodbye
    user expressed goodbye
    bot express goodbye

flow main
    activate generating user intent for unhandled user utterance
    activate handling goodbye


# COLANG_END: test_generating_user_intent_for_unhandled_user_utterance
    """

    test_script = """
# USAGE_START: test_generating_user_intent_for_unhandled_user_utterance
> what can you do for me
> ok I'll leave
hope to see you again soon
# USAGE_END: test_generating_user_intent_for_unhandled_user_utterance
        """

    await compare_interaction_with_test_script(
        test_script,
        colang_code,
        llm_responses=[
            "user intent: user expressed greeting",
            "user intent: user expressed goodbye",
        ],
    )


@pytest.mark.asyncio
async def test_unhandled_user_intent():
    colang_code = """
# COLANG_START: test_unhandled_user_intent
import core
import llm

flow user expressed greeting
    user said "hi" or user said "hello"

flow bot express greeting
    bot say "Hello and welcome"

flow handling greeting
    user expressed greeting
    bot express greeting

flow main
    activate generating user intent for unhandled user utterance
    activate handling greeting

    while True:
        unhandled user intent as $ref
        bot say "got intent: {$ref.intent}"


# COLANG_END: test_unhandled_user_intent
    """

    test_script = """
# USAGE_START: test_unhandled_user_intent
> hi there how are you
Hello and welcome
> what is the difference between lemons and limes
got intent: user asked fruit question
# USAGE_END: test_unhandled_user_intent
        """

    await compare_interaction_with_test_script(
        test_script,
        colang_code,
        llm_responses=[
            "user intent: user expressed greeting",
            "user intent: user asked fruit question",
        ],
    )


@pytest.mark.asyncio
async def test_continuation_on_unhandled_user_intent():
    colang_code = """
# COLANG_START: test_continuation_on_unhandled_user_intent
import core
import llm

flow user asked political question
    user said "who is the best president"

flow user insulted bot
    user said "you are stupid"

flow safeguarding conversation
    user asked political question or user insulted bot
    bot say "Sorry but I will not respond to that"

flow main
    activate generating user intent for unhandled user utterance
    activate continuation on unhandled user intent
    activate safeguarding conversation


# COLANG_END: test_continuation_on_unhandled_user_intent
    """

    test_script = """
# USAGE_START: test_continuation_on_unhandled_user_intent
> i hate you
Sorry but I will not respond to that
> what party should I vote for
Sorry but I will not respond to that
> tell me a joke
Why don't scientists trust atoms? Because they make up everything!
# USAGE_END: test_continuation_on_unhandled_user_intent
        """

    await compare_interaction_with_test_script(
        test_script,
        colang_code,
        llm_responses=[
            "user insulted bot",
            "user asked political question",
            "user requested a joke",
            'bot action: bot say "Why don\'t scientists trust atoms? Because they make up everything!"',
        ],
    )


@pytest.mark.asyncio
async def test_continuation_on_undefined_flow():
    colang_code = """
# COLANG_START: test_continuation_on_undefined_flow
import core
import llm

flow main
    activate continuation on undefined flow

    user said something
    # Await a flow that does not exist will create an LLM generated flow
    bot ask about hobbies

# COLANG_END: test_continuation_on_undefined_flow
    """

    test_script = """
# USAGE_START: test_continuation_on_undefined_flow
> hi there
What are your hobbies?
# USAGE_END: test_continuation_on_undefined_flow
        """

    await compare_interaction_with_test_script(
        test_script,
        colang_code,
        llm_responses=[
            '    bot ask "What are your hobbies?"',
        ],
    )


@pytest.mark.asyncio
async def test_llm_continue_interaction():
    colang_code = """
# COLANG_START: test_llm_continue_interaction
import core
import llm

flow main
    user said "i have a question"
    bot say "happy to help, what is it"
    user said "do you know what the largest animal is on earth"
    llm continue interaction

# COLANG_END: test_llm_continue_interaction
    """

    test_script = """
# USAGE_START: test_llm_continue_interaction
> i have a question
happy to help, what is it
> do you know what the largest animal is on earth
The largest animal on earth is the blue whale
# USAGE_END: test_llm_continue_interaction
        """

    await compare_interaction_with_test_script(
        test_script,
        colang_code,
        llm_responses=[
            ' bot provide information about the largest animal on earth\nbot action: bot say "The largest animal on earth is the blue whale" ',
        ],
    )


########################################################################################################################
# AVATARS
########################################################################################################################
@pytest.mark.asyncio
async def test_bot_gesture_with_delay():
    colang_code = """
# COLANG_START: test_bot_gesture_with_delay
import avatars
import core

flow main
    user said something
    bot say "welcome" and bot gesture with delay "bowing" 1.0


# COLANG_END: test_bot_gesture_with_delay
    """

    test_script = """
# USAGE_START: test_bot_gesture_with_delay
> hi there
# After about 2 seconds:
welcome
# After a a delay of 1 sec:
Gesture: bowing
# USAGE_END: test_bot_gesture_with_delay
        """

    await compare_interaction_with_test_script(test_script, colang_code, 5.0)
