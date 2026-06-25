# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Behavioral equivalence gate for the ``LLMGenerationActions`` decomposition.

This is the Phase 0 characterization harness. It pins the
observable behavior of ``generate_user_intent``, ``generate_next_steps``,
``generate_bot_message``, ``generate_value`` and ``generate_intent_steps_message``
*before* the refactor so that every later phase can be proven behavior
preserving: the same assertions must stay green after each phase.

An event-only oracle is insufficient because three things the refactor moves are
not in the event stream, so this harness captures four channels per scenario:

1. Emitted event list -- the ordered, generation-relevant projection of
   ``GenerationLog.internal_events`` (``event_sequence``). The ``flow_id`` of
   ``start_flow`` (a random uuid) is normalized away by projecting only the flow
   body; the ``passthrough_output`` ``ContextUpdate`` *event* and the
   ``skip_output_rails`` / ``_last_bot_prompt`` context updates appear here in
   order, so the "event form vs dict form" distinction is captured by position.
2. Per-call task -- ``GenerationLog.llm_calls[i].task`` (``llm_tasks``), e.g. the
   known oddity that the passthrough bot-message branch sets
   ``generate_bot_message`` LLMCallInfo but parses as ``general`` shows up as a
   ``general`` task in that path.
3. The ``stop`` argument and whether the call streamed -- captured by
   ``RecordingFakeLLM`` because ``LLMCallInfo`` has no ``stop`` field and
   ``FakeLLMModel`` discards ``stop``, so a dropped ``stop=["User:"]`` is
   invisible to every other channel. This is the only channel that distinguishes
   the four ``Task.GENERAL`` call sites the Phase 1 helper unifies.
4. Streaming chunk sequence -- collected from ``stream_async`` for the streaming
   scenarios (Phase 4).
"""

import asyncio
import hashlib
import os
from typing import cast
from unittest.mock import MagicMock

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from tests.utils import TestChat

LOG_OPTS = {"log": {"llm_calls": True, "internal_events": True}}


class RecordingFakeLLM(FakeLLMModel):
    """A ``FakeLLMModel`` that records the ``stop`` argument and call mode.

    ``LLMCallInfo`` does not carry ``stop`` and the base fake discards it, so this
    is the only place a dropped/added ``stop`` (the Phase 1 divergence axis) is
    observable. ``calls`` is a list of ``(mode, stop)`` where ``mode`` is
    ``"generate"`` or ``"stream"`` (i.e. whether a streaming handler was passed
    into ``llm_call``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    async def generate_async(self, prompt, *, stop=None, **kwargs):
        self.calls.append(("generate", tuple(stop) if stop else None))
        return await super().generate_async(prompt, stop=stop, **kwargs)

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        self.calls.append(("stream", tuple(stop) if stop else None))
        async for chunk in super().stream_async(prompt, stop=stop, **kwargs):
            yield chunk


# Context-update keys emitted by the generation actions that the oracle tracks.
_GENERATION_CONTEXT_KEYS = [
    "skip_output_rails",
    "passthrough_output",
    "bot_thinking",
    "_last_bot_prompt",
]


def event_sequence(response):
    """Project ``internal_events`` to the ordered, generation-relevant events.

    Volatile fields (uids, timestamps) are dropped; the ``start_flow`` flow_id
    (a random uuid) is normalized away by keeping only the flow body. The large
    ``_last_bot_prompt`` value is reduced to ``<present>`` -- its content is
    pinned via the prompt-fingerprint channel where it matters.
    """
    sequence = []
    for event in response.log.internal_events or []:
        event_type = event.get("type")
        if event_type == "UserIntent":
            tag = f"UserIntent:{event['intent']}"
            if event.get("additional_info"):
                cache_keys = ",".join(sorted(event["additional_info"].keys()))
                tag += f"|cache:{cache_keys}"
            sequence.append(tag)
        elif event_type == "BotIntent":
            sequence.append(f"BotIntent:{event['intent']}")
        elif event_type == "BotMessage":
            sequence.append(f"BotMessage:{event['text']}")
        elif event_type == "BotThinking":
            sequence.append(f"BotThinking:{event['content']}")
        elif event_type == "BotToolCalls":
            sequence.append(f"BotToolCalls:{len(event.get('tool_calls') or [])}")
        elif event_type == "start_flow":
            sequence.append(f"start_flow:{event.get('flow_body')!r}")
        elif event_type == "ContextUpdate":
            data = event.get("data") or {}
            for key in _GENERATION_CONTEXT_KEYS:
                if key in data:
                    value = "<present>" if key == "_last_bot_prompt" else data[key]
                    sequence.append(f"ctx:{key}={value}")
    return sequence


def llm_tasks(response):
    """The ordered list of ``LLMCallInfo.task`` values for the generation."""
    return [call.task for call in (response.log.llm_calls or [])]


def prompt_fingerprints(response):
    """Stable short fingerprints of each LLM call prompt.

    Used only for the ``Task.GENERAL`` call sites the Phase 1 helper unifies, to
    catch render drift (e.g. a changed ``relevant_chunks`` render context) that
    would otherwise be invisible. A mismatch means the prompt string changed;
    regenerate by inspecting ``response.log.llm_calls[i].prompt``.
    """
    return [hashlib.sha256((call.prompt or "").encode()).hexdigest()[:10] for call in (response.log.llm_calls or [])]


def generate(config, completions, message):
    """Run a single-turn generation, returning ``(response, recording_llm)``."""
    recording_llm = RecordingFakeLLM(responses=completions)
    chat = TestChat(config, llm=recording_llm)
    response = cast(GenerationResponse, chat.app.generate(message, options=LOG_OPTS))
    return response, recording_llm


# ---------------------------------------------------------------------------
# Dialog mode (multi-call): the classic three-phase pipeline.
# ---------------------------------------------------------------------------


def test_dialog_three_phase():
    """User intent without a flow -> next-step prediction -> bot-message LLM."""
    config = RailsConfig.from_content(
        """
        define user ask booking
            "I want to book a flight"
        """
    )
    response, llm = generate(
        config,
        ["  ask booking", "bot respond booking", '  "Sure, I can help with booking."'],
        "I want to book a flight",
    )

    assert response.response == "Sure, I can help with booking."
    assert event_sequence(response) == [
        "UserIntent:ask booking",
        "BotIntent:respond booking",
        "ctx:_last_bot_prompt=<present>",
        "BotMessage:Sure, I can help with booking.",
    ]
    assert llm_tasks(response) == [
        "generate_user_intent",
        "generate_next_steps",
        "generate_bot_message",
    ]
    assert llm.calls == [("generate", None), ("generate", None), ("generate", None)]


def test_dialog_predefined_bot_message():
    """A predefined bot message short-circuits the phase-3 LLM call."""
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"
        define flow
            user express greeting
            bot express greeting
        define bot express greeting
            "Hello there!"
        """
    )
    response, llm = generate(config, ["  express greeting"], "hello there!")

    assert response.response == "Hello there!"
    assert event_sequence(response) == [
        "UserIntent:express greeting",
        "BotIntent:express greeting",
        "ctx:skip_output_rails=True",
        "BotMessage:Hello there!",
        "ctx:skip_output_rails=False",
    ]
    assert llm_tasks(response) == ["generate_user_intent"]
    assert llm.calls == [("generate", None)]


def test_dialog_context_var_bot_intent():
    """A ``bot $var`` intent renders the bot message from a context variable.

    Exercises the ``generate_bot_message`` branch where the bot intent starts
    with ``$`` and names a context variable; no phase-3 LLM call is made.
    """
    config = RailsConfig.from_content(
        """
        define user give name
            "my name is X"
        define flow
            user give name
            $name = "World"
            bot $name
        """
    )
    response, llm = generate(config, ["  give name"], "my name is World")

    assert response.response == "World"
    assert event_sequence(response) == [
        "UserIntent:give name",
        "BotIntent:$name",
        "BotMessage:World",
    ]
    assert llm_tasks(response) == ["generate_user_intent"]
    assert llm.calls == [("generate", None)]


def test_dialog_multi_step_generation():
    """``enable_multi_step_generation`` emits a ``start_flow`` with a parsed flow.

    The phase-2 LLM returns a multi-line flow; the runtime starts it (random
    ``flow_id`` normalized away) and runs two bot intents/messages from it.
    """
    config = RailsConfig.from_path(os.path.join(os.path.dirname(__file__), "test_configs", "multi_step_generation"))
    llm = RecordingFakeLLM(
        responses=[
            "  express greeting",
            "  request appointment",
            '  "What\'s your name?"',
            "  provide date",
            "bot acknowledge the date\nbot confirm appointment",
            '  "Ok, an appointment for tomorrow."',
            '  "Your appointment is now confirmed."',
        ]
    )
    chat = TestChat(config, llm=llm)
    # The completions are consumed across turns, so drive the first two turns to
    # advance state before capturing the multi-step turn (the third).
    chat.app.generate("hi")
    chat.app.generate(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hey there!"},
            {"role": "user", "content": "i need to make an appointment"},
        ]
    )
    response = cast(
        GenerationResponse,
        chat.app.generate(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hey there!"},
                {"role": "user", "content": "i need an appointment"},
                {"role": "assistant", "content": "What's your name?"},
                {"role": "user", "content": "I want to come tomorrow"},
            ],
            options=LOG_OPTS,
        ),
    )

    assert event_sequence(response) == [
        "UserIntent:provide date",
        "start_flow:'bot acknowledge the date\\nbot confirm appointment'",
        "BotIntent:acknowledge the date",
        "ctx:_last_bot_prompt=<present>",
        "BotMessage:Ok, an appointment for tomorrow.",
        "BotIntent:confirm appointment",
        "ctx:_last_bot_prompt=<present>",
        "BotMessage:Your appointment is now confirmed.",
    ]
    assert llm_tasks(response) == [
        "generate_user_intent",
        "generate_next_steps",
        "generate_bot_message",
        "generate_bot_message",
    ]


# ---------------------------------------------------------------------------
# Dialog mode with embeddings_only intent detection.
# ---------------------------------------------------------------------------

_EMBEDDINGS_COLANG = """
define user express greeting
    "hi"
define bot express greeting
    "Hello!"
define flow
    user express greeting
    bot express greeting
"""


def _embeddings_config(threshold, fallback_intent="express greeting"):
    config = RailsConfig.from_content(
        _EMBEDDINGS_COLANG,
        f"""
        rails:
            dialog:
                user_messages:
                    embeddings_only: True
                    embeddings_only_similarity_threshold: {threshold}
                    embeddings_only_fallback_intent: {fallback_intent!r}
        """,
    )
    return config


def test_embeddings_only_index_hit():
    """A similarity hit returns the matched intent with zero LLM calls."""
    # Threshold 0.0 guarantees the index search returns a hit.
    config = _embeddings_config(threshold=0.0)
    response, llm = generate(config, [], "hi")

    assert response.response == "Hello!"
    assert "UserIntent:express greeting" in event_sequence(response)
    assert llm_tasks(response) == []
    assert llm.calls == []


def test_embeddings_only_threshold_miss_uses_fallback_intent():
    """A below-threshold search falls back to the configured intent, no LLM."""
    # Threshold 1.0 guarantees a miss, exercising the fallback-intent branch.
    config = _embeddings_config(threshold=1.0)
    response, llm = generate(config, [], "totally unrelated request")

    assert response.response == "Hello!"
    assert "UserIntent:express greeting" in event_sequence(response)
    assert llm_tasks(response) == []
    assert llm.calls == []


def test_embeddings_only_threshold_miss_falls_through_to_llm():
    """A miss with no fallback intent falls through to LLM intent detection."""
    config = _embeddings_config(threshold=1.0)
    config.rails.dialog.user_messages.embeddings_only_fallback_intent = None
    response, llm = generate(config, ["  express greeting"], "totally unrelated request")

    assert "UserIntent:express greeting" in event_sequence(response)
    assert llm_tasks(response) == ["generate_user_intent"]
    # Canonical-form detection, not the general path: no ``stop``.
    assert llm.calls == [("generate", None)]


# ---------------------------------------------------------------------------
# Single-call mode (generate_intent_steps_message).
# ---------------------------------------------------------------------------


def test_single_call_intent_steps_message():
    """One LLM call produces UserIntent + cached bot intent/message events."""
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"
        define flow
            user express greeting
            bot express greeting
        """
    )
    config.rails.dialog.single_call.enabled = True
    response, llm = generate(
        config,
        ['  express greeting\nbot express greeting\n  "Hello, there!"'],
        "hello there!",
    )

    assert response.response == "Hello, there!"
    assert event_sequence(response) == [
        "UserIntent:express greeting|cache:bot_intent_event,bot_message_event",
        "BotIntent:express greeting",
        "BotMessage:Hello, there!",
    ]
    # A single LLM call computes all three phases; 2 and 3 unpack the cache.
    assert llm_tasks(response) == ["generate_intent_steps_message"]
    assert llm.calls == [("generate", None)]


def test_single_call_multimodal_input():
    """Multimodal (list) user content is normalized before intent detection.

    generate_intent_steps_message previously passed event["text"] (a list) to the
    index search unchanged; it now joins the text parts like generate_user_intent.
    """
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"
        define flow
            user express greeting
            bot express greeting
        """
    )
    config.rails.dialog.single_call.enabled = True
    llm = RecordingFakeLLM(responses=['  express greeting\nbot express greeting\n  "Hello, there!"'])
    chat = TestChat(config, llm=llm)
    response = cast(
        GenerationResponse,
        chat.app.generate(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello there!"}]}],
            options=LOG_OPTS,
        ),
    )

    assert event_sequence(response) == [
        "UserIntent:express greeting|cache:bot_intent_event,bot_message_event",
        "BotIntent:express greeting",
        "BotMessage:Hello, there!",
    ]
    assert llm_tasks(response) == ["generate_intent_steps_message"]


def test_single_call_streaming_handoff():
    """The ``<<STREAMING[uid]>>`` handoff streams the bot message via stop tokens.

    Streaming uses ``stream_async`` (no ``GenerationLog``), so only the
    model-call channel (mode + ``stop``) and the streamed chunk sequence are
    asserted -- these are the Phase 4 surface.
    """

    async def run():
        config = RailsConfig.from_content(
            """
            define user express greeting
                "hello"
            define flow
                user express greeting
                bot express greeting
            """,
            "streaming: True\n",
        )
        config.rails.dialog.single_call.enabled = True
        llm = RecordingFakeLLM(responses=['  express greeting\nbot express greeting\n  "Hello, there!"'])
        chat = TestChat(config, llm=llm, streaming=True)
        chunks = []
        async for chunk in chat.app.stream_async(messages=[{"role": "user", "content": "hello there!"}]):
            chunks.append(chunk)
        return llm, chunks

    llm, chunks = asyncio.run(run())

    # The single intent-steps call streams with the intent stop tokens.
    assert llm.calls == [("stream", ("\nuser ", "\nUser "))]
    assert chunks == ["Hello, ", "there!"]


def test_general_streaming_push():
    """The no-user-messages general path streams during the call (site 1).

    The call passes the streaming handler into ``llm_call`` (mode ``stream``) with
    ``stop=["User:"]``; the post-call ``push_chunk`` does not add a duplicate
    visible chunk. Pinning the exact chunk sequence guards the Phase 1/Phase 4
    handling of this path.
    """

    async def run():
        config = RailsConfig.from_content(yaml_content="models: []\nstreaming: True\n")
        llm = RecordingFakeLLM(responses=["hello world foo"])
        chat = TestChat(config, llm=llm, streaming=True)
        chunks = []
        async for chunk in chat.app.stream_async(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        return llm, chunks

    llm, chunks = asyncio.run(run())

    assert llm.calls == [("stream", ("User:",))]
    assert chunks == ["hello ", "world ", "foo"]


# ---------------------------------------------------------------------------
# Passthrough / general mode (no user_messages).
# ---------------------------------------------------------------------------


def test_general_no_user_messages():
    """No user messages, not passthrough: the non-passthrough general path.

    This is Phase 1 site 1: it renders the GENERAL prompt with ``relevant_chunks``
    and calls the LLM with ``stop=["User:"]``. The ``stop`` and the prompt
    fingerprint guard the helper extraction.
    """
    config = RailsConfig.from_content(yaml_content="models: []\n")
    response, llm = generate(config, ["I am a general answer."], "tell me something")

    assert response.response == "I am a general answer."
    assert event_sequence(response) == ["BotMessage:I am a general answer."]
    assert llm_tasks(response) == ["general"]
    assert llm.calls == [("generate", ("User:",))]
    assert prompt_fingerprints(response) == ["9ec81a223e"]


def test_general_reasoning_trace_emits_bot_thinking():
    """A reasoning trace produces ``bot_thinking`` context + a ``BotThinking`` event.

    Guards the reasoning-trace packaging that Phase 2 extracts into
    ``_emit_general_bot_turn``: the context update precedes the ``BotThinking``
    event, which precedes the ``BotMessage``.
    """
    config = RailsConfig.from_content(yaml_content="models: []\n")
    llm = RecordingFakeLLM(llm_responses=[LLMResponse(content="the answer", reasoning="let me think")])
    chat = TestChat(config, llm=llm)
    response = cast(GenerationResponse, chat.app.generate("question?", options=LOG_OPTS))

    assert response.response == "the answer"
    assert event_sequence(response) == [
        "ctx:bot_thinking=let me think",
        "BotThinking:let me think",
        "BotMessage:the answer",
    ]
    assert llm_tasks(response) == ["general"]


def test_passthrough_no_fn():
    """Passthrough without a passthrough fn: the raw-prompt general path.

    This is Phase 1 site 1b: the prompt is the raw user input (no rendered
    GENERAL template) and the LLM is called with no ``stop`` -- the divergence
    from site 1 that only the model-call channel can see.
    """
    config = RailsConfig.from_content(yaml_content="passthrough: true\n")
    response, llm = generate(config, ["passthrough llm answer"], "hello")

    assert response.response == "passthrough llm answer"
    assert event_sequence(response) == ["BotMessage:passthrough llm answer"]
    assert llm_tasks(response) == ["general"]
    assert llm.calls == [("generate", None)]
    assert prompt_fingerprints(response) == ["55b2b1b03f"]


def test_passthrough_with_fn():
    """A passthrough fn supplies the output and a ``passthrough_output`` event.

    The ``passthrough_output`` rides as a ``ContextUpdate`` *event* emitted before
    the ``BotMessage`` (distinct from ``generate_bot_message``, which returns it
    as a ``context_updates`` dict entry); the ordering pins that form.
    """
    config = RailsConfig.from_content(yaml_content="passthrough: true\n")
    llm = RecordingFakeLLM(responses=["unused"])
    chat = TestChat(config, llm=llm)

    async def passthrough_fn(context, events):
        return "fn output text", {"extra": 1}

    chat.app.passthrough_fn = passthrough_fn
    response = cast(GenerationResponse, chat.app.generate("hello", options=LOG_OPTS))

    assert response.response == "fn output text"
    assert event_sequence(response) == [
        "ctx:passthrough_output={'extra': 1}",
        "BotMessage:fn output text",
    ]
    assert llm_tasks(response) == []
    assert llm.calls == []


def test_passthrough_tool_calls():
    """Tool calls in passthrough mode emit a ``BotToolCalls`` event."""
    config = RailsConfig.from_content(
        yaml_content="""
        passthrough: true
        models:
          - type: main
            engine: openai
            model: gpt-4
        """
    )
    llm = RecordingFakeLLM(
        llm_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=ToolCallFunction(name="t", arguments={"p": "v"}),
                    )
                ],
            )
        ]
    )
    chat = TestChat(config, llm=llm)
    response = cast(GenerationResponse, chat.app.generate("call the tool", options=LOG_OPTS))

    assert event_sequence(response) == ["BotToolCalls:1"]
    assert llm_tasks(response) == ["general"]


def test_single_call_general_no_user_messages():
    """Single-call enabled but no user messages: the intent-steps general path.

    This is Phase 1 site 2: ``generate_intent_steps_message`` renders the GENERAL
    prompt with no context and calls the LLM with no ``stop``. The prompt
    fingerprint guards the (no-context) render against the site-1 render shape.
    """
    config = RailsConfig.from_content(yaml_content="passthrough: false\n")
    config.rails.dialog.single_call.enabled = True
    response, llm = generate(config, ["a general single-call answer"], "hi")

    assert response.response == "a general single-call answer"
    assert event_sequence(response) == ["BotMessage:a general single-call answer"]
    assert llm_tasks(response) == ["general"]
    assert llm.calls == [("generate", None)]
    assert prompt_fingerprints(response) == ["85c3b5ec27"]


# ---------------------------------------------------------------------------
# generate_value.
# ---------------------------------------------------------------------------


def test_generate_value_three_phase():
    """``$var = ...`` triggers a ``generate_value`` LLM call between phases."""
    config = RailsConfig.from_path(os.path.join(os.path.dirname(__file__), "test_configs", "generate_value"))
    llm = RecordingFakeLLM(
        responses=[
            "  ask math question",
            '"What is the largest prime factor for 1024?"',
            '  "The largest prime factor for 1024 is 2."',
        ]
    )
    chat = TestChat(config, llm=llm)

    async def mock_wolfram_alpha_request_action(query):
        return "2"

    chat.app.register_action(mock_wolfram_alpha_request_action, "wolfram alpha request")
    response = cast(
        GenerationResponse,
        chat.app.generate("What is the largest prime factor for 1024", options=LOG_OPTS),
    )

    assert response.response == "The largest prime factor for 1024 is 2."
    assert llm_tasks(response) == [
        "generate_user_intent",
        "generate_value",
        "generate_bot_message",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "completion,expected",
    [
        ("42", "42"),
        ("42;", "42"),  # trailing ``;`` is stripped before safe_eval
        ('"hello"', "hello"),
    ],
)
async def test_generate_value_parsing(completion, expected):
    """The value-parsing tail (first line, ``;`` strip, safe_eval) is preserved.

    Note: ``safe_eval`` is intentionally lenient (it wraps unparseable input as a
    string rather than raising), so the action's ``except -> ValueError`` branch
    is effectively unreachable in practice and is not exercised here.
    """
    config = RailsConfig.from_content(yaml_content="models: []\n")
    actions = LLMGenerationActions(
        config=config,
        llm=FakeLLMModel(responses=[completion]),
        llm_task_manager=LLMTaskManager(config),
        get_embedding_search_provider_instance=MagicMock(return_value=None),
    )
    events = [
        {"type": "UserMessage", "text": "give me a number"},
        {"type": "StartInternalSystemAction", "action_name": "generate_value", "action_result_key": "x"},
    ]
    value = await actions.generate_value(instructions="give a number", events=events, var_name="x")
    assert value == expected
