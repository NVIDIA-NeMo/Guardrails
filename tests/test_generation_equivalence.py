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

"""Behavioral regression suite for ``LLMGenerationActions``.

Pins the observable behavior of ``generate_user_intent``, ``generate_next_steps``,
``generate_bot_message``, ``generate_value`` and ``generate_intent_steps_message``
across the decomposition of that module -- including the behavior changes the
fix commits intentionally introduce (multimodal normalization, the converged
single-call general turn, and streaming-handoff eviction).

An event-only oracle is insufficient because some observable behavior is not in
the event stream, so each scenario is pinned across four channels:

1. Emitted event list -- the ordered, generation-relevant projection of
   ``GenerationLog.internal_events`` (``event_sequence``). The random ``flow_id``
   of ``start_flow`` is normalized away by keeping only the flow body; the
   ``passthrough_output`` ``ContextUpdate`` event and the ``skip_output_rails`` /
   ``_last_bot_prompt`` context updates appear here in order, so the "event form
   vs dict form" distinction is captured by position.
2. Per-call task -- ``GenerationLog.llm_calls[i].task`` (``llm_tasks``), e.g. the
   passthrough bot-message branch sets a ``generate_bot_message`` LLMCallInfo but
   parses as ``general``, which shows up as a ``general`` task in that path.
3. The ``stop`` argument and whether the call streamed -- captured by
   ``RecordingFakeLLM`` because ``LLMCallInfo`` has no ``stop`` field and the base
   fake discards it, so a dropped ``stop=["User:"]`` is invisible to every other
   channel. This is the only channel that distinguishes the ``Task.GENERAL`` call
   sites (rendered-template vs raw-prompt).
4. Streaming chunk sequence -- collected from ``stream_async`` for the streaming
   scenarios.
"""

import asyncio
import os
from typing import cast
from unittest.mock import MagicMock

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.llm.generation import LLMGenerationActions, build_single_call_payload
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from nemoguardrails.utils import new_event_dict
from tests.utils import TestChat

LOG_OPTS = {"log": {"llm_calls": True, "internal_events": True}}

# Context-update keys emitted by the generation actions that the oracle tracks.
_GENERATION_CONTEXT_KEYS = [
    "skip_output_rails",
    "passthrough_output",
    "bot_thinking",
    "_last_bot_prompt",
]


class RecordingFakeLLM(FakeLLMModel):
    """A ``FakeLLMModel`` that records the ``stop`` argument and call mode.

    ``LLMCallInfo`` does not carry ``stop`` and the base fake discards it, so this
    is the only place a dropped or added ``stop`` is observable. ``calls`` is a
    list of ``(mode, stop)`` where ``mode`` is ``"generate"`` or ``"stream"``
    (i.e. whether a streaming handler was passed into ``llm_call``).
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


def event_sequence(response):
    """Project ``internal_events`` to the ordered, generation-relevant events.

    Volatile fields (uids, timestamps) are dropped; the ``start_flow`` flow_id
    (a random uuid) is normalized away by keeping only the flow body. The large
    ``_last_bot_prompt`` value is reduced to ``<present>`` -- its content is
    asserted through ``call_prompts`` where it matters.
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


def call_prompts(response):
    """The prompt string passed to each LLM call, in order.

    Used to distinguish the ``Task.GENERAL`` call sites: the templated path
    renders the GENERAL prompt (starts with the general instruction), while the
    raw passthrough path sends the conversation verbatim with no template.
    """
    return [call.prompt or "" for call in (response.log.llm_calls or [])]


def generate(config, completions, message):
    """Run a single-turn generation, returning ``(response, recording_llm)``."""
    recording_llm = RecordingFakeLLM(responses=completions)
    chat = TestChat(config, llm=recording_llm)
    response = cast(GenerationResponse, chat.app.generate(message, options=LOG_OPTS))
    return response, recording_llm


class TestDialogMode:
    """Multi-call dialog: the classic three-phase pipeline."""

    def test_three_phase(self):
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
        # Each rendered prompt carries the dynamic context (guards a dropped
        # render-context key, which the prompt-independent fake would otherwise hide).
        prompts = call_prompts(response)
        assert 'user "I want to book a flight"' in prompts[0]
        assert "user ask booking" in prompts[1]
        assert "bot respond booking" in prompts[2]

    def test_predefined_bot_message(self):
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

    def test_context_var_bot_intent(self):
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

    def test_multi_step_generation(self):
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
        # The completions are consumed across turns, so drive the first two turns
        # to advance state before capturing the multi-step turn (the third).
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
    return RailsConfig.from_content(
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


class TestEmbeddingsOnly:
    """Dialog mode with embeddings-only intent detection."""

    def test_index_hit(self):
        """A similarity hit returns the matched intent with zero LLM calls.

        Threshold 0.0 guarantees the index search returns a hit.
        """
        config = _embeddings_config(threshold=0.0)
        response, llm = generate(config, [], "hi")

        assert response.response == "Hello!"
        assert "UserIntent:express greeting" in event_sequence(response)
        assert llm_tasks(response) == []
        assert llm.calls == []

    def test_threshold_miss_uses_fallback_intent(self):
        """A below-threshold search falls back to the configured intent, no LLM.

        Threshold 1.0 guarantees a miss, exercising the fallback-intent branch.
        """
        config = _embeddings_config(threshold=1.0)
        response, llm = generate(config, [], "totally unrelated request")

        assert response.response == "Hello!"
        assert "UserIntent:express greeting" in event_sequence(response)
        assert llm_tasks(response) == []
        assert llm.calls == []

    def test_threshold_miss_falls_through_to_llm(self):
        """A miss with no fallback intent falls through to LLM intent detection.

        Canonical-form detection, not the general path, so there is no ``stop``.
        """
        config = _embeddings_config(threshold=1.0)
        config.rails.dialog.user_messages.embeddings_only_fallback_intent = None
        response, llm = generate(config, ["  express greeting"], "totally unrelated request")

        assert "UserIntent:express greeting" in event_sequence(response)
        assert llm_tasks(response) == ["generate_user_intent"]
        assert llm.calls == [("generate", None)]


class TestSingleCall:
    """Single-call mode via ``generate_intent_steps_message``."""

    def test_intent_steps_message(self):
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
        assert 'user "hello there!"' in call_prompts(response)[0]

    def test_multimodal_input(self):
        """Multimodal (list) user content is normalized before intent detection.

        ``generate_intent_steps_message`` previously passed ``event["text"]`` (a
        list) to the index search unchanged; it now joins the text parts like
        ``generate_user_intent``. Non-text parts (e.g. ``image_url``) are ignored,
        so text normalization still yields the same intent and response.
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
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello there!"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                        ],
                    }
                ],
                options=LOG_OPTS,
            ),
        )

        assert event_sequence(response) == [
            "UserIntent:express greeting|cache:bot_intent_event,bot_message_event",
            "BotIntent:express greeting",
            "BotMessage:Hello, there!",
        ]
        assert llm_tasks(response) == ["generate_intent_steps_message"]

    def test_streaming_handoff(self):
        """The ``<<STREAMING[uid]>>`` handoff streams the bot message via stop tokens.

        Streaming uses ``stream_async`` (no ``GenerationLog``), so only the
        model-call channel (mode + ``stop``) and the streamed chunk sequence are
        asserted.
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

    @pytest.mark.asyncio
    async def test_cache_miss_searches_with_bot_intent(self):
        """On a single-call cache miss, the bot-message search uses the bot intent.

        Regression: the single-call branch previously rebound ``event`` to the
        user-intent event, so when the cached bot intent did not match the
        ``BotIntent`` being generated, the fall-through ``bot_message_index``
        search ran with the user intent instead of the bot intent.
        """
        config = RailsConfig.from_content(
            """
            define user express greeting
                "hello"
            """,
            yaml_content="rails:\n  dialog:\n    single_call:\n      enabled: True\n",
        )
        actions = LLMGenerationActions(
            config=config,
            llm=FakeLLMModel(responses=['  "regenerated bot message"']),
            llm_task_manager=LLMTaskManager(config),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

        # Record the text the bot-message index is searched with.
        searched_with = []

        async def recording_search(text, max_results, threshold):
            searched_with.append(text)
            return []

        actions.bot_message_index = MagicMock()
        actions.bot_message_index.search = recording_search

        # A single-call cache whose bot intent does NOT match the BotIntent being
        # generated, so the cache is a miss and generation falls through.
        events = [
            new_event_dict(
                "UserIntent",
                intent="express greeting",
                additional_info=build_single_call_payload(
                    bot_intent_event=new_event_dict("BotIntent", intent="cached bot intent"),
                    bot_message_event=new_event_dict("BotMessage", text="cached message"),
                ),
            ),
            new_event_dict("BotIntent", intent="respond kindly"),
        ]

        await actions.generate_bot_message(events=events, context={})

        assert searched_with == ["respond kindly"]

    @pytest.mark.asyncio
    async def test_cache_miss_evicts_streaming_handoff(self):
        """A single-call cache miss evicts an embedded streaming handoff (no leak).

        If the cached bot message carried a ``<<STREAMING[uid]>>`` marker and the
        bot intent no longer matches, the registered handler must be evicted
        rather than lingering for the module lifetime.
        """
        from nemoguardrails.actions.llm.generation import _streaming_handoff
        from nemoguardrails.streaming import StreamingHandler

        config = RailsConfig.from_content(
            """
            define user express greeting
                "hello"
            """,
            yaml_content="rails:\n  dialog:\n    single_call:\n      enabled: True\n",
        )
        actions = LLMGenerationActions(
            config=config,
            llm=FakeLLMModel(responses=['  "regenerated bot message"']),
            llm_task_manager=LLMTaskManager(config),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

        async def _empty_search(text, max_results, threshold):
            return []

        actions.bot_message_index = MagicMock()
        actions.bot_message_index.search = _empty_search

        handler = StreamingHandler()
        marker = _streaming_handoff.register(handler)

        events = [
            new_event_dict(
                "UserIntent",
                intent="express greeting",
                additional_info=build_single_call_payload(
                    bot_intent_event=new_event_dict("BotIntent", intent="cached bot intent"),
                    bot_message_event=new_event_dict("BotMessage", text=marker),
                ),
            ),
            new_event_dict("BotIntent", intent="respond kindly"),
        ]

        await actions.generate_bot_message(events=events, context={})

        # The handoff was consumed on the cache-miss fall-through, not leaked.
        with pytest.raises(KeyError):
            _streaming_handoff.take(handler.uid)

    @pytest.mark.asyncio
    async def test_predefined_message_evicts_pending_handoff(self):
        """A predefined-message bot turn still evicts a pending single-call handoff.

        If phase 1 (streaming) registered a handoff but phase 3 resolves the bot
        intent to a predefined ``define bot`` message, the handler must not leak.
        """
        from nemoguardrails.actions.llm.generation import _streaming_handoff
        from nemoguardrails.streaming import StreamingHandler

        config = RailsConfig.from_content(
            """
            define user express greeting
                "hello"
            define bot express greeting
                "Hello!"
            """,
            yaml_content="rails:\n  dialog:\n    single_call:\n      enabled: True\n",
        )
        actions = LLMGenerationActions(
            config=config,
            llm=FakeLLMModel(responses=[]),
            llm_task_manager=LLMTaskManager(config),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

        handler = StreamingHandler()
        marker = _streaming_handoff.register(handler)

        events = [
            new_event_dict(
                "UserIntent",
                intent="express greeting",
                additional_info=build_single_call_payload(
                    bot_intent_event=new_event_dict("BotIntent", intent="express greeting"),
                    bot_message_event=new_event_dict("BotMessage", text=marker),
                ),
            ),
            new_event_dict("BotIntent", intent="express greeting"),
        ]

        await actions.generate_bot_message(events=events, context={})

        # The predefined-message branch bypassed the cache but still released the handoff.
        with pytest.raises(KeyError):
            _streaming_handoff.take(handler.uid)

    @pytest.mark.asyncio
    async def test_cache_returns_none_when_last_user_event_not_user_intent(self):
        """The cache falls back when the last user-intent event is a ``UserMessage``
        rather than a ``UserIntent`` (current silent fall-through)."""
        config = RailsConfig.from_content(
            'define user express greeting\n    "hello"\n',
            yaml_content="rails:\n  dialog:\n    single_call:\n      enabled: True\n",
        )
        actions = LLMGenerationActions(
            config=config,
            llm=FakeLLMModel(responses=[]),
            llm_task_manager=LLMTaskManager(config),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )
        result = await actions._bot_message_from_single_call_cache(
            {"type": "UserMessage", "text": "hi"},
            events=[new_event_dict("BotIntent", intent="respond")],
            streaming_handler=None,
        )
        assert result is None


class TestStreaming:
    """Streaming surfaces and the handoff registry."""

    def test_general_streaming_push(self):
        """The no-user-messages general path streams during the call.

        The call passes the streaming handler into ``llm_call`` (mode ``stream``)
        with ``stop=["User:"]``; the post-call ``push_chunk`` does not add a
        duplicate visible chunk.
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

    def test_handoff_registry_round_trip_and_eviction(self):
        """The handoff registry round-trips the marker and evicts on take (no leak)."""
        from nemoguardrails.actions.llm.generation import _StreamingHandoffRegistry
        from nemoguardrails.streaming import StreamingHandler

        registry = _StreamingHandoffRegistry()
        handler = StreamingHandler()

        marker = registry.register(handler)
        assert marker == f'Bot message: "<<STREAMING[{handler.uid}]>>"'
        assert registry.parse_marker(marker) == handler.uid
        assert registry.parse_marker("not a marker") is None

        # Malformed markers (prefix present but suffix missing/trailing) are not
        # markers -- parse_marker must reject them rather than return a bad uid.
        assert registry.parse_marker(f'Bot message: "<<STREAMING[{handler.uid}') is None
        assert registry.parse_marker(f'Bot message: "<<STREAMING[{handler.uid}]>> trailing') is None

        assert registry.take(handler.uid) is handler
        # Taken once, the handler is gone -- it does not linger for the module lifetime.
        with pytest.raises(KeyError):
            registry.take(handler.uid)


class TestGeneralAndPassthrough:
    """No-user-messages general path and passthrough mode."""

    def test_general_no_user_messages(self):
        """No user messages, not passthrough: the rendered-GENERAL path.

        Renders the GENERAL prompt with ``relevant_chunks`` and calls the LLM with
        ``stop=["User:"]``; the prompt and ``stop`` guard the helper extraction.
        """
        config = RailsConfig.from_content(yaml_content="models: []\n")
        response, llm = generate(config, ["I am a general answer."], "tell me something")

        assert response.response == "I am a general answer."
        assert event_sequence(response) == ["BotMessage:I am a general answer."]
        assert llm_tasks(response) == ["general"]
        assert llm.calls == [("generate", ("User:",))]
        prompt = call_prompts(response)[0]
        assert prompt.startswith("Below is a conversation")
        assert prompt.endswith("User: tell me something\nAssistant:")

    def test_general_reasoning_trace_emits_bot_thinking(self):
        """A reasoning trace produces ``bot_thinking`` context + a ``BotThinking`` event.

        The context update precedes the ``BotThinking`` event, which precedes the
        ``BotMessage``.
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

    def test_passthrough_no_fn(self):
        """Passthrough without a passthrough fn: the raw-prompt path.

        The prompt is the raw conversation (no rendered GENERAL template) and the
        LLM is called with no ``stop`` -- the divergence from the rendered path
        that only the model-call channel can see.
        """
        config = RailsConfig.from_content(yaml_content="passthrough: true\n")
        response, llm = generate(config, ["passthrough llm answer"], "hello")

        assert response.response == "passthrough llm answer"
        assert event_sequence(response) == ["BotMessage:passthrough llm answer"]
        assert llm_tasks(response) == ["general"]
        assert llm.calls == [("generate", None)]
        prompt = call_prompts(response)[0]
        assert "Below is a conversation" not in prompt
        assert "hello" in prompt

    def test_passthrough_with_fn(self):
        """A passthrough fn supplies the output and a ``passthrough_output`` event.

        The ``passthrough_output`` rides as a ``ContextUpdate`` event emitted
        before the ``BotMessage`` (distinct from ``generate_bot_message``, which
        returns it as a ``context_updates`` dict entry); the ordering pins that
        form.
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

    def test_passthrough_tool_calls(self):
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
        # The full tool-call payload (id, function name, arguments) is surfaced on
        # the public response, not just the count.
        assert response.response == ""
        assert response.tool_calls == [
            {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": {"p": "v"}}}
        ]

    def test_single_call_general_no_user_messages(self):
        """Single-call enabled but no user messages: the converged general bot turn.

        This branch delegates to ``_emit_general_bot_turn``, so it behaves like
        ``generate_user_intent``'s general path: a GENERAL prompt rendered with
        ``relevant_chunks`` and an LLM call with ``stop=["User:"]`` (previously it
        rendered with no context and no stop). Rendering GENERAL with empty
        ``relevant_chunks`` is identical to rendering it with no context.
        """
        config = RailsConfig.from_content(yaml_content="passthrough: false\n")
        config.rails.dialog.single_call.enabled = True
        response, llm = generate(config, ["a general single-call answer"], "hi")

        assert response.response == "a general single-call answer"
        assert event_sequence(response) == ["BotMessage:a general single-call answer"]
        assert llm_tasks(response) == ["general"]
        assert llm.calls == [("generate", ("User:",))]
        assert call_prompts(response)[0].startswith("Below is a conversation")

    def test_single_call_general_reasoning_trace(self):
        """The single-call general path emits ``BotThinking`` like the multi-call one.

        ``generate_intent_steps_message``'s else-branch previously dropped
        reasoning traces; converging it onto ``_emit_general_bot_turn`` now
        packages them.
        """
        config = RailsConfig.from_content(yaml_content="passthrough: false\n")
        config.rails.dialog.single_call.enabled = True
        llm = RecordingFakeLLM(llm_responses=[LLMResponse(content="the answer", reasoning="let me think")])
        chat = TestChat(config, llm=llm)
        response = cast(GenerationResponse, chat.app.generate("hi", options=LOG_OPTS))

        assert response.response == "the answer"
        assert event_sequence(response) == [
            "ctx:bot_thinking=let me think",
            "BotThinking:let me think",
            "BotMessage:the answer",
        ]
        assert llm_tasks(response) == ["general"]

    def test_dialog_with_passthrough_uses_bot_message_branch(self):
        """user_messages + passthrough: generate_bot_message takes the passthrough
        branch, which reports the ``generate_bot_message`` task but parses GENERAL."""
        config = RailsConfig.from_content(
            """
            define user express greeting
                "hello"
            define flow
                user express greeting
                bot respond kindly
            """,
            yaml_content="passthrough: true\n",
        )
        response, llm = generate(config, ["  express greeting", "passthrough bot answer"], "hello")

        assert response.response == "passthrough bot answer"
        assert llm_tasks(response) == ["generate_user_intent", "generate_bot_message"]
        assert event_sequence(response)[-1] == "BotMessage:passthrough bot answer"


class TestGenerateValue:
    """``generate_value`` action ($var = ...)."""

    def test_three_phase(self):
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
    async def test_parsing(self, completion, expected):
        """The value-parsing tail (first line, ``;`` strip, safe_eval) is preserved.

        ``safe_eval`` is intentionally lenient (it wraps unparseable input as a
        string rather than raising), so the action's ``except -> ValueError``
        branch is effectively unreachable in practice and is not exercised here.
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


class TestStreamingPatternFor:
    """Contract of ``_streaming_pattern_for(output_parser, *, include_bot_message_parser)``."""

    def test_pattern_matrix(self):
        from nemoguardrails.actions.llm.generation import _streaming_pattern_for

        verbose = ('Bot message: "', '"')
        plain = ('  "', '"')

        # verbose_v1 is always verbose, regardless of the flag.
        assert _streaming_pattern_for("verbose_v1", include_bot_message_parser=False) == verbose
        assert _streaming_pattern_for("verbose_v1", include_bot_message_parser=True) == verbose
        # bot_message is verbose only when the flag is set (the generate_bot_message site).
        assert _streaming_pattern_for("bot_message", include_bot_message_parser=True) == verbose
        assert _streaming_pattern_for("bot_message", include_bot_message_parser=False) == plain
        # Anything else is the plain pattern.
        assert _streaming_pattern_for("something_else", include_bot_message_parser=True) == plain
        assert _streaming_pattern_for(None, include_bot_message_parser=False) == plain


class TestBotTurnOutputEvents:
    """Contract of the module-level ``_bot_turn_output_events`` helper."""

    def test_reasoning_trace_prepends_bot_thinking_and_records_context(self):
        from nemoguardrails.actions.llm.generation import _bot_turn_output_events
        from nemoguardrails.context import reasoning_trace_var

        final = new_event_dict("BotMessage", text="hi")
        context_updates = {}
        reasoning_trace_var.set("let me think")
        try:
            events = _bot_turn_output_events(final, context_updates)
        finally:
            reasoning_trace_var.set(None)

        assert [event["type"] for event in events] == ["BotThinking", "BotMessage"]
        assert events[0]["content"] == "let me think"
        assert events[1] is final
        assert context_updates["bot_thinking"] == "let me think"

    def test_reasoning_trace_without_context_updates_does_not_record(self):
        from nemoguardrails.actions.llm.generation import _bot_turn_output_events
        from nemoguardrails.context import reasoning_trace_var

        reasoning_trace_var.set("let me think")
        try:
            events = _bot_turn_output_events(new_event_dict("BotMessage", text="hi"))
        finally:
            reasoning_trace_var.set(None)

        # BotThinking is still emitted; there is just no dict to record into.
        assert [event["type"] for event in events] == ["BotThinking", "BotMessage"]

    def test_no_reasoning_trace_returns_only_final(self):
        from nemoguardrails.actions.llm.generation import _bot_turn_output_events
        from nemoguardrails.context import reasoning_trace_var

        reasoning_trace_var.set(None)
        final = new_event_dict("BotMessage", text="hi")
        events = _bot_turn_output_events(final, {})
        assert events == [final]


class TestBuildIntentStepsExamples:
    """Contract of the single-call few-shot example builder (prompt-only, so the
    equivalence scenarios cannot observe it end-to-end)."""

    def _actions(self):
        config = RailsConfig.from_content('define user express greeting\n    "hi"\n')
        return LLMGenerationActions(
            config=config,
            llm=FakeLLMModel(responses=[]),
            llm_task_manager=LLMTaskManager(config),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

    @pytest.mark.asyncio
    async def test_pairs_intent_with_flow_and_bot_message(self):
        from types import SimpleNamespace

        actions = self._actions()

        async def user_search(text, max_results, threshold):
            return [SimpleNamespace(text="hello", meta={"intent": "express greeting"})]

        async def flows_search(text, max_results, threshold):
            flow = "user express greeting\nbot express greeting"
            return [SimpleNamespace(text=flow, meta={"flow": flow})]

        async def bot_search(text, max_results, threshold):
            return [SimpleNamespace(text="express greeting", meta={"text": "Hello!"})]

        actions.user_message_index = MagicMock()
        actions.user_message_index.search = user_search
        actions.flows_index = MagicMock()
        actions.flows_index.search = flows_search
        actions.bot_message_index = MagicMock()
        actions.bot_message_index.search = bot_search

        examples, intents = await actions._build_intent_steps_examples("hi")

        assert intents == ["express greeting"]
        assert len(examples) == 1
        assert 'user "hello"' in examples[0]
        assert "  express greeting" in examples[0]
        assert "bot express greeting" in examples[0]
        assert '"Hello!"' in examples[0]

    @pytest.mark.asyncio
    async def test_intent_without_flow_is_skipped(self):
        from types import SimpleNamespace

        actions = self._actions()

        async def user_search(text, max_results, threshold):
            return [SimpleNamespace(text="hello", meta={"intent": "express greeting"})]

        async def empty_flows_search(text, max_results, threshold):
            return []

        actions.user_message_index = MagicMock()
        actions.user_message_index.search = user_search
        actions.flows_index = MagicMock()
        actions.flows_index.search = empty_flows_search
        actions.bot_message_index = None

        examples, intents = await actions._build_intent_steps_examples("hi")

        # The intent is still a candidate, but with no flow it yields no example.
        assert intents == ["express greeting"]
        assert examples == []


class TestNextStepsBranches:
    """Single-step branches of ``generate_next_steps`` (non-multi-step)."""

    def _actions(self, response):
        config = RailsConfig.from_content('define user express greeting\n    "hi"\n')
        return LLMGenerationActions(
            config=config,
            llm=FakeLLMModel(responses=[response]),
            llm_task_manager=LLMTaskManager(config),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

    @pytest.mark.asyncio
    async def test_bot_intent_comma_cleanup(self):
        actions = self._actions("bot respond politely, and more")
        events = [new_event_dict("UserIntent", intent="express greeting")]
        result = await actions.generate_next_steps(events=events)
        assert result.events[0]["type"] == "BotIntent"
        assert result.events[0]["intent"] == "respond politely"

    @pytest.mark.asyncio
    async def test_non_bot_line_yields_general_response(self):
        actions = self._actions("this is not a bot line")
        events = [new_event_dict("UserIntent", intent="express greeting")]
        result = await actions.generate_next_steps(events=events)
        assert result.events[0]["type"] == "BotIntent"
        assert result.events[0]["intent"] == "general response"


class TestPassthroughRawPromptList:
    """The passthrough ``raw_llm_request`` is-a-list branch of ``_emit_general_bot_turn``."""

    @pytest.mark.asyncio
    async def test_last_user_content_replaced_in_prompt(self):
        from nemoguardrails.context import raw_llm_request

        captured = {}

        class CapturingFake(FakeLLMModel):
            async def generate_async(self, prompt, *, stop=None, **kwargs):
                captured["prompt"] = prompt
                return await super().generate_async(prompt, stop=stop, **kwargs)

        config = RailsConfig.from_content(yaml_content="passthrough: true\n")
        actions = LLMGenerationActions(
            config=config,
            llm=CapturingFake(responses=["passthrough answer"]),
            llm_task_manager=LLMTaskManager(config),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

        raw_list = [{"role": "user", "content": "original"}]
        token = raw_llm_request.set(raw_list)
        try:
            events = [new_event_dict("UserMessage", text="altered by input rails")]
            await actions.generate_user_intent(events=events, context={}, config=config)
        finally:
            raw_llm_request.reset(token)

        # The prompt sent to the LLM reflects the input-rail-altered user content
        # (the shallow copy shares the element dict, so the mutation is visible);
        # the original content is gone.
        assert isinstance(captured["prompt"], list)
        prompt_text = str(captured["prompt"])
        assert "altered by input rails" in prompt_text
        assert "original" not in prompt_text


class TestGenerateGeneralResponse:
    """Contract of the shared ``_generate_general_response`` helper."""

    @pytest.mark.asyncio
    async def test_reports_llm_call_task_distinct_from_parse_task(self):
        from nemoguardrails.context import llm_call_info_var
        from nemoguardrails.llm.types import Task

        config = RailsConfig.from_content(yaml_content="models: []\n")
        actions = LLMGenerationActions(
            config=config,
            llm=FakeLLMModel(responses=["output"]),
            llm_task_manager=LLMTaskManager(config),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

        llm_call_info_var.set(None)
        await actions._generate_general_response(
            generation_llm=actions.llm,
            prompt="hello",
            stream_during_call=False,
            llm_call_task=Task.GENERATE_BOT_MESSAGE,
            parse_task=Task.GENERAL,
        )

        info = llm_call_info_var.get()
        assert info is not None
        # The call is reported under llm_call_task even though it parses as parse_task.
        assert info.task == Task.GENERATE_BOT_MESSAGE.value


class TestMultiCallMultimodal:
    """Non-single-call multimodal normalization in ``generate_user_intent``."""

    def test_non_text_parts_ignored(self):
        config = RailsConfig.from_content('define user express greeting\n    "hello"\n')
        llm = RecordingFakeLLM(responses=["  express greeting", "bot respond", '  "Hi there!"'])
        chat = TestChat(config, llm=llm)
        response = cast(
            GenerationResponse,
            chat.app.generate(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello there"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                        ],
                    }
                ],
                options=LOG_OPTS,
            ),
        )
        assert "UserIntent:express greeting" in event_sequence(response)
