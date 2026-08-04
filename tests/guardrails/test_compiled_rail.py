# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for CompiledRail, the manifest-driven rail unit that replaces RailAction.

A ``CompiledRail`` is built once per configured flow string: it reads the flow's
``RailSurface`` from the manifest catalog, resolves the library action, freezes a binding
plan, and thereafter turns a request into ``action(**kwargs)`` and the returned
``RailOutcome`` into a ``RailResult``.

Two properties are load-bearing and are pinned here rather than left to review:

*Compilation fails at construction, not at request time.* An unknown surface, a
wrong-direction surface, or a missing required ``$param=`` must raise
``RailCompilationError`` while the engine is being built, because ``unsupported_reason``
decides whether IORails can serve a config by attempting exactly this compilation.

*Rail model calls are captured from a sink, never by reading a contextvar afterwards.*
``CompiledRail`` installs a fresh ``processing_log_var`` list around the action call.
A rail that makes no model call contributes nothing to its own empty list, so it cannot
inherit a previous rail's record — structurally, not by clearing anything. Do not
"simplify" this into a post-hoc read of ``llm_call_info_var``: library actions open and
close their own scope *inside* the action, so such a read yields the outer value.

Deferred to later commits, deliberately not covered here: transform outcomes (PR 5),
``model_caches`` (PR 6), and the ``RailsManager``/``unsupported_reason`` rewiring.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.compiled_rail import (
    CompiledRail,
    RailCompilationError,
    RailDependencies,
    compile_rail,
    messages_to_events,
)
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.manifests import RailDirection
from nemoguardrails.testing.fake_model import FakeLLMModel

CONTENT_SAFETY_INPUT = "content safety check input $model=content_safety"
CONTENT_SAFETY_ACTION = "nemoguardrails.library.content_safety.actions.content_safety_check_input"
TOPIC_SAFETY_INPUT = "topic safety check input $model=topic_control"
TOPIC_SAFETY_ACTION = "nemoguardrails.library.topic_safety.actions.topic_safety_check_input"
JAILBREAK_INPUT = "jailbreak detection model"

USER_MESSAGES = [{"role": "user", "content": "hello there"}]


class RecordingAction:
    """Stand-in for a library action that records how it was called."""

    def __init__(self, outcome: Any = None):
        self.outcome = outcome if outcome is not None else RailOutcome.allow()
        self.kwargs: dict = {}

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.outcome


@pytest.fixture
def deps() -> RailDependencies:
    """The dependency bundle CompiledRail injects, with inert stand-ins for live engines.

    The models are real ``FakeLLMModel`` instances rather than mocks so an action that
    genuinely calls one behaves as it would in production; nothing here reaches a network.
    """
    return RailDependencies(
        llms={
            "main": FakeLLMModel(responses=["main"]),
            "content_safety": FakeLLMModel(responses=["safe"]),
            "topic_control": FakeLLMModel(responses=["on-topic"]),
        },
        llm_task_manager=MagicMock(),
        config=MagicMock(),
        http_client=MagicMock(),
        model_caches=None,
        tracer=None,
    )


@pytest.fixture
def record_llm_call():
    """Append an LLMCallInfo to the active processing-log sink, as track_llm_call does.

    Simulates a rail's model call without standing up an engine, so capture behavior can be
    asserted directly. Mirrors ``logging/llm_tracker.py:59-61``; if that append changes
    shape, this fixture and the production reader must change together.
    """

    def _record(task: str) -> None:
        processing_log = processing_log_var.get()
        assert processing_log is not None, "no processing-log sink is installed; CompiledRail should install one"
        processing_log.append({"type": "llm_call_info", "timestamp": 0.0, "data": LLMCallInfo(task=task)})

    return _record


class TestCompilation:
    """Compilation resolves the surface and freezes a binding plan, or fails loudly."""

    def test_compiles_a_shipped_surface(self, deps):
        """A configured flow string resolves to its manifest surface and library action."""
        rail = compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps)

        assert isinstance(rail, CompiledRail)
        assert rail.flow == CONTENT_SAFETY_INPUT
        assert rail.surface_name == "content safety check input"

    def test_unknown_surface_name_raises(self, deps):
        """A flow with no catalog surface fails at compile time with the name in the message."""
        with pytest.raises(RailCompilationError, match="no surface"):
            compile_rail("not a real rail", RailDirection.INPUT, deps)

    def test_wrong_direction_raises(self, deps):
        """An output-only surface configured as an input rail fails at compile time."""
        with pytest.raises(RailCompilationError, match="direction"):
            compile_rail("content safety check output $model=content_safety", RailDirection.INPUT, deps)

    def test_missing_required_surface_param_raises_at_compile_time(self, deps):
        """A surface needing $model= fails when the config omits it, not on the first request."""
        with pytest.raises(RailCompilationError, match="model"):
            compile_rail("content safety check input", RailDirection.INPUT, deps)


class TestBindingResolution:
    """Each BindingKind fills its action parameter from the right source."""

    @pytest.mark.asyncio
    async def test_surface_param_binding_supplies_the_configured_value(self, deps, monkeypatch):
        """$model=content_safety reaches the action as model_name."""
        action = RecordingAction()
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, action)

        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert action.kwargs is not None
        assert action.kwargs["model_name"] == "content_safety"

    @pytest.mark.asyncio
    async def test_context_carries_the_request_messages(self, deps, monkeypatch):
        """The per-request context dict exposes user_message for context-bound actions."""
        action = RecordingAction()
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, action)

        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert action.kwargs["context"]["user_message"] == "hello there"

    @pytest.mark.asyncio
    async def test_bot_response_reaches_an_output_rail_as_bot_message(self, deps, monkeypatch):
        """An output rail's context carries the generated response under bot_message."""
        action = RecordingAction()
        monkeypatch.setattr("nemoguardrails.library.content_safety.actions.content_safety_check_output", action)

        rail = compile_rail("content safety check output $model=content_safety", RailDirection.OUTPUT, deps)
        await rail.run(USER_MESSAGES, bot_response="the reply")

        assert action.kwargs["context"]["bot_message"] == "the reply"


class TestDependencyInjection:
    """Injection is driven by inspect.signature, so an action gets only what it declares."""

    @pytest.mark.asyncio
    async def test_action_receives_only_the_parameters_it_declares(self, deps, monkeypatch):
        """An action taking just llm_task_manager is not handed llms, context, or events."""
        captured = {}

        async def narrow_action(llm_task_manager):
            captured["llm_task_manager"] = llm_task_manager
            return RailOutcome.allow()

        monkeypatch.setattr(TOPIC_SAFETY_ACTION, narrow_action)

        result = await compile_rail(TOPIC_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert result.is_safe
        assert captured == {"llm_task_manager": deps.llm_task_manager}

    @pytest.mark.asyncio
    async def test_events_are_supplied_to_actions_that_declare_them(self, deps, monkeypatch):
        """topic_safety_check_input declares events, so it receives synthesized ones."""
        action = RecordingAction()
        monkeypatch.setattr(TOPIC_SAFETY_ACTION, action)

        await compile_rail(TOPIC_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert action.kwargs["events"] == [{"type": "UserMessage", "text": "hello there"}]

    @pytest.mark.asyncio
    async def test_http_client_is_supplied_to_vendor_actions(self, deps, monkeypatch):
        """jailbreak_detection_model declares http_client and receives the managed one."""
        action = RecordingAction()
        monkeypatch.setattr("nemoguardrails.library.jailbreak_detection.actions.jailbreak_detection_model", action)

        await compile_rail(JAILBREAK_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert action.kwargs["http_client"] is deps.http_client


class TestOutcomeMapping:
    """RailOutcome maps onto RailResult field for field, inventing nothing."""

    @pytest.mark.asyncio
    async def test_allow_becomes_a_safe_result(self, deps, monkeypatch):
        """ALLOW yields is_safe=True and carries the metadata into return_value."""
        monkeypatch.setattr(
            CONTENT_SAFETY_ACTION, RecordingAction(RailOutcome.allow(metadata={"policy_violations": []}))
        )

        result = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert result.is_safe is True
        assert result.return_value == {"allowed": True, "policy_violations": []}

    @pytest.mark.asyncio
    async def test_block_becomes_an_unsafe_result(self, deps, monkeypatch):
        """BLOCK yields is_safe=False and carries the evidence into return_value."""
        outcome = RailOutcome.block(metadata={"policy_violations": ["S1: Violence"]})
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction(outcome))

        result = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert result.is_safe is False
        assert result.return_value == {"allowed": False, "policy_violations": ["S1: Violence"]}

    @pytest.mark.asyncio
    async def test_reason_is_passed_through_untouched(self, deps, monkeypatch):
        """A rail that supplies a reason keeps it verbatim; the engine never rewrites it."""
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction(RailOutcome.block(reason="policy 4 tripped")))

        result = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert result.reason == "policy 4 tripped"

    @pytest.mark.asyncio
    async def test_absent_reason_stays_absent(self, deps, monkeypatch):
        """A rail that supplies no reason yields reason=None rather than invented text."""
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction(RailOutcome.block()))

        result = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert result.reason is None

    @pytest.mark.asyncio
    async def test_non_outcome_return_is_rejected(self, deps, monkeypatch):
        """An action returning something other than RailOutcome fails closed, not silently."""
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction(outcome="not an outcome"))

        result = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert result.is_safe is False


class TestMessagesToEvents:
    """The message-to-event mapper feeds actions that read conversation history.

    Pinned from both ends: these shapes are what ``llm/filters.py:263-281`` consumes, so a
    change to that vocabulary must break here rather than silently drop topic safety's
    history.
    """

    def test_user_message_becomes_a_user_event(self):
        """A user turn maps to the UserMessage shape to_chat_messages reads."""
        assert messages_to_events([{"role": "user", "content": "hi"}]) == [{"type": "UserMessage", "text": "hi"}]

    def test_assistant_message_becomes_an_utterance_event(self):
        """An assistant turn maps to StartUtteranceBotAction with its script."""
        events = messages_to_events([{"role": "assistant", "content": "hello"}])

        assert events == [{"type": "StartUtteranceBotAction", "script": "hello"}]

    def test_system_message_becomes_a_system_event(self):
        """A system turn maps to SystemMessage with its content."""
        events = messages_to_events([{"role": "system", "content": "be brief"}])

        assert events == [{"type": "SystemMessage", "content": "be brief"}]

    def test_contentless_turn_is_skipped(self):
        """An assistant tool-call turn has no content and is dropped rather than crashing."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
        ]

        assert messages_to_events(messages) == [{"type": "UserMessage", "text": "hi"}]

    def test_order_is_preserved(self):
        """Turns keep conversation order so history reads correctly."""
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]

        assert [event["type"] for event in messages_to_events(messages)] == [
            "SystemMessage",
            "UserMessage",
            "StartUtteranceBotAction",
            "UserMessage",
        ]


class TestModelCallCapture:
    """Model calls are captured from a per-rail sink, so attribution cannot leak."""

    @pytest.mark.asyncio
    async def test_a_rail_that_makes_no_model_call_reports_none(self, deps, monkeypatch):
        """A vendor rail that never reaches a model produces no captured call."""
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction())

        execution = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).execute(USER_MESSAGES)

        assert execution.llm_calls == ()

    @pytest.mark.asyncio
    async def test_a_model_call_is_captured_with_its_task_label(self, deps, monkeypatch, record_llm_call):
        """A rail that calls a model captures that call's LLMCallInfo."""

        async def calling_action(**kwargs):
            record_llm_call(task="content_safety_check_input $model=content_safety")
            return RailOutcome.allow()

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, calling_action)

        execution = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).execute(USER_MESSAGES)

        assert len(execution.llm_calls) == 1
        assert execution.llm_calls[0].task == "content_safety_check_input $model=content_safety"

    @pytest.mark.asyncio
    async def test_a_model_free_rail_does_not_inherit_the_previous_rails_call(self, deps, monkeypatch, record_llm_call):
        """Running a model-free rail straight after a model-backed one captures nothing.

        This is the misattribution regression the whole sink design exists to prevent: with
        a post-hoc contextvar read the second rail would report the first rail's task,
        token counts, and model name as its own.
        """

        async def calling_action(**kwargs):
            record_llm_call(task="content_safety_check_input $model=content_safety")
            return RailOutcome.allow()

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, calling_action)
        monkeypatch.setattr(TOPIC_SAFETY_ACTION, RecordingAction())

        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).execute(USER_MESSAGES)
        second = await compile_rail(TOPIC_SAFETY_INPUT, RailDirection.INPUT, deps).execute(USER_MESSAGES)

        assert second.llm_calls == ()


class TestFailsClosed:
    """A rail that raises is handled by the shared envelope, not by CompiledRail."""

    @pytest.mark.asyncio
    async def test_action_error_blocks(self, deps, monkeypatch):
        """An exception inside the action becomes a blocking result with a redacted reason."""

        async def exploding_action(**kwargs):
            raise RuntimeError("parser blew up")

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, exploding_action)

        result = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert result == RailResult(is_safe=False, reason="content safety check input error: parser blew up")

    @pytest.mark.asyncio
    async def test_status_bearing_error_propagates(self, deps, monkeypatch):
        """A provider 503 leaves the rail rather than being reported as a guardrail block."""
        from nemoguardrails.exceptions import LLMCallException

        async def failing_action(**kwargs):
            raise LLMCallException("upstream refused", status=503)

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, failing_action)

        with pytest.raises(LLMCallException):
            await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)


# Invariant 8 (no Colang runtime in the rail path) is asserted in
# tests/llm/test_call_import_graph.py, next to the static import walker that can actually
# observe it. A sys.modules check here would pass vacuously: importing nemoguardrails at all
# loads both Colang runtimes through its __init__, so every module trivially "has" them.
