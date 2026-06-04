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

"""Unit tests for content-capture helpers in nemoguardrails.guardrails.telemetry.

Covers the helpers introduced for OTEL GenAI content capture:
``is_content_capture_enabled``, ``_use_json_span_format``,
``_system_parts_from_messages``,
``_non_system_input_messages``, the two ``_set_llm_call_content_*``
branches, the ``set_llm_call_content`` dispatcher, and ``set_rail_content``.
"""

import json
from typing import Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.guardrails.telemetry import (
    _non_system_input_messages,
    _set_llm_call_content_events,
    _set_llm_call_content_json,
    _system_parts_from_messages,
    _use_json_span_format,
    is_content_capture_enabled,
    set_llm_call_content,
    set_rail_content,
    set_request_content,
)
from nemoguardrails.rails.llm.config import TracingConfig
from nemoguardrails.tracing.constants import (
    EventNames,
    GenAIAttributes,
    GuardrailsAttributes,
    OtelContentCapture,
)


@pytest.fixture(autouse=True)
def _clear_otel_envvars(monkeypatch):
    """Strip any inherited OTEL env vars so each test starts from a clean slate.

    The CI/dev shell may have ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``
    or ``OTEL_SEMCONV_STABILITY_OPT_IN`` set; without this, tests asserting
    "False" / "events branch" would be flaky depending on the runner's env.
    """
    monkeypatch.delenv(OtelContentCapture.CAPTURE_CONTENT_ENV, raising=False)
    monkeypatch.delenv(OtelContentCapture.STABILITY_OPT_IN_ENV, raising=False)


@pytest.fixture
def finished_span():
    """Factory that runs a callback inside a real span and returns the finished span.

    Uses an in-memory SDK exporter so tests can assert on real OTEL
    attributes and events instead of mocking ``span.set_attribute`` /
    ``span.add_event``.  Each invocation creates and tears down its own
    provider so tests stay isolated.
    """

    def _run(callback):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test-span") as span:
            callback(span)
        provider.shutdown()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        return spans[0]

    return _run


def _config_with_capture(value: Optional[bool]) -> TracingConfig:
    """Build a ``TracingConfig`` with the given ``enable_content_capture`` flag.

    ``value=None`` returns a default-constructed ``TracingConfig`` (where
    ``enable_content_capture`` defaults to False on the Pydantic model).
    Using the real model rather than a stand-in keeps the tests honest
    against the production type, including the default value contract.
    """
    if value is None:
        return TracingConfig()
    return TracingConfig(enable_content_capture=value)


class TestIsContentCaptureEnabled:
    """Resolution of the content-capture flag: env var wins, config field as fallback."""

    @pytest.mark.parametrize(
        "config_tracing, expected",
        [
            (None, False),  # no config object
            (TracingConfig(), False),  # default → enable_content_capture False
            (_config_with_capture(False), False),
            (_config_with_capture(True), True),
        ],
    )
    def test_config_decides_when_env_unset(self, config_tracing, expected):
        """With no env var, the config field (default False) decides."""
        assert is_content_capture_enabled(config_tracing) is expected

    @pytest.mark.parametrize("config_flag", [None, True, False])
    @pytest.mark.parametrize("env_value", ["true", "True", "TRUE", "1", "  true  "])
    def test_env_truthy_forces_on(self, monkeypatch, env_value, config_flag):
        """Truthy env (case-insensitive, whitespace-stripped) forces capture on regardless of config."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        config = None if config_flag is None else _config_with_capture(config_flag)
        assert is_content_capture_enabled(config) is True

    @pytest.mark.parametrize("config_flag", [None, True, False])
    @pytest.mark.parametrize("env_value", ["false", "False", "FALSE", "0", "  FALSE  "])
    def test_env_falsy_forces_off(self, monkeypatch, env_value, config_flag):
        """Falsy env (case-insensitive, whitespace-stripped) forces capture off regardless of config."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        config = None if config_flag is None else _config_with_capture(config_flag)
        assert is_content_capture_enabled(config) is False

    @pytest.mark.parametrize("env_value", ["yes", "no", "maybe", ""])
    def test_unrecognized_env_falls_through_to_config(self, monkeypatch, env_value):
        """Unrecognized/empty env values are ignored; the config field decides."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        assert is_content_capture_enabled(_config_with_capture(True)) is True
        assert is_content_capture_enabled(_config_with_capture(False)) is False


class TestUseJsonSpanFormat:
    """Parsing of OTEL_SEMCONV_STABILITY_OPT_IN to select JSON attrs vs legacy events."""

    def test_false_when_env_unset(self):
        """No stability opt-in set → legacy-event format (False)."""
        assert _use_json_span_format() is False

    @pytest.mark.parametrize(
        "env_value, expected",
        [
            (OtelContentCapture.STABILITY_OPT_IN_LATEST, True),  # exact token
            (f"http,{OtelContentCapture.STABILITY_OPT_IN_LATEST},db", True),  # token within CSV list
            (f"http,  {OtelContentCapture.STABILITY_OPT_IN_LATEST}  ,db", True),  # CSV token w/ whitespace
            ("gen_ai_legacy", False),  # unrelated token
            ("", False),  # empty value
        ],
    )
    def test_format_selection(self, monkeypatch, env_value, expected):
        """gen_ai_latest_experimental anywhere in the comma-separated token list selects JSON attrs."""
        monkeypatch.setenv(OtelContentCapture.STABILITY_OPT_IN_ENV, env_value)
        assert _use_json_span_format() is expected


_MIXED_MESSAGES = [
    {"role": "system", "content": "you are helpful"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
    {"role": "user", "content": "ok"},
]


class TestSystemPartsFromMessages:
    """Flat-parts extraction of system messages for gen_ai.system_instructions."""

    def test_returns_flat_parts_for_system_only(self):
        """System messages are returned as bare {type,content} parts, no role wrapper."""
        result = _system_parts_from_messages(_MIXED_MESSAGES)
        assert result == [{"type": "text", "content": "you are helpful"}]

    def test_empty_when_no_system_messages(self):
        """No system message → empty parts list."""
        messages = [{"role": "user", "content": "hi"}]
        assert _system_parts_from_messages(messages) == []

    def test_multiple_system_messages_preserves_order(self):
        """Multiple system messages are returned in their original order."""
        messages = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "ignored"},
            {"role": "system", "content": "second"},
        ]
        result = _system_parts_from_messages(messages)
        assert [p["content"] for p in result] == ["first", "second"]

    def test_skips_entries_missing_role_or_content(self):
        """Malformed entries missing role or content are skipped without crashing."""
        messages = [
            {"content": "no role"},
            {"role": "system"},
            {"role": "system", "content": "valid"},
        ]
        result = _system_parts_from_messages(messages)
        # Only the well-formed system message survives
        assert result == [{"type": "text", "content": "valid"}]


class TestNonSystemInputMessages:
    """Role-wrapped extraction of non-system messages for gen_ai.input.messages."""

    def test_returns_role_wrapped_non_system(self):
        """Non-system messages are returned role-wrapped with a single text part each."""
        result = _non_system_input_messages(_MIXED_MESSAGES)
        assert result == [
            {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
            {"role": "assistant", "parts": [{"type": "text", "content": "hi there"}]},
            {"role": "user", "parts": [{"type": "text", "content": "ok"}]},
        ]

    def test_empty_when_only_system_messages(self):
        """A messages list of only system entries yields an empty result."""
        messages = [{"role": "system", "content": "x"}]
        assert _non_system_input_messages(messages) == []

    def test_skips_entries_missing_role_or_content(self):
        """Malformed non-system entries missing role or content are skipped silently."""
        messages = [
            {"content": "no role"},
            {"role": "user"},
            {"role": "user", "content": "valid"},
        ]
        result = _non_system_input_messages(messages)
        assert result == [{"role": "user", "parts": [{"type": "text", "content": "valid"}]}]


class TestSetLlmCallContentJsonBranch:
    """Direct tests for the JSON-attributes branch helper."""

    def test_sets_all_three_attributes_when_data_present(self, finished_span):
        """System/input/output content all land on their respective JSON span attributes."""
        span = finished_span(lambda s: _set_llm_call_content_json(s, _MIXED_MESSAGES, "the answer"))
        attrs = span.attributes

        sysinst = json.loads(attrs[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
        assert sysinst == [{"type": "text", "content": "you are helpful"}]

        inputs = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
        assert [m["role"] for m in inputs] == ["user", "assistant", "user"]

        outputs = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert outputs == [{"role": "assistant", "parts": [{"type": "text", "content": "the answer"}]}]

    def test_omits_system_instructions_when_no_system_message(self, finished_span):
        """No system message → the system_instructions attribute is omitted, not set empty.

        Distinguishing "no system instructions" from "empty system instructions"
        is part of the contract — backends should not see an empty JSON array.
        """
        messages = [{"role": "user", "content": "hi"}]
        span = finished_span(lambda s: _set_llm_call_content_json(s, messages, "hello"))
        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in span.attributes

    def test_omits_input_messages_when_only_system(self, finished_span):
        """Only a system message → input.messages omitted, system_instructions set."""
        messages = [{"role": "system", "content": "be brief"}]
        span = finished_span(lambda s: _set_llm_call_content_json(s, messages, "x"))
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in span.attributes

    def test_omits_output_messages_when_output_text_is_none(self, finished_span):
        """output_text=None → output.messages omitted while input.messages is still set."""
        span = finished_span(lambda s: _set_llm_call_content_json(s, _MIXED_MESSAGES, None))
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes

    def test_emits_no_legacy_events(self, finished_span):
        """Cross-branch hygiene: the JSON-attr path must never add span events."""
        span = finished_span(lambda s: _set_llm_call_content_json(s, _MIXED_MESSAGES, "out"))
        assert span.events == ()


class TestSetLlmCallContentEventsBranch:
    """Direct tests for the legacy-events branch helper."""

    def test_emits_one_event_per_supported_role(self, finished_span):
        """Each supported-role message becomes one event, plus a final choice event."""
        span = finished_span(lambda s: _set_llm_call_content_events(s, _MIXED_MESSAGES, "the answer"))
        event_names = [e.name for e in span.events]
        # 4 input events (1 system + 1 assistant + 2 user) + 1 choice
        assert event_names == [
            EventNames.GEN_AI_SYSTEM_MESSAGE,
            EventNames.GEN_AI_USER_MESSAGE,
            EventNames.GEN_AI_ASSISTANT_MESSAGE,
            EventNames.GEN_AI_USER_MESSAGE,
            EventNames.GEN_AI_CHOICE,
        ]

    def test_event_attributes_carry_role_and_content(self, finished_span):
        """A message event carries the message's role and content as attributes."""
        messages = [{"role": "user", "content": "hello"}]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, "hi"))
        user_event = next(e for e in span.events if e.name == EventNames.GEN_AI_USER_MESSAGE)
        assert dict(user_event.attributes) == {"role": "user", "content": "hello"}

    def test_choice_event_carries_assistant_output(self, finished_span):
        """The choice event carries the assistant output text and index 0."""
        messages = [{"role": "user", "content": "hello"}]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, "the answer"))
        choice = next(e for e in span.events if e.name == EventNames.GEN_AI_CHOICE)
        assert dict(choice.attributes) == {
            "index": 0,
            "message.role": "assistant",
            "message.content": "the answer",
        }

    def test_no_choice_event_when_output_text_none(self, finished_span):
        """output_text=None → no choice event is emitted."""
        messages = [{"role": "user", "content": "hello"}]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, None))
        assert all(e.name != EventNames.GEN_AI_CHOICE for e in span.events)

    def test_tool_role_produces_tool_message_event(self, finished_span):
        """tool role maps to gen_ai.tool.message — forward-compatible for tool calling."""
        messages = [
            {"role": "user", "content": "u"},
            {"role": "tool", "content": "tool result"},
            {"role": "assistant", "content": "a"},
        ]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, None))
        names = [e.name for e in span.events]
        assert EventNames.GEN_AI_USER_MESSAGE in names
        assert EventNames.GEN_AI_TOOL_MESSAGE in names
        assert EventNames.GEN_AI_ASSISTANT_MESSAGE in names
        assert len(span.events) == 3

    def test_skips_unsupported_roles_silently(self, finished_span):
        """Roles without a legacy event mapping (e.g. function) are dropped, not errored."""
        messages = [
            {"role": "user", "content": "u"},
            {"role": "function", "content": "should be skipped"},
            {"role": "assistant", "content": "a"},
        ]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, None))
        names = [e.name for e in span.events]
        assert EventNames.GEN_AI_USER_MESSAGE in names
        assert EventNames.GEN_AI_ASSISTANT_MESSAGE in names
        # function role has no mapping — dropped silently
        assert len(span.events) == 2

    def test_skips_messages_missing_role_or_content(self, finished_span):
        """Malformed messages missing role or content are skipped without crashing.

        LLMMessage is typed dict[str, str], so missing fields are not normal
        but the helper must still not crash on them.
        """
        messages = [
            {"content": "no role"},
            {"role": "user"},
            {"role": "user", "content": "valid"},
        ]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, None))
        # Only the valid message produces an event
        assert len(span.events) == 1
        assert span.events[0].name == EventNames.GEN_AI_USER_MESSAGE
        assert dict(span.events[0].attributes)["content"] == "valid"

    def test_sets_no_json_attributes(self, finished_span):
        """Cross-branch hygiene: the events path must never set the JSON attrs."""
        span = finished_span(lambda s: _set_llm_call_content_events(s, _MIXED_MESSAGES, "out"))
        attrs = span.attributes
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attrs
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attrs
        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in attrs


class TestSetLlmCallContentDispatch:
    """End-to-end tests for the dispatcher choosing JSON vs events."""

    def test_uses_events_branch_when_opt_in_unset(self, finished_span):
        """No stability opt-in → dispatcher emits legacy events, no JSON attrs."""
        span = finished_span(lambda s: set_llm_call_content(s, _MIXED_MESSAGES, "out"))
        assert len(span.events) > 0
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes

    def test_uses_json_branch_when_opt_in_set(self, monkeypatch, finished_span):
        """Stability opt-in set → dispatcher emits JSON attrs, no legacy events."""
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            OtelContentCapture.STABILITY_OPT_IN_LATEST,
        )
        span = finished_span(lambda s: set_llm_call_content(s, _MIXED_MESSAGES, "out"))
        assert span.events == ()
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes

    def test_none_span_is_noop(self):
        """Passing span=None is a no-op and raises nothing."""
        set_llm_call_content(None, _MIXED_MESSAGES, "out")


class TestSetRailContent:
    """guardrails.rail.input and guardrails.rail.reason attribute capture on rail spans."""

    def test_sets_only_input_when_reason_is_none(self, finished_span):
        """reason=None → only the rail.input attribute is set, no rail.reason."""
        rail_input = {"messages": [{"role": "user", "content": "hi"}], "bot_response": None}
        span = finished_span(lambda s: set_rail_content(s, rail_input))
        attrs = span.attributes
        assert json.loads(attrs[GuardrailsAttributes.RAIL_INPUT]) == rail_input
        assert GuardrailsAttributes.RAIL_REASON not in attrs

    def test_sets_input_and_reason_when_reason_provided(self, finished_span):
        """A non-None reason sets both rail.input and rail.reason."""
        rail_input = {"messages": [{"role": "user", "content": "hi"}], "bot_response": "out"}
        span = finished_span(lambda s: set_rail_content(s, rail_input, reason="unsafe topic"))
        attrs = span.attributes
        assert json.loads(attrs[GuardrailsAttributes.RAIL_INPUT]) == rail_input
        assert attrs[GuardrailsAttributes.RAIL_REASON] == "unsafe topic"

    def test_none_span_is_noop(self):
        """Passing span=None is a no-op and raises nothing."""
        set_rail_content(None, {"messages": []}, reason="should not raise")


class TestSetRequestContent:
    """guardrails.request.input / guardrails.request.output capture on the SERVER span."""

    def test_sets_input_and_output(self, finished_span):
        """Both attrs are set: input as JSON messages, output as the returned string."""
        messages = [{"role": "user", "content": "hi"}]
        span = finished_span(lambda s: set_request_content(s, messages, "the answer"))
        attrs = span.attributes
        assert json.loads(attrs[GuardrailsAttributes.REQUEST_INPUT]) == messages
        assert attrs[GuardrailsAttributes.REQUEST_OUTPUT] == "the answer"

    def test_omits_output_when_output_text_none(self, finished_span):
        """output_text=None → only request.input is set, request.output is absent."""
        messages = [{"role": "user", "content": "hi"}]
        span = finished_span(lambda s: set_request_content(s, messages, None))
        attrs = span.attributes
        assert GuardrailsAttributes.REQUEST_INPUT in attrs
        assert GuardrailsAttributes.REQUEST_OUTPUT not in attrs

    def test_uses_guardrails_attrs_not_genai(self, finished_span):
        """Request capture uses guardrails.* attrs, never the gen_ai.* names or events."""
        messages = [{"role": "user", "content": "hi"}]
        span = finished_span(lambda s: set_request_content(s, messages, "out"))
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
        assert span.events == ()

    def test_none_span_is_noop(self):
        """Passing span=None is a no-op and raises nothing."""
        set_request_content(None, [{"role": "user", "content": "hi"}], "out")
