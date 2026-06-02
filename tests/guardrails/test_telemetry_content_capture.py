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
``_get_parts_by_role``, ``_system_parts_from_messages``,
``_non_system_messages_to_parts``, the two ``_set_llm_call_content_*``
branches, the ``set_llm_call_content`` dispatcher, and ``set_rail_content``.
"""

import json
from typing import Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.guardrails.telemetry import (
    _get_parts_by_role,
    _non_system_messages_to_parts,
    _set_llm_call_content_events,
    _set_llm_call_content_json,
    _system_parts_from_messages,
    _use_json_span_format,
    is_content_capture_enabled,
    set_llm_call_content,
    set_rail_content,
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
    """Resolution of the content-capture flag: config field wins, env var as fallback."""

    def test_returns_false_when_config_none_and_env_unset(self):
        assert is_content_capture_enabled(None) is False

    def test_returns_true_when_config_flag_set(self):
        assert is_content_capture_enabled(_config_with_capture(True)) is True

    def test_returns_false_when_config_flag_false_and_env_unset(self):
        assert is_content_capture_enabled(_config_with_capture(False)) is False

    def test_returns_false_for_default_tracing_config_and_env_unset(self):
        # Default TracingConfig() has enable_content_capture=False by spec
        assert is_content_capture_enabled(_config_with_capture(None)) is False

    @pytest.mark.parametrize("env_value", ["true", "True", "TRUE", "1"])
    def test_env_var_truthy_values_enable_capture(self, monkeypatch, env_value):
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        assert is_content_capture_enabled(None) is True

    @pytest.mark.parametrize("env_value", ["false", "0", "yes", "no", "", "  "])
    def test_env_var_other_values_do_not_enable_capture(self, monkeypatch, env_value):
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        assert is_content_capture_enabled(None) is False

    def test_env_var_takes_effect_when_config_unset(self, monkeypatch):
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, "true")
        assert is_content_capture_enabled(_config_with_capture(False)) is True

    def test_config_true_overrides_env_unset(self):
        assert is_content_capture_enabled(_config_with_capture(True)) is True

    def test_env_value_with_surrounding_whitespace(self, monkeypatch):
        # Leading/trailing whitespace gets stripped (.strip() in the helper)
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, "  true  ")
        assert is_content_capture_enabled(None) is True


class TestUseJsonSpanFormat:
    """Parsing of OTEL_SEMCONV_STABILITY_OPT_IN to select JSON attrs vs legacy events."""

    def test_returns_false_when_env_unset(self):
        assert _use_json_span_format() is False

    def test_returns_true_when_opt_in_token_present(self, monkeypatch):
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            OtelContentCapture.STABILITY_OPT_IN_LATEST,
        )
        assert _use_json_span_format() is True

    def test_returns_true_when_token_in_csv_list(self, monkeypatch):
        # Real usage: opt-in env is a comma-separated set of tokens
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            f"http,{OtelContentCapture.STABILITY_OPT_IN_LATEST},db",
        )
        assert _use_json_span_format() is True

    def test_strips_token_whitespace(self, monkeypatch):
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            f"http,  {OtelContentCapture.STABILITY_OPT_IN_LATEST}  ,db",
        )
        assert _use_json_span_format() is True

    def test_returns_false_for_unrelated_token(self, monkeypatch):
        monkeypatch.setenv(OtelContentCapture.STABILITY_OPT_IN_ENV, "gen_ai_legacy")
        assert _use_json_span_format() is False

    def test_returns_false_for_empty_string(self, monkeypatch):
        monkeypatch.setenv(OtelContentCapture.STABILITY_OPT_IN_ENV, "")
        assert _use_json_span_format() is False


_MIXED_MESSAGES = [
    {"role": "system", "content": "you are helpful"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
    {"role": "user", "content": "ok"},
]


class TestGetPartsByRole:
    """Role-filtered walker that powers the two specialised parts helpers."""

    def test_include_true_returns_matching_role(self):
        result = _get_parts_by_role(_MIXED_MESSAGES, "system", include=True)
        assert result == [{"role": "system", "parts": [{"type": "text", "content": "you are helpful"}]}]

    def test_include_false_returns_non_matching_roles(self):
        result = _get_parts_by_role(_MIXED_MESSAGES, "system", include=False)
        roles = [entry["role"] for entry in result]
        assert roles == ["user", "assistant", "user"]

    def test_empty_messages_returns_empty(self):
        assert _get_parts_by_role([], "system", include=True) == []

    def test_skips_entries_missing_role(self):
        messages = [{"content": "no role"}, {"role": "user", "content": "ok"}]
        result = _get_parts_by_role(messages, "user", include=True)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_skips_entries_missing_content(self):
        messages = [{"role": "user"}, {"role": "user", "content": "ok"}]
        result = _get_parts_by_role(messages, "user", include=True)
        assert len(result) == 1
        assert result[0]["parts"][0]["content"] == "ok"

    def test_no_matches_returns_empty(self):
        result = _get_parts_by_role(_MIXED_MESSAGES, "tool", include=True)
        assert result == []

    def test_all_match_when_excluded_role_absent(self):
        # When excluding a role that's not present, every valid entry comes through
        result = _get_parts_by_role(_MIXED_MESSAGES, "tool", include=False)
        assert len(result) == len(_MIXED_MESSAGES)


class TestSystemPartsFromMessages:
    """Flat-parts extraction of system messages for gen_ai.system_instructions."""

    def test_returns_flat_parts_for_system_only(self):
        # gen_ai.system_instructions is a flat list of {type,content} parts
        # — no role wrapping, even though the underlying walker produces it
        result = _system_parts_from_messages(_MIXED_MESSAGES)
        assert result == [{"type": "text", "content": "you are helpful"}]

    def test_empty_when_no_system_messages(self):
        messages = [{"role": "user", "content": "hi"}]
        assert _system_parts_from_messages(messages) == []

    def test_multiple_system_messages_preserves_order(self):
        messages = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "ignored"},
            {"role": "system", "content": "second"},
        ]
        result = _system_parts_from_messages(messages)
        assert [p["content"] for p in result] == ["first", "second"]


class TestNonSystemMessagesToParts:
    """Role-wrapped extraction of non-system messages for gen_ai.input.messages."""

    def test_returns_role_wrapped_non_system(self):
        result = _non_system_messages_to_parts(_MIXED_MESSAGES)
        assert result == [
            {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
            {"role": "assistant", "parts": [{"type": "text", "content": "hi there"}]},
            {"role": "user", "parts": [{"type": "text", "content": "ok"}]},
        ]

    def test_empty_when_only_system_messages(self):
        messages = [{"role": "system", "content": "x"}]
        assert _non_system_messages_to_parts(messages) == []


class TestSetLlmCallContentJsonBranch:
    """Direct tests for the JSON-attributes branch helper."""

    def test_sets_all_three_attributes_when_data_present(self, finished_span):
        span = finished_span(lambda s: _set_llm_call_content_json(s, _MIXED_MESSAGES, "the answer"))
        attrs = span.attributes

        sysinst = json.loads(attrs[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
        assert sysinst == [{"type": "text", "content": "you are helpful"}]

        inputs = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
        assert [m["role"] for m in inputs] == ["user", "assistant", "user"]

        outputs = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert outputs == [{"role": "assistant", "parts": [{"type": "text", "content": "the answer"}]}]

    def test_omits_system_instructions_when_no_system_message(self, finished_span):
        # Distinguishing "no system instructions" from "empty system instructions"
        # is part of the contract — backends should not see an empty JSON array
        messages = [{"role": "user", "content": "hi"}]
        span = finished_span(lambda s: _set_llm_call_content_json(s, messages, "hello"))
        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in span.attributes

    def test_omits_input_messages_when_only_system(self, finished_span):
        messages = [{"role": "system", "content": "be brief"}]
        span = finished_span(lambda s: _set_llm_call_content_json(s, messages, "x"))
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in span.attributes

    def test_omits_output_messages_when_output_text_is_none(self, finished_span):
        span = finished_span(lambda s: _set_llm_call_content_json(s, _MIXED_MESSAGES, None))
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes

    def test_emits_no_legacy_events(self, finished_span):
        span = finished_span(lambda s: _set_llm_call_content_json(s, _MIXED_MESSAGES, "out"))
        # Cross-branch hygiene: JSON-attr path must never add events
        assert span.events == ()


class TestSetLlmCallContentEventsBranch:
    """Direct tests for the legacy-events branch helper."""

    def test_emits_one_event_per_supported_role(self, finished_span):
        span = finished_span(lambda s: _set_llm_call_content_events(s, _MIXED_MESSAGES, "the answer"))
        event_names = [e.name for e in span.events]
        # 3 input events (1 system + 2 user + 1 assistant = 4) + 1 choice
        assert event_names == [
            EventNames.GEN_AI_SYSTEM_MESSAGE,
            EventNames.GEN_AI_USER_MESSAGE,
            EventNames.GEN_AI_ASSISTANT_MESSAGE,
            EventNames.GEN_AI_USER_MESSAGE,
            EventNames.GEN_AI_CHOICE,
        ]

    def test_event_attributes_carry_role_and_content(self, finished_span):
        messages = [{"role": "user", "content": "hello"}]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, "hi"))
        user_event = next(e for e in span.events if e.name == EventNames.GEN_AI_USER_MESSAGE)
        assert dict(user_event.attributes) == {"role": "user", "content": "hello"}

    def test_choice_event_carries_assistant_output(self, finished_span):
        messages = [{"role": "user", "content": "hello"}]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, "the answer"))
        choice = next(e for e in span.events if e.name == EventNames.GEN_AI_CHOICE)
        assert dict(choice.attributes) == {
            "index": 0,
            "message.role": "assistant",
            "message.content": "the answer",
        }

    def test_no_choice_event_when_output_text_none(self, finished_span):
        messages = [{"role": "user", "content": "hello"}]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, None))
        assert all(e.name != EventNames.GEN_AI_CHOICE for e in span.events)

    def test_skips_unsupported_roles_silently(self, finished_span):
        # IORails doesn't yet emit tool/function events; unknown roles must
        # be dropped rather than crashing the capture path
        messages = [
            {"role": "user", "content": "u"},
            {"role": "tool", "content": "should be skipped"},
            {"role": "assistant", "content": "a"},
        ]
        span = finished_span(lambda s: _set_llm_call_content_events(s, messages, None))
        names = [e.name for e in span.events]
        assert EventNames.GEN_AI_USER_MESSAGE in names
        assert EventNames.GEN_AI_ASSISTANT_MESSAGE in names
        # No event was raised for the tool role
        assert len(span.events) == 2

    def test_sets_no_json_attributes(self, finished_span):
        # Cross-branch hygiene: events path must never set the JSON attrs
        span = finished_span(lambda s: _set_llm_call_content_events(s, _MIXED_MESSAGES, "out"))
        attrs = span.attributes
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attrs
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attrs
        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in attrs


class TestSetLlmCallContentDispatch:
    """End-to-end tests for the dispatcher choosing JSON vs events."""

    def test_uses_events_branch_when_opt_in_unset(self, finished_span):
        span = finished_span(lambda s: set_llm_call_content(s, _MIXED_MESSAGES, "out"))
        assert len(span.events) > 0
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes

    def test_uses_json_branch_when_opt_in_set(self, monkeypatch, finished_span):
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            OtelContentCapture.STABILITY_OPT_IN_LATEST,
        )
        span = finished_span(lambda s: set_llm_call_content(s, _MIXED_MESSAGES, "out"))
        assert span.events == ()
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes

    def test_none_span_is_noop(self):
        # No exception even though we pass None for the span and structured input
        set_llm_call_content(None, _MIXED_MESSAGES, "out")


class TestSetRailContent:
    """guardrails.rail.input and guardrails.rail.reason attribute capture on rail spans."""

    def test_sets_only_input_when_reason_is_none(self, finished_span):
        rail_input = {"messages": [{"role": "user", "content": "hi"}], "bot_response": None}
        span = finished_span(lambda s: set_rail_content(s, rail_input))
        attrs = span.attributes
        assert json.loads(attrs[GuardrailsAttributes.RAIL_INPUT]) == rail_input
        assert GuardrailsAttributes.RAIL_REASON not in attrs

    def test_sets_input_and_reason_when_reason_provided(self, finished_span):
        rail_input = {"messages": [{"role": "user", "content": "hi"}], "bot_response": "out"}
        span = finished_span(lambda s: set_rail_content(s, rail_input, reason="unsafe topic"))
        attrs = span.attributes
        assert json.loads(attrs[GuardrailsAttributes.RAIL_INPUT]) == rail_input
        assert attrs[GuardrailsAttributes.RAIL_REASON] == "unsafe topic"

    def test_none_span_is_noop(self):
        # No exception even when we pass real input and reason
        set_rail_content(None, {"messages": []}, reason="should not raise")
