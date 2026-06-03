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
    _get_parts_by_role,
    _non_system_input_messages,
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
    Also clears the ``_use_json_span_format`` cache so each test re-reads
    the stability opt-in env var fresh — without this, the first test's
    value would stick across the whole module.
    """
    monkeypatch.delenv(OtelContentCapture.CAPTURE_CONTENT_ENV, raising=False)
    monkeypatch.delenv(OtelContentCapture.STABILITY_OPT_IN_ENV, raising=False)
    _use_json_span_format.cache_clear()


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

    def test_returns_false_when_config_none_and_env_unset(self):
        """No config and no env var → capture off."""
        assert is_content_capture_enabled(None) is False

    def test_returns_true_when_config_flag_set(self):
        """Config enable_content_capture=True with no env var → capture on."""
        assert is_content_capture_enabled(_config_with_capture(True)) is True

    def test_returns_false_when_config_flag_false_and_env_unset(self):
        """Config enable_content_capture=False with no env var → capture off."""
        assert is_content_capture_enabled(_config_with_capture(False)) is False

    def test_returns_false_for_default_tracing_config_and_env_unset(self):
        """Default TracingConfig() has enable_content_capture=False by spec → capture off."""
        assert is_content_capture_enabled(_config_with_capture(None)) is False

    @pytest.mark.parametrize("env_value", ["true", "True", "TRUE", "1"])
    def test_env_var_truthy_values_enable_capture(self, monkeypatch, env_value):
        """Truthy env-var values (case-insensitive 'true' / '1') force capture on."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        assert is_content_capture_enabled(None) is True

    @pytest.mark.parametrize("env_value", ["false", "0", "yes", "no", "", "  "])
    def test_env_var_other_values_do_not_enable_capture(self, monkeypatch, env_value):
        """Falsy and unrecognized env values do not enable capture when config is unset."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        assert is_content_capture_enabled(None) is False

    def test_env_var_takes_effect_when_config_is_false(self, monkeypatch):
        """env=true overrides an explicit config=False (env-var-wins semantic)."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, "true")
        assert is_content_capture_enabled(_config_with_capture(False)) is True

    def test_config_true_overrides_env_unset(self):
        """Config=True with no env var → capture on (config is the fallback signal)."""
        assert is_content_capture_enabled(_config_with_capture(True)) is True

    def test_env_value_with_surrounding_whitespace(self, monkeypatch):
        """Leading/trailing whitespace on the env value is stripped before matching."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, "  true  ")
        assert is_content_capture_enabled(None) is True

    @pytest.mark.parametrize("env_value", ["false", "False", "FALSE", "0"])
    def test_env_var_falsy_disables_capture_even_when_config_true(self, monkeypatch, env_value):
        """Explicit env=false/0 overrides config=True so operators can disable capture fleet-wide."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        assert is_content_capture_enabled(_config_with_capture(True)) is False

    def test_env_var_falsy_with_whitespace_disables_capture(self, monkeypatch):
        """A falsy env value with surrounding whitespace still overrides config=True."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, "  FALSE  ")
        assert is_content_capture_enabled(_config_with_capture(True)) is False

    @pytest.mark.parametrize("env_value", ["yes", "no", "maybe"])
    def test_unrecognized_env_value_falls_through_to_config(self, monkeypatch, env_value):
        """Unrecognized env values are neither enable nor disable signals — config decides."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, env_value)
        assert is_content_capture_enabled(_config_with_capture(True)) is True
        assert is_content_capture_enabled(_config_with_capture(False)) is False


class TestUseJsonSpanFormat:
    """Parsing of OTEL_SEMCONV_STABILITY_OPT_IN to select JSON attrs vs legacy events."""

    def test_returns_false_when_env_unset(self):
        """No stability opt-in → legacy-event format (False)."""
        assert _use_json_span_format() is False

    def test_returns_true_when_opt_in_token_present(self, monkeypatch):
        """The exact opt-in token alone selects JSON-attribute format."""
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            OtelContentCapture.STABILITY_OPT_IN_LATEST,
        )
        assert _use_json_span_format() is True

    def test_returns_true_when_token_in_csv_list(self, monkeypatch):
        """The opt-in token is recognized within a comma-separated token list."""
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            f"http,{OtelContentCapture.STABILITY_OPT_IN_LATEST},db",
        )
        assert _use_json_span_format() is True

    def test_strips_token_whitespace(self, monkeypatch):
        """Whitespace around individual CSV tokens is stripped before matching."""
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            f"http,  {OtelContentCapture.STABILITY_OPT_IN_LATEST}  ,db",
        )
        assert _use_json_span_format() is True

    def test_returns_false_for_unrelated_token(self, monkeypatch):
        """A different opt-in token does not select JSON format."""
        monkeypatch.setenv(OtelContentCapture.STABILITY_OPT_IN_ENV, "gen_ai_legacy")
        assert _use_json_span_format() is False

    def test_returns_false_for_empty_string(self, monkeypatch):
        """An empty opt-in value selects legacy-event format (False)."""
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
        """include=True keeps only messages whose role equals the target role."""
        result = _get_parts_by_role(_MIXED_MESSAGES, "system", include=True)
        assert result == [{"role": "system", "parts": [{"type": "text", "content": "you are helpful"}]}]

    def test_include_false_returns_non_matching_roles(self):
        """include=False keeps every message whose role differs from the target."""
        result = _get_parts_by_role(_MIXED_MESSAGES, "system", include=False)
        roles = [entry["role"] for entry in result]
        assert roles == ["user", "assistant", "user"]

    def test_empty_messages_returns_empty(self):
        """An empty message list yields an empty result."""
        assert _get_parts_by_role([], "system", include=True) == []

    def test_skips_entries_missing_role(self):
        """Messages without a ``role`` field are skipped silently."""
        messages = [{"content": "no role"}, {"role": "user", "content": "ok"}]
        result = _get_parts_by_role(messages, "user", include=True)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_skips_entries_missing_content(self):
        """Messages without a ``content`` field are skipped silently."""
        messages = [{"role": "user"}, {"role": "user", "content": "ok"}]
        result = _get_parts_by_role(messages, "user", include=True)
        assert len(result) == 1
        assert result[0]["parts"][0]["content"] == "ok"

    def test_no_matches_returns_empty(self):
        """include=True with a role absent from the messages yields an empty result."""
        result = _get_parts_by_role(_MIXED_MESSAGES, "tool", include=True)
        assert result == []

    def test_all_match_when_excluded_role_absent(self):
        """include=False excluding an absent role returns every valid entry."""
        result = _get_parts_by_role(_MIXED_MESSAGES, "tool", include=False)
        assert len(result) == len(_MIXED_MESSAGES)


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
        # 3 input events (1 system + 2 user + 1 assistant = 4) + 1 choice
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

    def test_skips_unsupported_roles_silently(self, finished_span):
        """Roles without a legacy event mapping (e.g. tool) are dropped, not errored.

        IORails doesn't yet emit tool/function events; unknown roles must
        be dropped rather than crashing the capture path.
        """
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
