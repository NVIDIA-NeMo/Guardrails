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

"""Unit tests for the shared IORails rail error envelope.

Two entry points over one private helper, differing only in return shape. Both are tested
directly, since transitive execution from another module marks the lines covered without
asserting anything about them.
"""

import logging
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.guardrails.rail_guard import rail_error_outcome, rail_error_result
from nemoguardrails.guardrails.telemetry import action_span
from nemoguardrails.guardrails.tool_rail_action import ToolRailAction

ACTION_NAME = "content safety check input"


def recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    """Return a tracer whose finished spans can be inspected, and its exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def exception_events(exporter: InMemorySpanExporter) -> list:
    """Every exception event recorded across the exporter's finished spans."""
    return [event for span in exporter.get_finished_spans() for event in span.events if event.name == "exception"]


status_bearing_types = pytest.mark.parametrize(
    "make_exc",
    [
        pytest.param(
            lambda status: ModelEngineError("upstream refused", model_name="guard-model", status=status),
            id="ModelEngineError",
        ),
        pytest.param(
            lambda status: APIEngineError("upstream refused", endpoint="https://vendor/api", status=status),
            id="APIEngineError",
        ),
        pytest.param(
            lambda status: LLMCallException("upstream refused", status=status),
            id="LLMCallException",
        ),
    ],
)


entry_points = pytest.mark.parametrize(
    "call", [rail_error_result, rail_error_outcome], ids=["rail_error_result", "rail_error_outcome"]
)


def verdict_of(returned) -> tuple[bool, str | None]:
    """Flatten either return shape to ``(blocked, reason)``.

    Lets one test cover both entry points, which pins that they agree by construction rather
    than by a separate parity assertion someone could forget to update.
    """
    if isinstance(returned, RailResult):
        return not returned.is_safe, returned.reason
    return returned.is_blocked, returned.reason


class TestFailsClosed:
    """A rail that raises without an HTTP status blocks, identically at both entry points."""

    @entry_points
    def test_unexpected_exception_blocks_with_a_reason(self, call):
        """An arbitrary exception becomes a blocking verdict naming the action."""
        blocked, reason = verdict_of(call(None, ACTION_NAME, RuntimeError("parser blew up")))

        assert blocked
        assert reason == "content safety check input error: parser blew up"

    @entry_points
    @status_bearing_types
    def test_status_bearing_exception_without_a_status_blocks(self, call, make_exc):
        """A connection-level failure carries status=None, so it fails closed rather than propagating."""
        blocked, reason = verdict_of(call(None, ACTION_NAME, make_exc(None)))

        assert blocked
        assert reason is not None
        assert "upstream refused" in reason

    @entry_points
    def test_reason_redacts_secrets(self, call):
        """Credentials in an exception message are redacted before reaching the reason."""
        _, reason = verdict_of(call(None, ACTION_NAME, RuntimeError("auth rejected token nvapi-abc123secret")))

        assert reason == "content safety check input error: auth rejected token nvapi-***"

    @entry_points
    def test_blocking_verdict_records_the_span_error(self, call):
        """A blocked rail marks its span, so the failure is visible in the trace."""
        span = MagicMock()

        call(span, ACTION_NAME, RuntimeError("parser blew up"))

        span.record_exception.assert_called_once()
        span.set_attribute.assert_any_call("error.type", "RuntimeError")


class TestPropagatesUpstreamStatus:
    """An exception carrying an HTTP status propagates so the server can map it."""

    @entry_points
    @status_bearing_types
    def test_exception_with_a_status_is_reraised(self, call, make_exc):
        """A 503 from the upstream provider propagates rather than becoming a block."""
        exc = make_exc(503)

        with pytest.raises(type(exc)) as excinfo:
            call(None, ACTION_NAME, exc)

        assert excinfo.value is exc


class TestReturnShapes:
    """The entry points differ only in the type they return, asserted whole."""

    def test_tool_rail_entry_point_returns_a_rail_result(self):
        """``rail_error_result`` yields a blocking RailResult."""
        returned = rail_error_result(None, ACTION_NAME, RuntimeError("parser blew up"))

        assert returned == RailResult(is_safe=False, reason="content safety check input error: parser blew up")

    def test_compiled_rail_entry_point_returns_a_rail_outcome(self):
        """``rail_error_outcome`` yields a blocking RailOutcome with no metadata or transforms."""
        returned = rail_error_outcome(None, ACTION_NAME, RuntimeError("parser blew up"))

        assert returned == RailOutcome.block(reason="content safety check input error: parser blew up")


class TestLogsAreRedacted:
    """Credentials are kept out of the log on both paths, not just out of the reason.

    The propagating path logs without returning a reason, so the log line is the only place
    that text appears.
    """

    SECRET = "nvapi-abc123secret"
    REDACTED = "nvapi-***"
    LOGGER = "nemoguardrails.guardrails.rail_guard"

    def test_blocking_path_logs_the_redacted_message(self, caplog):
        """A blocked rail logs the redacted text rather than the raw credential."""
        with caplog.at_level(logging.ERROR, logger=self.LOGGER):
            rail_error_result(None, ACTION_NAME, RuntimeError(f"auth rejected {self.SECRET}"))

        assert self.SECRET not in caplog.text
        assert self.REDACTED in caplog.text

    def test_propagating_path_logs_the_redacted_message(self, caplog):
        """A provider failure carrying a status is logged redacted before it is re-raised."""
        exc = ModelEngineError(f"auth rejected {self.SECRET}", model_name="guard-model", status=401)

        with caplog.at_level(logging.ERROR, logger=self.LOGGER):
            with pytest.raises(ModelEngineError):
                rail_error_result(None, ACTION_NAME, exc)

        assert self.SECRET not in caplog.text
        assert self.REDACTED in caplog.text

    def test_propagated_exception_itself_is_not_rewritten(self, caplog):
        """Redaction applies to the log and the reason, never to the exception that propagates.

        The server maps the original exception to a response, and rewriting its message would
        change what an operator sees in a traceback for no security gain — the client-facing
        payload is sanitised separately.
        """
        exc = ModelEngineError(f"auth rejected {self.SECRET}", model_name="guard-model", status=401)

        with caplog.at_level(logging.ERROR, logger=self.LOGGER):
            with pytest.raises(ModelEngineError) as excinfo:
                rail_error_result(None, ACTION_NAME, exc)

        assert excinfo.value is exc
        assert self.SECRET in str(excinfo.value)


class TestSpanErrorRecording:
    """A failure produces exactly one exception event on the action span, never two.

    Neither layer is wrong alone — the envelope records so a swallowed error stays visible,
    ``action_span`` records so an escaping one does — so only the composition shows the
    duplicate, and no standalone test of either could catch it.
    """

    def test_reraising_leaves_the_recording_to_the_enclosing_span(self):
        """The envelope does not record on the path where it re-raises.

        Pinned on the helper as well as through the compositions below, so restoring the
        unconditional record cannot pass by fixing only one of them.
        """
        span = MagicMock()
        exc = ModelEngineError("upstream refused", model_name="guard-model", status=503)

        with pytest.raises(ModelEngineError):
            rail_error_result(span, ACTION_NAME, exc)

        span.record_exception.assert_not_called()

    def test_blocking_path_records_once(self):
        """A blocked rail records a single exception event on its action span."""
        tracer, exporter = recording_tracer()

        with action_span(tracer, ACTION_NAME) as span:
            result = rail_error_result(span, ACTION_NAME, RuntimeError("parser blew up"))

        assert result.is_safe is False
        events = exception_events(exporter)
        assert len(events) == 1
        assert events[0].attributes["exception.type"] == "RuntimeError"

    def test_propagating_path_records_once(self):
        """A re-raised provider failure is recorded by the span alone, not twice."""
        tracer, exporter = recording_tracer()
        exc = ModelEngineError("upstream refused", model_name="guard-model", status=503)

        with pytest.raises(ModelEngineError):
            with action_span(tracer, ACTION_NAME) as span:
                rail_error_result(span, ACTION_NAME, exc)

        events = exception_events(exporter)
        assert len(events) == 1
        # OTEL qualifies non-builtin exception names, so match the suffix rather than
        # coupling this assertion to the module ModelEngineError happens to live in.
        assert events[0].attributes["exception.type"].endswith("ModelEngineError")

    def test_propagating_path_still_marks_the_span_errored(self):
        """Dropping the envelope's own record must not cost the error.type attribute."""
        tracer, exporter = recording_tracer()
        exc = ModelEngineError("upstream refused", model_name="guard-model", status=503)

        with pytest.raises(ModelEngineError):
            with action_span(tracer, ACTION_NAME) as span:
                rail_error_result(span, ACTION_NAME, exc)

        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert finished.attributes["error.type"] == "ModelEngineError"


class DummyToolRail(ToolRailAction):
    """Minimal tool rail whose guarded check raises whatever it is handed."""

    action_name = "tool call validation"

    def __init__(self, error=None):
        super().__init__()
        self._error = error

    def check(self) -> RailResult:
        """Run the guarded check, raising the configured error when there is one."""

        def _check() -> RailResult:
            if self._error is not None:
                raise self._error
            return RailResult(is_safe=True)

        return self._guarded(_check)


class TestToolRailsShareTheEnvelope:
    """ToolRailAction routes through the same helper rather than keeping its own copy."""

    def test_successful_check_is_returned_unchanged(self):
        """A check that returns normally is passed through untouched."""
        assert DummyToolRail().check() == RailResult(is_safe=True)

    def test_exception_becomes_a_blocking_result_with_secrets_redacted(self):
        """A malformed payload fails closed, named after the rail and with credentials removed."""
        result = DummyToolRail(ValueError("bad header sk-abc123secret")).check()

        assert result == RailResult(is_safe=False, reason="tool call validation error: bad header sk-***")

    def test_status_bearing_exception_is_reraised(self):
        """A tool rail is held to the same propagation policy, though no shipped one can raise this."""
        exc = ModelEngineError("upstream refused", model_name="guard-model", status=503)

        with pytest.raises(ModelEngineError):
            DummyToolRail(exc).check()
