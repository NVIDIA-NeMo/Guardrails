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

``rail_error_result`` is the single home for the engine's fail-closed policy: a rail
that raises produces a blocking ``RailResult`` with a redacted reason, unless the
exception carries an upstream HTTP status, in which case it propagates so the server
can map it to the right response code.

The policy previously lived in ``RailAction.run`` and was duplicated in
``ToolRailAction._guarded``; these tests pin it at the extracted helper and at both
call sites, so the coverage survives the deletion of ``RailAction``.
"""

import logging
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.guardrails.rail_guard import rail_error_result
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


# APIEngineError belongs to this set only until the commit that deletes APIEngine; drop
# its parameter there rather than leaving a case that constructs a deleted class.
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


class TestFailsClosed:
    """A rail that raises without an HTTP status blocks rather than propagating."""

    def test_unexpected_exception_returns_a_blocking_result(self):
        """An arbitrary exception becomes RailResult(is_safe=False) instead of propagating."""
        result = rail_error_result(None, ACTION_NAME, RuntimeError("parser blew up"))

        assert result == RailResult(is_safe=False, reason="content safety check input error: parser blew up")

    @status_bearing_types
    def test_status_bearing_exception_without_a_status_blocks(self, make_exc):
        """A connection-level failure carries status=None, so it fails closed rather than propagating."""
        result = rail_error_result(None, ACTION_NAME, make_exc(None))

        assert result.is_safe is False
        assert result.reason is not None
        assert "upstream refused" in result.reason

    def test_reason_redacts_secrets(self):
        """Credentials in an exception message are redacted before reaching the reason."""
        result = rail_error_result(None, ACTION_NAME, RuntimeError("auth rejected token nvapi-abc123secret"))

        assert result.reason == "content safety check input error: auth rejected token nvapi-***"


class TestPropagatesUpstreamStatus:
    """An exception carrying an HTTP status propagates so the server can map it."""

    @status_bearing_types
    def test_exception_with_a_status_is_reraised(self, make_exc):
        """A 503 from the upstream provider propagates rather than becoming a block."""
        exc = make_exc(503)

        with pytest.raises(type(exc)) as excinfo:
            rail_error_result(None, ACTION_NAME, exc)

        assert excinfo.value is exc


class TestLogsAreRedacted:
    """Credentials are kept out of the log on both paths, not just out of the reason.

    The returned reason has always been redacted. The log lines were not, so the same
    exception text was protected in one direction and written verbatim in the other —
    and the propagating path logs without returning a reason at all, so it was the only
    place that text appeared.
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
    """The failure is recorded on the action span on both the blocking and propagating paths."""

    def test_error_recorded_when_blocking(self):
        """A blocked rail still marks its span as errored."""
        span = MagicMock()

        rail_error_result(span, ACTION_NAME, RuntimeError("parser blew up"))

        span.record_exception.assert_called_once()
        span.set_attribute.assert_any_call("error.type", "RuntimeError")

    def test_reraising_leaves_the_recording_to_the_enclosing_span(self):
        """The envelope does not record on the path where it re-raises.

        ``action_span`` records any exception that escapes it, so recording here too would
        double up. Pinned directly on the helper as well as through the composition below,
        so that restoring the unconditional record cannot pass by only fixing one of them.
        """
        span = MagicMock()
        exc = ModelEngineError("upstream refused", model_name="guard-model", status=503)

        with pytest.raises(ModelEngineError):
            rail_error_result(span, ACTION_NAME, exc)

        span.record_exception.assert_not_called()

    def test_absent_span_is_accepted(self):
        """Tracing being disabled yields span=None, which the envelope tolerates."""
        result = rail_error_result(None, ACTION_NAME, RuntimeError("parser blew up"))

        assert result.is_safe is False


class TestSpanRecordsTheFailureExactlyOnce:
    """Composed with a real action span, a failure produces one exception event, not two.

    This is the assertion that matters, because neither layer is wrong on its own: the
    envelope records so a swallowed error is still visible, and ``action_span`` records so an
    escaping one is. Only their composition reveals the duplicate, which is why the standalone
    helper tests above could not catch it.
    """

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

    def test_exception_becomes_a_blocking_result(self):
        """A malformed tool payload fails closed with the rail's name in the reason."""
        result = DummyToolRail(ValueError("malformed tool payload")).check()

        assert result == RailResult(is_safe=False, reason="tool call validation error: malformed tool payload")

    def test_reason_redacts_secrets(self):
        """Tool rails gain the secret redaction that only RailAction applied before."""
        result = DummyToolRail(ValueError("bad header sk-abc123secret")).check()

        assert result.reason == "tool call validation error: bad header sk-***"

    def test_status_bearing_exception_is_reraised(self):
        """A tool rail is held to the same propagation policy, though no shipped one can raise this."""
        exc = ModelEngineError("upstream refused", model_name="guard-model", status=503)

        with pytest.raises(ModelEngineError):
            DummyToolRail(exc).check()
