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

"""Tests for the typed StreamError model and the buffer-skip path."""

import json

import pytest

from nemoguardrails.rails.llm.buffer import RollingBuffer
from nemoguardrails.stream_errors import (
    StreamError,
    StreamErrorCode,
    StreamErrorDetails,
    StreamErrorType,
    parse_stream_error_chunk,
    stream_error_to_chunk,
)


class TestStreamErrorEnums:
    """Lock down the on-the-wire vocabulary that downstream consumers rely on."""

    def test_type_values_match_historical_strings(self):
        # These string values previously appeared as inline literals in both
        # IORails and LLMRails. They must stay identical so existing consumers
        # parsing the JSON shape do not break.
        assert StreamErrorType.GENERATION_ERROR.value == "generation_error"
        assert StreamErrorType.GUARDRAILS_VIOLATION.value == "guardrails_violation"
        assert StreamErrorType.INTERNAL_ERROR.value == "internal_error"

    def test_code_values_match_historical_strings(self):
        assert StreamErrorCode.GENERATION_FAILED.value == "generation_failed"
        assert StreamErrorCode.CONTENT_BLOCKED.value == "content_blocked"
        assert StreamErrorCode.RAIL_EXECUTION_FAILURE.value == "rail_execution_failure"

    def test_enums_are_str_subclasses(self):
        # Enum values are used directly in JSON serialization, so they must
        # behave like strings (so model_dump_json emits the bare value rather
        # than an enum repr).
        assert isinstance(StreamErrorType.GENERATION_ERROR, str)
        assert isinstance(StreamErrorCode.CONTENT_BLOCKED, str)


class TestStreamErrorSerialization:
    """The serialized shape is the public contract, so pin it explicitly."""

    def test_to_chunk_matches_legacy_shape_with_param(self):
        err = StreamError.content_blocked("Blocked by foo rails.", param="foo")
        payload = json.loads(err.to_chunk())
        assert payload == {
            "error": {
                "message": "Blocked by foo rails.",
                "type": "guardrails_violation",
                "code": "content_blocked",
                "param": "foo",
            }
        }

    def test_to_chunk_includes_null_param_when_absent(self):
        err = StreamError.generation_failed("LLM exploded")
        payload = json.loads(err.to_chunk())
        # Backwards compatibility: legacy IORails payloads omitted ``param``,
        # but JSON-with-explicit-null is parse-compatible for OpenAI-style
        # consumers that .get("param") and tolerate None.
        assert payload["error"]["message"] == "LLM exploded"
        assert payload["error"]["type"] == "generation_error"
        assert payload["error"]["code"] == "generation_failed"
        assert payload["error"]["param"] is None

    def test_str_equals_to_chunk(self):
        err = StreamError.rail_execution_failure("Boom", param="rail x")
        assert str(err) == err.to_chunk()

    def test_stream_error_to_chunk_helper_matches_method(self):
        err = StreamError.content_blocked("hi")
        assert stream_error_to_chunk(err) == err.to_chunk()


class TestStreamErrorFromErrorDict:
    """``from_error_dict`` is the bridge from ``extract_error_json`` output."""

    def test_full_dict_round_trips(self):
        original = {
            "error": {
                "message": "boom",
                "type": "guardrails_violation",
                "code": "content_blocked",
                "param": "flow-id",
            }
        }
        err = StreamError.from_error_dict(original)
        assert json.loads(err.to_chunk()) == original

    def test_partial_dict_fills_defaults(self):
        # ``extract_error_json`` returns ``{"error": {"message": ...}}`` for
        # non-JSON upstream messages. The model must accept that shape and
        # supply sensible default type/code values rather than crashing.
        err = StreamError.from_error_dict({"error": {"message": "network down"}})
        payload = json.loads(err.to_chunk())
        assert payload["error"]["message"] == "network down"
        assert payload["error"]["type"] == StreamErrorType.INTERNAL_ERROR.value
        assert payload["error"]["code"] == StreamErrorCode.GENERATION_FAILED.value

    def test_unknown_keys_preserved(self):
        # OpenAI-style upstream errors sometimes include extra fields like
        # ``request_id``. ``extra="allow"`` keeps them so downstream consumers
        # can still see them.
        original = {
            "error": {
                "message": "boom",
                "type": "internal_error",
                "code": "generation_failed",
                "request_id": "req_abc",
            }
        }
        err = StreamError.from_error_dict(original)
        payload = json.loads(err.to_chunk())
        assert payload["error"].get("request_id") == "req_abc"

    def test_non_dict_input_yields_unknown_error(self):
        err = StreamError.from_error_dict({"error": "string-not-dict"})
        payload = json.loads(err.to_chunk())
        # Falls back to a sane default rather than raising.
        assert payload["error"]["message"] == "Unknown error"

    def test_completely_empty_input(self):
        err = StreamError.from_error_dict({})
        payload = json.loads(err.to_chunk())
        assert payload["error"]["message"] == "Unknown error"


class TestParseStreamErrorChunk:
    """Round-trip detection via parse_stream_error_chunk -- this is what the
    RollingBuffer and consumers use to distinguish error chunks from text.
    """

    def test_parses_serialized_stream_error(self):
        original = StreamError.content_blocked("nope", param="rail")
        recovered = parse_stream_error_chunk(original.to_chunk())
        assert recovered is not None
        assert recovered.error.message == "nope"
        assert recovered.error.code == "content_blocked"
        assert recovered.error.param == "rail"

    def test_returns_none_for_plain_text(self):
        assert parse_stream_error_chunk("Hello world") is None
        assert parse_stream_error_chunk("") is None
        assert parse_stream_error_chunk(" ") is None

    def test_returns_none_for_non_error_json(self):
        # A JSON object that happens to start with ``{`` but is not an error
        # envelope must not be misclassified.
        assert parse_stream_error_chunk('{"text": "hello"}') is None

    def test_returns_none_for_malformed_json(self):
        assert parse_stream_error_chunk("{not json") is None
        assert parse_stream_error_chunk('{"error": ') is None

    def test_accepts_dict_input(self):
        recovered = parse_stream_error_chunk(
            {"error": {"message": "x", "type": "internal_error", "code": "generation_failed"}}
        )
        assert recovered is not None
        assert recovered.error.type == "internal_error"

    def test_passthrough_stream_error_instance(self):
        original = StreamError.generation_failed("x")
        assert parse_stream_error_chunk(original) is original

    def test_returns_none_for_non_string_non_dict(self):
        assert parse_stream_error_chunk(42) is None
        assert parse_stream_error_chunk(None) is None
        assert parse_stream_error_chunk(["error", "list"]) is None

    def test_leading_whitespace_tolerated(self):
        chunk = "   " + StreamError.generation_failed("x").to_chunk()
        assert parse_stream_error_chunk(chunk) is not None


class TestRollingBufferSkipsStreamErrors:
    """Integration: the buffer must keep StreamError chunks out of the
    processing context so they cannot pollute output rails.
    """

    @pytest.mark.asyncio
    async def test_error_chunk_yielded_with_empty_processing_context(self):
        """A standalone error chunk arrives in its own batch with no context."""
        error_chunk = stream_error_to_chunk(StreamError.generation_failed("LLM died"))

        async def stream():
            yield error_chunk

        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        batches = [b async for b in buf.process_stream(stream())]

        assert len(batches) == 1
        assert batches[0].user_output_chunks == [error_chunk]
        # The crucial invariant: error chunks NEVER appear in processing_context
        # since that is what flows into output rails.
        assert batches[0].processing_context == []

    @pytest.mark.asyncio
    async def test_error_chunk_after_content_flushes_pending_text(self):
        """An error mid-stream still surfaces prior text, but the error is its
        own batch and the prior text is not concatenated with the error JSON
        inside ``processing_context``.
        """
        error_chunk = stream_error_to_chunk(StreamError.generation_failed("LLM died"))

        async def stream():
            yield "Hello"
            yield " "
            yield "world"
            yield error_chunk

        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        batches = [b async for b in buf.process_stream(stream())]

        # At least one batch flushes the prior text, then exactly one batch
        # carries the error chunk on its own.
        text_batches = [b for b in batches if b.user_output_chunks and error_chunk not in b.user_output_chunks]
        error_batches = [b for b in batches if error_chunk in b.user_output_chunks]

        assert error_batches, "expected an error batch"
        assert len(error_batches) == 1
        assert error_batches[0].processing_context == []

        flushed_text = "".join(c for batch in text_batches for c in batch.user_output_chunks)
        assert flushed_text == "Hello world"

    @pytest.mark.asyncio
    async def test_regular_chunks_unaffected(self):
        """Sanity: the new code path does not change behavior for plain text."""

        async def stream():
            for word in ["Hello", " ", "world", "!"]:
                yield word

        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        batches = [b async for b in buf.process_stream(stream())]

        all_user_chunks = [c for batch in batches for c in batch.user_output_chunks]
        assert "".join(all_user_chunks) == "Hello world!"

        # No batch should have an empty processing_context for plain text.
        for batch in batches:
            if batch.user_output_chunks:
                assert batch.processing_context, "plain-text batches should still carry context"


class TestStreamErrorDetailsExtraAllow:
    """``extra=allow`` keeps unknown error fields rather than raising, which
    matters when ``extract_error_json`` recovers an OpenAI-style payload.
    """

    def test_extra_field_round_trip(self):
        details = StreamErrorDetails(
            message="m",
            type="t",
            code="c",
            request_id="rid",
        )
        assert details.model_dump()["request_id"] == "rid"
