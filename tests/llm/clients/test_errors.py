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

import pytest

from nemoguardrails.exceptions import LLMBadRequestError, LLMCallException, LLMClientError
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.llm.clients._errors import _redact_secrets, as_client_error, client_facing_message


class TestAsClientError:
    """Unwrapping the client error out of whatever wrapped it."""

    def test_bare_client_error_is_returned(self):
        """An LLMClientError that was never wrapped resolves to itself."""
        error = LLMBadRequestError(400, "bad request")

        assert as_client_error(error) is error

    def test_single_wrapper_is_unwrapped(self):
        """The LLMRails path wraps the client error once, in LLMCallException."""
        error = LLMBadRequestError(400, "bad request")

        assert as_client_error(LLMCallException(error, status=400)) is error

    def test_two_wrappers_are_unwrapped(self):
        """The IORails library-rail path nests twice: LLMCallException -> ModelEngineError -> client error.

        Stopping at the first link would null the provider's code and param for every rail
        that reaches its model through ``llm_call``.
        """
        error = LLMBadRequestError(400, "bad request")
        engine_error = ModelEngineError("HTTP 400 from model 'm'", model_name="m", status=400, inner_exception=error)

        assert as_client_error(LLMCallException(engine_error, status=400)) is error

    def test_wrapper_without_a_client_error_resolves_to_none(self):
        """A transport failure that could not be classified carries no client error."""
        engine_error = ModelEngineError("something failed", model_name="m")

        assert as_client_error(LLMCallException(engine_error)) is None

    def test_unrelated_exception_resolves_to_none(self):
        """An exception with no inner_exception attribute at all is not a client error."""
        assert as_client_error(RuntimeError("boom")) is None

    def test_string_inner_exception_resolves_to_none(self):
        """LLMCallException accepts a str inner_exception, which must not break the walk."""
        assert as_client_error(LLMCallException("upstream refused")) is None

    def test_self_referencing_chain_terminates(self):
        """A wrapper pointing at itself must not spin the walk forever."""

        class SelfReferencing(Exception):
            inner_exception: Exception

        exc = SelfReferencing()
        exc.inner_exception = exc

        assert as_client_error(exc) is None

    def test_message_prefers_the_nested_client_error(self):
        """Through two wrappers the caller still sees the provider's message, not str(exc)."""
        error = LLMClientError(404, "model not found", model_name="internal-model")
        engine_error = ModelEngineError(
            "HTTP 404 from model 'internal-model'", model_name="internal-model", status=404, inner_exception=error
        )

        message = client_facing_message(LLMCallException(engine_error, status=404))

        assert message == "model not found"
        assert "internal-model" not in message


class TestRedactSecrets:
    @pytest.mark.parametrize(
        "raw,redacted_marker",
        [
            ("Auth failed for sk-proj-AbCdEfG12345", "sk-***"),
            ("Auth failed for sk-ant-api03-AbCdEfG", "sk-***"),
            ("Token: nvapi-XYZ_abc123", "nvapi-***"),
            ("Authorization: Bearer eyJhbGciOiJIUzI1Ni", "Bearer ***"),
            ("Google key AIzaSyD-x9K8z7QWERTYuiopasdfgh-1234567890 leaked", "AIza***"),
        ],
    )
    def test_known_prefixes_redacted(self, raw, redacted_marker):
        out = _redact_secrets(raw)
        assert redacted_marker in out

    def test_aiza_lowercase_also_caught(self):
        out = _redact_secrets("aizasydddd-key")
        assert "***" in out
        assert "aizasydddd-key" not in out

    def test_no_secrets_unchanged(self):
        msg = "Model returned an empty response"
        assert _redact_secrets(msg) == msg

    def test_partial_key_not_leaked(self):
        out = _redact_secrets("Auth failed for sk-proj-AbCdEfG12345")
        assert "AbCdEfG12345" not in out
