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

from nemoguardrails.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMUnsupportedParamsError,
)
from tests.llm.clients._helpers import make_client, mock_httpx_post, ok_response

_HINT_FRAGMENT = "If you upgraded from 0.21"
_FRAMEWORK_ENV = "NEMOGUARDRAILS_LLM_FRAMEWORK=langchain"


class TestMigrationHintAppendedOnUnknownParam400:
    @pytest.mark.asyncio
    async def test_unknown_parameter_message_triggers_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [(400, {"error": {"message": "Unknown parameter: 'streaming'"}}, {})],
        )
        with pytest.raises(LLMUnsupportedParamsError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        message = exc_info.value.error_message
        assert "Unknown parameter: 'streaming'" in message
        assert _HINT_FRAGMENT in message
        assert _FRAMEWORK_ENV in message

    @pytest.mark.asyncio
    async def test_unrecognized_token_triggers_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [(400, {"error": {"message": "Field 'verbose' is unrecognized"}}, {})],
        )
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert _HINT_FRAGMENT in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_extra_fields_triggers_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [(400, {"error": {"message": "Request contains extra fields: callbacks"}}, {})],
        )
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert _HINT_FRAGMENT in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_additional_properties_triggers_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [(400, {"error": {"message": "Additional properties are not allowed (cache, tags)"}}, {})],
        )
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert _HINT_FRAGMENT in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_is_not_allowed_triggers_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [(400, {"error": {"message": "field 'metadata' is not allowed here"}}, {})],
        )
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert _HINT_FRAGMENT in exc_info.value.error_message


class TestMigrationHintNotAppendedOnUnrelated400:
    @pytest.mark.asyncio
    async def test_generic_400_no_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [(400, {"error": {"message": "Invalid value for temperature"}}, {})],
        )
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert _HINT_FRAGMENT not in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_context_window_400_no_hint(self):
        from nemoguardrails.exceptions import LLMContextWindowError

        client = make_client()
        mock_httpx_post(
            client,
            [(400, {"error": {"message": "maximum context length exceeded"}}, {})],
        )
        with pytest.raises(LLMContextWindowError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert _HINT_FRAGMENT not in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_rate_limit_shaped_400_no_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [(400, {"error": {"message": "Quota exceeded for this organization"}}, {})],
        )
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert _HINT_FRAGMENT not in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_auth_shaped_401_no_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [(401, {"error": {"message": "Invalid API key"}}, {})],
        )
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert _HINT_FRAGMENT not in exc_info.value.error_message


class TestNoEnrichmentOn200:
    @pytest.mark.asyncio
    async def test_200_response_unchanged(self):
        from nemoguardrails.llm.clients.base import HTTPResponse

        client = make_client()
        mock_httpx_post(client, [(200, ok_response(), {})])
        result = await client.chat_completion("gpt-4o", [])
        assert isinstance(result, HTTPResponse)
        assert result.status_code == 200
        assert result.body["choices"][0]["message"]["content"] == "Hello"


class TestPreservesOriginalProviderError:
    @pytest.mark.asyncio
    async def test_appended_hint_does_not_replace_original(self):
        client = make_client()
        original_message = "Unknown parameter: 'nvidia_api_key'"
        mock_httpx_post(
            client,
            [(400, {"error": {"message": original_message}}, {})],
        )
        with pytest.raises(LLMUnsupportedParamsError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        message = exc_info.value.error_message
        assert original_message in message
        assert _HINT_FRAGMENT in message
        assert message.index(original_message) < message.index(_HINT_FRAGMENT)
