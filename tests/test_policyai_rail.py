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

import json
from contextlib import asynccontextmanager

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.http import HTTPResponse
from nemoguardrails.http.testing import RecordingHTTPClient
from nemoguardrails.library.policyai import actions as policyai_actions
from nemoguardrails.library.policyai.actions import call_policyai_api
from tests.policyai_fixtures import POLICYAI_SAFE_OUTCOME_KWARGS, POLICYAI_UNSAFE_OUTCOME_KWARGS
from tests.utils import TestChat


class aioresponses:
    def __init__(self):
        self.client = RecordingHTTPClient()
        self.urls = []

    def post(self, url, *, payload=None, status=200, body=None):
        content = body.encode() if body is not None else json.dumps(payload).encode()
        self.client.add_response(HTTPResponse(status_code=status, content=content))
        self.urls.append(url)

    def __enter__(self):
        self.original_resolver = policyai_actions.resolve_http_client

        @asynccontextmanager
        async def resolve(client=None):
            yield client or self.client

        policyai_actions.resolve_http_client = resolve
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        policyai_actions.resolve_http_client = self.original_resolver
        if exc_type is None:
            assert all(request.url in self.urls for request in self.client.requests)


def test_input_safe(monkeypatch):
    """Test that safe input is allowed through."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    config = RailsConfig.from_content(
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"
        """,
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              input:
                flows:
                  - policyai moderation on input
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    with aioresponses() as m:
        # PolicyAI returns SAFE assessment
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "Safe",
                        "severity": 0,
                        "reason": "Content is safe",
                    }
                ]
            },
        )

        _ = chat >> "Hello!"
        _ = chat << "Hello! How can I assist you today?"


def test_input_unsafe(monkeypatch):
    """Test that unsafe input is blocked."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    config = RailsConfig.from_content(
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"
        """,
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              input:
                flows:
                  - policyai moderation on input
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    with aioresponses() as m:
        # PolicyAI returns UNSAFE assessment
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "UNSAFE",
                        "category": "HarmfulContent",
                        "severity": 2,
                        "reason": "Content contains harmful language",
                    }
                ]
            },
        )

        _ = chat >> "some harmful content"
        _ = chat << "I'm sorry, I can't respond to that."


def test_output_safe(monkeypatch):
    """Test that safe output is allowed through."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    config = RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              output:
                flows:
                  - policyai moderation on output
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            " Hello! How can I help you today?",
        ],
    )

    with aioresponses() as m:
        # PolicyAI returns SAFE assessment
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "Safe",
                        "severity": 0,
                        "reason": "Content is safe",
                    }
                ]
            },
        )

        _ = chat >> "Hello!"
        _ = chat << "Hello! How can I help you today?"


def test_output_unsafe(monkeypatch):
    """Test that unsafe output is blocked."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    config = RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              output:
                flows:
                  - policyai moderation on output
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            " I promise you a full refund of $500!",
        ],
    )

    with aioresponses() as m:
        # PolicyAI returns UNSAFE assessment (e.g., unauthorized refund promise)
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "UNSAFE",
                        "category": "UnauthorizedCommitment",
                        "severity": 2,
                        "reason": "AI made unauthorized commitment about refund",
                    }
                ]
            },
        )

        _ = chat >> "Can I get a refund?"
        _ = chat << "I'm sorry, I can't respond to that."


def test_custom_tag_via_env(monkeypatch):
    """Test using a custom policy tag via environment variable."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "custom-tag")

    config = RailsConfig.from_content(
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"
        """,
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              input:
                flows:
                  - policyai moderation on input
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    with aioresponses() as m:
        # Note: The URL should use the custom tag from env var
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/custom-tag",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "Safe",
                        "severity": 0,
                        "reason": "Content is safe",
                    }
                ]
            },
        )

        _ = chat >> "Hello!"
        _ = chat << "Hello! How can I assist you today?"


def test_multiple_policies(monkeypatch):
    """Test when multiple policies are evaluated (tag has multiple policies)."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    config = RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              output:
                flows:
                  - policyai moderation on output
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            " Here's some content",
        ],
    )

    with aioresponses() as m:
        # Multiple policies - first is SAFE, second is UNSAFE
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "ToxicityCheck",
                        "severity": 0,
                        "reason": "No toxic content detected",
                    },
                    {
                        "status": "success",
                        "assessment": "UNSAFE",
                        "category": "PIIDetection",
                        "severity": 1,
                        "reason": "PII detected in content",
                    },
                ]
            },
        )

        _ = chat >> "Hello!"
        # Should be blocked because one policy returned UNSAFE
        _ = chat << "I'm sorry, I can't respond to that."


@pytest.mark.asyncio
async def test_empty_data_array_raises_error(monkeypatch):
    """Test that empty data array (no policies attached to tag) raises an error."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "empty-tag")

    with aioresponses() as m:
        # PolicyAI returns empty data array (no policies attached to tag)
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/empty-tag",
            payload={"data": []},
        )

        with pytest.raises(ValueError) as exc_info:
            await call_policyai_api(text="Hello!")

        assert "no policy results" in str(exc_info.value).lower()
        assert "empty-tag" in str(exc_info.value)


@pytest.mark.asyncio
async def test_api_error_raises_exception(monkeypatch):
    """Test that API errors (non-200 status) raise an exception."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    with aioresponses() as m:
        # PolicyAI returns 500 error
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            status=500,
            body="Internal Server Error",
        )

        with pytest.raises(ValueError) as exc_info:
            await call_policyai_api(text="Hello!")

        assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_all_policies_failed_raises_error(monkeypatch):
    """Test that all policies failing raises an error."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "failing-tag")

    with aioresponses() as m:
        # All policies return failed status
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/failing-tag",
            payload={
                "data": [
                    {
                        "status": "failed",
                        "error": "Policy configuration error",
                    },
                    {
                        "status": "failed",
                        "error": "Policy timeout",
                    },
                ]
            },
        )

        with pytest.raises(ValueError) as exc_info:
            await call_policyai_api(text="Hello!")

        assert "all" in str(exc_info.value).lower()
        assert "failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_missing_api_key_raises_error(monkeypatch):
    """Test that missing API key raises an error."""
    # Ensure POLICYAI_API_KEY is not set
    monkeypatch.delenv("POLICYAI_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc_info:
        await call_policyai_api(text="Hello!")

    assert "POLICYAI_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_text_parameter(monkeypatch):
    """Test handling of empty text parameter."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    with aioresponses() as m:
        # PolicyAI should still process empty/None text
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "Safe",
                        "severity": 0,
                        "reason": "Empty content is safe",
                    }
                ]
            },
        )

        # Test with empty string
        result = await call_policyai_api(text="")
        assert result.is_blocked is False
        assert result.metadata["assessment"] == "SAFE"


@pytest.mark.asyncio
async def test_none_text_parameter(monkeypatch):
    """Test handling of None text parameter."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    with aioresponses() as m:
        # PolicyAI should still process None text
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "Safe",
                        "severity": 0,
                        "reason": "Null content is safe",
                    }
                ]
            },
        )

        # Test with None
        result = await call_policyai_api(text=None)
        assert result.is_blocked is False
        assert result.metadata["assessment"] == "SAFE"


@pytest.mark.asyncio
async def test_partial_policy_failures(monkeypatch):
    """Test that partial policy failures still work if some succeed."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    with aioresponses() as m:
        # Some policies fail, but one succeeds with SAFE
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "failed",
                        "error": "Policy timeout",
                    },
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "ToxicityCheck",
                        "severity": 0,
                        "reason": "No toxic content",
                    },
                ]
            },
        )

        result = await call_policyai_api(text="Hello!")
        assert result.is_blocked is False
        assert result.metadata["assessment"] == "SAFE"
        assert result.reason == POLICYAI_SAFE_OUTCOME_KWARGS["reason"]
        assert result.metadata == {key: value for key, value in POLICYAI_SAFE_OUTCOME_KWARGS.items() if key != "reason"}


@pytest.mark.asyncio
async def test_custom_base_url_with_trailing_slash(monkeypatch):
    """Test that custom base URL with trailing slash is handled correctly."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_BASE_URL", "https://custom.api.example.com/")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    with aioresponses() as m:
        # URL should have trailing slash stripped
        m.post(
            "https://custom.api.example.com/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "Safe",
                        "severity": 0,
                        "reason": "Content is safe",
                    }
                ]
            },
        )

        result = await call_policyai_api(text="Hello!")
        assert result.is_blocked is False
        assert result.metadata["assessment"] == "SAFE"


@pytest.mark.asyncio
async def test_tag_name_parameter_overrides_env(monkeypatch):
    """Test that tag_name parameter overrides environment variable."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "env-tag")

    with aioresponses() as m:
        # Should use parameter tag, not env var tag
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/param-tag",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "Safe",
                        "severity": 0,
                        "reason": "Content is safe",
                    }
                ]
            },
        )

        result = await call_policyai_api(text="Hello!", tag_name="param-tag")
        assert result.is_blocked is False
        assert result.metadata["assessment"] == "SAFE"


@pytest.mark.asyncio
async def test_unsafe_with_missing_fields(monkeypatch):
    """Test UNSAFE response with missing optional fields uses defaults."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.setenv("POLICYAI_TAG_NAME", "test")

    with aioresponses() as m:
        # UNSAFE response without category, severity, or reason
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/test",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "UNSAFE",
                    }
                ]
            },
        )

        result = await call_policyai_api(text="Bad content")
        assert result.is_blocked is True
        assert result.metadata["assessment"] == "UNSAFE"
        assert result.metadata["category"] == "Unknown"
        assert result.metadata["severity"] == 0
        assert result.reason == "Policy violation detected"


def test_policyai_action_metadata_is_registration_only():
    assert set(getattr(call_policyai_api, "action_meta")) == {
        "name",
        "is_system_action",
        "execute_async",
    }


def test_policyai_fixture_metadata_matches_action_shape():
    assert POLICYAI_SAFE_OUTCOME_KWARGS["assessment"] == "SAFE"
    assert POLICYAI_UNSAFE_OUTCOME_KWARGS["assessment"] == "UNSAFE"


@pytest.mark.asyncio
async def test_default_tag_name_prod(monkeypatch):
    """Test that default tag 'prod' is used when env var is not set."""
    monkeypatch.setenv("POLICYAI_API_KEY", "test-api-key")
    monkeypatch.delenv("POLICYAI_TAG_NAME", raising=False)

    with aioresponses() as m:
        # Should use default "prod" tag
        m.post(
            "https://api.musubilabs.ai/policyai/v1/decisions/evaluate/prod",
            payload={
                "data": [
                    {
                        "status": "success",
                        "assessment": "SAFE",
                        "category": "Safe",
                        "severity": 0,
                        "reason": "Content is safe",
                    }
                ]
            },
        )

        result = await call_policyai_api(text="Hello!")
        assert result.is_blocked is False
        assert result.metadata["assessment"] == "SAFE"
