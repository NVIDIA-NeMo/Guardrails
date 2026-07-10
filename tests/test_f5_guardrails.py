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

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest
from aioresponses import aioresponses

from nemoguardrails import RailsConfig
from nemoguardrails.library.f5.actions import f5_guardrails_scan
from tests.utils import TestChat


@pytest.fixture
def config():  # language=yaml
    return RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-5-nano

            rails:
              input:
                flows:
                  - f5 guardrails scan input
              output:
                flows:
                  - f5 guardrails scan output
        """,
    )


@pytest.fixture
def config_fail_open():  # language=yaml
    return RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-5-nano

            rails:
              config:
                f5:
                  fail_open: true
              input:
                flows:
                  - f5 guardrails scan input
              output:
                flows:
                  - f5 guardrails scan output
        """,
    )


def test_f5_guardrails_api_key_not_set(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "")
    chat = TestChat(config)
    chat.user("Hello! How are you?")
    chat.bot("I'm sorry, an internal error has occurred.")


def test_f5_guardrails_input_cleared(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
            repeat=True,
        )

        chat >> "Hello!"
        chat << "express greeting"
        chat << "Hello! How can I assist you today?"


def test_f5_guardrails_input_blocked(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "flagged"}},
            repeat=True,
        )

        chat >> "bad message"
        chat << "I'm sorry, I can't respond to that."


def test_f5_guardrails_output_blocked(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config,
        llm_completions=[
            " This is a bad response",
        ],
    )

    with aioresponses() as m:
        # Input scan cleared
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )
        # Output scan blocked
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "flagged"}},
        )

        chat >> "Hello"
        chat << "I'm sorry, I can't respond to that."


def test_f5_guardrails_fail_open(config_fail_open, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config_fail_open,
        llm_completions=[
            "  express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    with aioresponses() as m:
        # Simulate an API error (e.g., 500)
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=500,
            repeat=True,
        )

        chat >> "Hello!"
        chat << "express greeting"
        chat << "Hello! How can I assist you today?"


def test_f5_guardrails_fail_closed(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(config)

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=500,
            repeat=True,
        )

        chat >> "Hello!"
        chat << "I'm sorry, an internal error has occurred."


@pytest.mark.asyncio
async def test_f5_guardrails_timeout_fail_open(config_fail_open, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            exception=asyncio.TimeoutError(),
        )

        result = await f5_guardrails_scan(text="Hello!", config=config_fail_open)

    assert result == {"result": {"outcome": "cleared"}, "fail_open": True}


@pytest.mark.asyncio
async def test_f5_guardrails_fail_open_marker_on_http_error(config_fail_open, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=500,
            body="upstream failure",
        )

        result = await f5_guardrails_scan(text="Hello!", config=config_fail_open)

    assert result == {"result": {"outcome": "cleared"}, "fail_open": True}


@pytest.mark.asyncio
async def test_f5_guardrails_error_body_not_logged(config_fail_open, monkeypatch, caplog):
    """Vendor error bodies must not be echoed into logs.

    Some upstreams reflect scanned content in error responses. The action
    must log only structural fields (status, content-type, body length),
    never the body itself.
    """
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    sentinel = "SENSITIVE-USER-INPUT-DO-NOT-LOG-12345"
    caplog.set_level(logging.DEBUG, logger="nemoguardrails.library.f5.actions")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=500,
            body=f'{{"error": "rejected input", "input": "{sentinel}"}}',
            content_type="application/json",
        )

        await f5_guardrails_scan(text=sentinel, config=config_fail_open)

    for record in caplog.records:
        assert sentinel not in record.getMessage(), f"Vendor error body leaked into log record: {record.getMessage()!r}"


@pytest.fixture
def config_v2():  # language=yaml
    return RailsConfig.from_content(
        yaml_content="""
            colang_version: 2.x
            models: []
        """,
        colang_content="""
            import core
            import llm
            import guardrails
            import nemoguardrails.library.f5

            flow input rails $input_text
              f5 guardrails scan input $input_text

            flow output rails $output_text
              f5 guardrails scan output $output_text

            flow main
              activate llm continuation
              user said something
              bot say "Hello! How can I assist you today?"
        """,
    )


def test_f5_guardrails_colang_2_input_blocked(config_v2, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(config_v2)

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "flagged"}},
            repeat=True,
        )

        chat >> "bad message"
        chat << "I'm sorry, I can't respond to that."


def test_f5_guardrails_colang_2_input_cleared(config_v2, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(config_v2)

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
            repeat=True,
        )

        chat >> "Hello!"
        chat << "Hello! How can I assist you today?"


@pytest.mark.asyncio
async def test_f5_guardrails_timeout_fail_closed(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            exception=asyncio.TimeoutError(),
        )

        with pytest.raises(RuntimeError, match="timed out"):
            await f5_guardrails_scan(text="Hello!", config=config)


@pytest.mark.asyncio
async def test_f5_guardrails_custom_api_url(monkeypatch):
    """rails.config.f5.api_url overrides the default endpoint."""
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    custom_config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                f5:
                  api_url: https://custom.example.com
        """,
    )

    with aioresponses() as m:
        m.post(
            "https://custom.example.com/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )

        result = await f5_guardrails_scan(text="Hello!", config=custom_config)

    assert result == {"result": {"outcome": "cleared"}}


@pytest.mark.asyncio
async def test_f5_guardrails_api_url_env_fallback(config, monkeypatch):
    """F5_GUARDRAILS_API_URL is used when rails.config.f5.api_url is unset."""
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    monkeypatch.setenv("F5_GUARDRAILS_API_URL", "https://env.example.com")

    with aioresponses() as m:
        m.post(
            "https://env.example.com/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )

        result = await f5_guardrails_scan(text="Hello!", config=config)

    assert result == {"result": {"outcome": "cleared"}}


@pytest.mark.asyncio
async def test_f5_guardrails_config_api_url_wins_over_env(monkeypatch):
    """F5_GUARDRAILS_API_URL overrides rails.config.f5.api_url."""
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    monkeypatch.setenv("F5_GUARDRAILS_API_URL", "https://beta.example.com")

    custom_config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                f5:
                  api_url: https://www.example.com
        """,
    )

    with aioresponses() as m:
        m.post(
            "https://beta.example.com/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )

        result = await f5_guardrails_scan(text="Hello!", config=custom_config)

    assert result == {"result": {"outcome": "cleared"}}


@pytest.fixture
def config_exceptions():  # language=yaml
    return RailsConfig.from_content(
        yaml_content="""
            enable_rails_exceptions: true

            rails:
              input:
                flows:
                  - f5 guardrails scan input
              output:
                flows:
                  - f5 guardrails scan output
        """,
    )


@pytest.mark.asyncio
async def test_f5_guardrails_input_rails_exception(config_exceptions, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config_exceptions,
        llm_completions=[
            "  express greeting",
        ],
    )

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "flagged"}},
            repeat=True,
        )

        messages = [{"role": "user", "content": "bad message"}]
        result = await chat.app.generate_async(messages=messages)

    assert result["role"] == "exception"
    assert result["content"]["type"] == "F5GuardrailsRailException"
    assert "f5 guardrails scan input" in result["content"]["message"]


@pytest.mark.asyncio
async def test_f5_guardrails_output_rails_exception(config_exceptions, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config_exceptions,
        llm_completions=[
            "  express greeting",
            "This is a response.",
        ],
    )

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "flagged"}},
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = await chat.app.generate_async(messages=messages)

    assert result["role"] == "exception"
    assert result["content"]["type"] == "F5GuardrailsRailException"
    assert "f5 guardrails scan output" in result["content"]["message"]


@pytest.fixture
def config_no_backoff():  # language=yaml
    """Config with fast retries so the tests do not sleep for real."""
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                f5:
                  max_retries: 2
                  max_retry_after_seconds: 5.0
                  retry_backoff_seconds: 0.0
        """,
    )


@pytest.fixture
def config_no_backoff_fail_open():  # language=yaml
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                f5:
                  fail_open: true
                  max_retries: 2
                  max_retry_after_seconds: 5.0
                  retry_backoff_seconds: 0.0
        """,
    )


@pytest.mark.asyncio
async def test_f5_guardrails_429_then_success(config_no_backoff, monkeypatch):
    """A 429 with Retry-After: 0 is retried and the second call succeeds."""
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=429,
            headers={"Retry-After": "0"},
        )
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )

        with patch(
            "nemoguardrails.library.f5.actions.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ) as sleep_mock:
            result = await f5_guardrails_scan(text="Hello!", config=config_no_backoff)

    assert result == {"result": {"outcome": "cleared"}}
    sleep_mock.assert_awaited_once()
    assert sleep_mock.await_args.args[0] == 0.0


@pytest.mark.asyncio
async def test_f5_guardrails_429_retry_after_http_date(config_no_backoff, monkeypatch):
    """HTTP-date Retry-After values are parsed and honored."""
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=429,
            headers={"Retry-After": "Wed, 01 Jan 2020 00:00:00 GMT"},
        )
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )

        with patch(
            "nemoguardrails.library.f5.actions.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ) as sleep_mock:
            result = await f5_guardrails_scan(text="Hello!", config=config_no_backoff)

    assert result == {"result": {"outcome": "cleared"}}
    sleep_mock.assert_awaited_once()
    assert sleep_mock.await_args.args[0] == 0.0


@pytest.mark.asyncio
async def test_f5_guardrails_429_retry_after_capped(config_no_backoff, monkeypatch):
    """Retry-After values larger than max_retry_after_seconds are clamped."""
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=429,
            headers={"Retry-After": "9999"},
        )
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )

        with patch(
            "nemoguardrails.library.f5.actions.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ) as sleep_mock:
            result = await f5_guardrails_scan(text="Hello!", config=config_no_backoff)

    assert result == {"result": {"outcome": "cleared"}}
    sleep_mock.assert_awaited_once()
    assert sleep_mock.await_args.args[0] == 5.0  # max_retry_after_seconds


@pytest.mark.asyncio
async def test_f5_guardrails_429_exhausted_fail_open(config_no_backoff_fail_open, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=429,
            headers={"Retry-After": "0"},
            repeat=True,
        )

        with patch(
            "nemoguardrails.library.f5.actions.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ):
            result = await f5_guardrails_scan(text="Hello!", config=config_no_backoff_fail_open)

    assert result == {"result": {"outcome": "cleared"}, "fail_open": True}


@pytest.mark.asyncio
async def test_f5_guardrails_429_exhausted_fail_closed(config_no_backoff, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=429,
            headers={"Retry-After": "0"},
            repeat=True,
        )

        with patch(
            "nemoguardrails.library.f5.actions.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(RuntimeError, match="rate limited"):
                await f5_guardrails_scan(text="Hello!", config=config_no_backoff)


@pytest.mark.asyncio
async def test_f5_guardrails_429_no_retry_after_uses_backoff(monkeypatch):
    """When Retry-After is missing, retry_backoff_seconds * 2**attempt is used."""
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")

    cfg = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                f5:
                  max_retries: 2
                  max_retry_after_seconds: 60.0
                  retry_backoff_seconds: 0.25
        """,
    )

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=429,
        )
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=429,
        )
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )

        with patch(
            "nemoguardrails.library.f5.actions.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ) as sleep_mock:
            result = await f5_guardrails_scan(text="Hello!", config=cfg)

    assert result == {"result": {"outcome": "cleared"}}
    assert sleep_mock.await_count == 2
    # Attempt 0 -> 0.25 * 2**0, attempt 1 -> 0.25 * 2**1
    assert sleep_mock.await_args_list[0].args[0] == 0.25
    assert sleep_mock.await_args_list[1].args[0] == 0.5
