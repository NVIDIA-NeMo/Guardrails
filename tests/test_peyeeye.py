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

"""Tests for the Peyeeye PII redaction & rehydration guardrail."""

import re

import pytest
from aioresponses import aioresponses

from nemoguardrails import RailsConfig
from nemoguardrails.library.peyeeye.actions import (
    peyeeye_redact,
    peyeeye_rehydrate,
)
from nemoguardrails.library.peyeeye.request import (
    PEyeEyeGuardrailAPIError,
    PEyeEyeGuardrailMissingSecrets,
)
from tests.utils import TestChat

REDACT_URL = "https://api.peyeeye.ai/v1/redact"
REHYDRATE_URL = "https://api.peyeeye.ai/v1/rehydrate"
SESSION_DELETE_RE = re.compile(r"https://api\.peyeeye\.ai/v1/sessions/.+")


def _build_config(yaml_extra: str = "", colang_content: str | None = None) -> RailsConfig:
    yaml_content = (
        """
        models: []
        rails:
          config:
            peyeeye:
              api_base: https://api.peyeeye.ai
        """
        + yaml_extra
    )
    return RailsConfig.from_content(
        yaml_content=yaml_content,
        colang_content=colang_content
        or """
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )


# ---------------------------------------------------------------- unit: redact


@pytest.mark.asyncio
async def test_redact_happy_path_stateful(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")

    config = _build_config()

    with aioresponses() as m:
        m.post(
            REDACT_URL,
            payload={
                "text": ["hi [EMAIL_1]"],
                "session_id": "ses_abc123",
            },
        )
        result = await peyeeye_redact(
            source="input",
            text="hi me@example.com",
            config=config,
        )

    assert result == {
        "redacted_text": "hi [EMAIL_1]",
        "session_id": "ses_abc123",
        "redacted": True,
    }


@pytest.mark.asyncio
async def test_redact_stashes_session_id_on_context(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()
    ctx: dict = {}

    with aioresponses() as m:
        m.post(
            REDACT_URL,
            payload={"text": ["[EMAIL_1]"], "session_id": "ses_xyz"},
        )
        await peyeeye_redact(
            source="input",
            text="me@example.com",
            config=config,
            context=ctx,
        )

    assert ctx["peyeeye_session_input"] == "ses_xyz"


@pytest.mark.asyncio
async def test_redact_stateless_returns_skey(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config(
        yaml_extra="""
              input:
                session_mode: stateless
        """
    )

    with aioresponses() as m:
        m.post(
            REDACT_URL,
            payload={
                "text": ["hi [EMAIL_1]"],
                "rehydration_key": "skey_sealed_blob",
            },
        )
        result = await peyeeye_redact(
            source="input",
            text="hi me@example.com",
            config=config,
        )

    assert result["session_id"] == "skey_sealed_blob"
    assert result["redacted"] is True
    assert result["redacted_text"] == "hi [EMAIL_1]"


@pytest.mark.asyncio
async def test_redact_length_mismatch_raises(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        # We send 2 texts but server returns 1 — must NOT be silently merged.
        m.post(
            REDACT_URL,
            payload={"text": ["only one"], "session_id": "ses_x"},
        )
        with pytest.raises(PEyeEyeGuardrailAPIError, match="returned 1 texts for 2"):
            await peyeeye_redact(
                source="input",
                text=["a", "b"],
                config=config,
            )


@pytest.mark.asyncio
async def test_redact_unexpected_response_shape_raises(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(
            REDACT_URL,
            payload={"unexpected": "shape"},
        )
        with pytest.raises(PEyeEyeGuardrailAPIError, match="unexpected response shape"):
            await peyeeye_redact(
                source="input",
                text="something",
                config=config,
            )


@pytest.mark.asyncio
async def test_redact_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("PEYEEYE_API_KEY", raising=False)
    config = _build_config()

    with pytest.raises(PEyeEyeGuardrailMissingSecrets):
        await peyeeye_redact(
            source="input",
            text="hi",
            config=config,
        )


@pytest.mark.asyncio
async def test_redact_401_raises_missing_secrets(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "bad-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(REDACT_URL, status=401, payload={"detail": "invalid key"})
        with pytest.raises(PEyeEyeGuardrailMissingSecrets):
            await peyeeye_redact(
                source="input",
                text="hi",
                config=config,
            )


@pytest.mark.asyncio
async def test_redact_5xx_raises_api_error(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(REDACT_URL, status=500, payload={"detail": "boom"})
        with pytest.raises(PEyeEyeGuardrailAPIError, match="status 500"):
            await peyeeye_redact(
                source="input",
                text="hi",
                config=config,
            )


@pytest.mark.asyncio
async def test_redact_invalid_source_raises(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with pytest.raises(ValueError, match="source must be one of"):
        await peyeeye_redact(
            source="banana",
            text="hi",
            config=config,
        )


@pytest.mark.asyncio
async def test_redact_empty_text_skips_call(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        # No mock registered — if a call is made, aioresponses raises.
        result = await peyeeye_redact(
            source="input",
            text="",
            config=config,
        )
        # Sanity: also no calls registered.
        assert not m.requests

    assert result == {"redacted_text": "", "session_id": None, "redacted": False}


@pytest.mark.asyncio
async def test_redact_passes_entities_and_locale(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config(
        yaml_extra="""
              input:
                entities:
                  - EMAIL
                  - PHONE
                locale: en-US
        """
    )

    with aioresponses() as m:
        m.post(
            REDACT_URL,
            payload={"text": ["[EMAIL_1]"], "session_id": "ses_x"},
        )
        await peyeeye_redact(
            source="input",
            text="me@example.com",
            config=config,
        )

        # Inspect the captured request body via aioresponses' request log.
        req_key = next(iter(m.requests))
        req = m.requests[req_key][0]
        body = req.kwargs["json"]
        assert body["entities"] == ["EMAIL", "PHONE"]
        assert body["locale"] == "en-US"
        assert "session" not in body  # stateful default


# ------------------------------------------------------------- unit: rehydrate


@pytest.mark.asyncio
async def test_rehydrate_replaces_placeholders_and_cleans_up(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(
            REHYDRATE_URL,
            payload={"text": "hi me@example.com", "replaced": 1},
        )
        # DELETE for cleanup; aioresponses requires explicit registration.
        m.delete(SESSION_DELETE_RE, status=204)

        result = await peyeeye_rehydrate(
            text="hi [EMAIL_1]",
            session_id="ses_abc123",
            config=config,
        )

        # Cleanup runs in the background; drain so the task completes before
        # ``aioresponses`` tears down its mock.
        import asyncio as _asyncio

        pending = [t for t in _asyncio.all_tasks() if t is not _asyncio.current_task() and not t.done()]
        if pending:
            await _asyncio.gather(*pending, return_exceptions=True)

    assert result == {"text": "hi me@example.com", "replaced": 1}


@pytest.mark.asyncio
async def test_rehydrate_no_op_when_inputs_empty(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    # No HTTP mock — must short-circuit.
    result = await peyeeye_rehydrate(text="", session_id="", config=config)
    assert result == {"text": "", "replaced": 0}

    result = await peyeeye_rehydrate(text="hi", session_id="", config=config)
    assert result == {"text": "hi", "replaced": 0}


@pytest.mark.asyncio
async def test_rehydrate_swallows_api_error_and_returns_original(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(REHYDRATE_URL, status=500, payload={"detail": "boom"})

        result = await peyeeye_rehydrate(
            text="hi [EMAIL_1]",
            session_id="ses_abc123",
            config=config,
            cleanup=False,
        )

    assert result == {"text": "hi [EMAIL_1]", "replaced": 0}


@pytest.mark.asyncio
async def test_rehydrate_skips_delete_for_stateless(monkeypatch):
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(REHYDRATE_URL, payload={"text": "hi me@example.com", "replaced": 1})
        # No DELETE mock registered: if the action attempts one, aioresponses raises.
        result = await peyeeye_rehydrate(
            text="hi [EMAIL_1]",
            session_id="skey_sealed",
            config=config,
        )

    assert result["replaced"] == 1


@pytest.mark.asyncio
async def test_rehydrate_falls_back_when_payload_text_is_null(monkeypatch):
    """A buggy provider returning ``{"text": null}`` must not propagate ``None``
    into ``$bot_message``; we fall back to the original (still redacted) text."""
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(REHYDRATE_URL, payload={"text": None, "replaced": 0})
        m.delete(SESSION_DELETE_RE, status=204)

        result = await peyeeye_rehydrate(
            text="hi [EMAIL_1]",
            session_id="ses_abc123",
            config=config,
        )

    assert result == {"text": "hi [EMAIL_1]", "replaced": 0}


@pytest.mark.asyncio
async def test_redact_rejects_non_object_json_payload(monkeypatch):
    """A 200 response with a JSON list (not an object) must surface as an API
    error rather than crashing later with ``AttributeError`` on ``.get``."""
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(REDACT_URL, payload=["unexpected", "list"])
        with pytest.raises(PEyeEyeGuardrailAPIError, match="non-object JSON payload"):
            await peyeeye_redact(
                source="input",
                text="hi",
                config=config,
            )


@pytest.mark.asyncio
async def test_rehydrate_cleanup_runs_in_background(monkeypatch):
    """The DELETE /v1/sessions/{id} call must be fired as a background task
    rather than awaited inline — otherwise a slow cleanup endpoint adds latency
    to every rehydrated response."""
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    with aioresponses() as m:
        m.post(REHYDRATE_URL, payload={"text": "hi me@example.com", "replaced": 1})
        m.delete(SESSION_DELETE_RE, status=204)

        result = await peyeeye_rehydrate(
            text="hi [EMAIL_1]",
            session_id="ses_abc123",
            config=config,
        )

        # At return time the DELETE may not have fired yet — drain pending
        # tasks so the background DELETE has a chance to run before we assert.
        import asyncio as _asyncio

        pending = [t for t in _asyncio.all_tasks() if t is not _asyncio.current_task() and not t.done()]
        if pending:
            await _asyncio.gather(*pending, return_exceptions=True)

    assert result == {"text": "hi me@example.com", "replaced": 1}
    delete_calls = [c for k, calls in m.requests.items() if k[0] == "DELETE" for c in calls]
    assert delete_calls, "expected best-effort DELETE to fire (eventually)"


@pytest.mark.asyncio
async def test_rehydrate_error_body_is_truncated(monkeypatch):
    """The exception message for a 4xx/5xx must not include the full provider
    response body verbatim — large bodies are truncated."""
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")
    config = _build_config()

    huge = "X" * 5000

    with aioresponses() as m:
        m.post(REDACT_URL, status=500, body=huge, content_type="text/plain")
        with pytest.raises(PEyeEyeGuardrailAPIError) as exc_info:
            await peyeeye_redact(
                source="input",
                text="hi",
                config=config,
            )

    msg = str(exc_info.value)
    # The full body must not appear.
    assert huge not in msg
    # But a truncation marker should.
    assert "truncated" in msg


@pytest.mark.asyncio
async def test_rehydrate_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("PEYEEYE_API_KEY", raising=False)
    config = _build_config()

    with pytest.raises(PEyeEyeGuardrailMissingSecrets):
        await peyeeye_rehydrate(
            text="hi [EMAIL_1]",
            session_id="ses_abc123",
            config=config,
        )


# -------------------------------------------------------- end-to-end rail test


def test_rail_redacts_input_and_rehydrates_output(monkeypatch):
    """Happy path through the v1 colang rail: input is redacted before reaching
    the LLM and the LLM's response is rehydrated before returning to the user."""
    monkeypatch.setenv("PEYEEYE_API_KEY", "test-key")

    config = RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
            rails:
              config:
                peyeeye:
                  api_base: https://api.peyeeye.ai
              input:
                flows:
                  - peyeeye redact input
              output:
                flows:
                  - peyeeye rehydrate output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    with aioresponses() as m:
        m.post(
            REDACT_URL,
            payload={
                "text": ["hi, my email is [EMAIL_1]"],
                "session_id": "ses_e2e",
            },
        )
        # Bot greeting is hard-coded so rehydrate gets the literal greeting,
        # but the rail still runs; mock /rehydrate to echo the bot text.
        m.post(
            REHYDRATE_URL,
            payload={
                "text": "Hello! How can I assist you today?",
                "replaced": 0,
            },
        )
        m.delete(SESSION_DELETE_RE, status=204)

        chat >> "hi, my email is me@example.com"
        chat << "Hello! How can I assist you today?"

        # The redact mock must have been called with the original user text.
        redact_calls = [
            c for k, calls in m.requests.items() if k[0] == "POST" for c in calls if str(k[1]).endswith("/v1/redact")
        ]
        assert redact_calls, "expected at least one /v1/redact call"
        body = redact_calls[0].kwargs["json"]
        assert body["text"] == ["hi, my email is me@example.com"]
