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

"""Model engine for IORails.

Wraps a single Model config and makes raw HTTP calls to its
OpenAI-compatible /v1/chat/completions endpoint via aiohttp.
Retries are handled by aiohttp-retry (ExponentialRetry).
"""

import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, NamedTuple, Optional, cast

import aiohttp
from aiohttp_retry import RetryClient

from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
    safe_read_body,
)
from nemoguardrails.guardrails.base_engine import BaseEngine
from nemoguardrails.guardrails.guardrails_types import LLMMessages, get_request_id, truncate
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.types import LLMResponse, LLMResponseChunk, UsageInfo

log = logging.getLogger(__name__)

# Default base URLs by engine type
_ENGINE_BASE_URLS = {
    "nim": "https://integrate.api.nvidia.com",
    "openai": "https://api.openai.com",
}

_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


class _RequestParams(NamedTuple):
    """Pre-built parameters for an HTTP request to the completions endpoint."""

    client: RetryClient
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


def _parse_usage(usage_dict: dict) -> UsageInfo:
    """Build UsageInfo from an OpenAI-format usage dict.

    Picks up reasoning_tokens from completion_tokens_details (OpenAI reasoning
    models) and cached_tokens from prompt_tokens_details when present.
    """
    completion_details = usage_dict.get("completion_tokens_details") or {}
    prompt_details = usage_dict.get("prompt_tokens_details") or {}
    return UsageInfo(
        input_tokens=usage_dict.get("prompt_tokens", 0),
        output_tokens=usage_dict.get("completion_tokens", 0),
        total_tokens=usage_dict.get("total_tokens", 0),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        cached_tokens=prompt_details.get("cached_tokens"),
    )


def _parse_chat_completion(response: dict) -> LLMResponse:
    """Convert a /v1/chat/completions response dict into an LLMResponse.

    Reasoning is read from ``message.reasoning_content`` when the provider
    exposes it (NIM, DeepSeek-style). Tool calls are out of scope for this
    PR series and are not currently surfaced.
    """
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected /v1/chat/completions response shape: {exc}") from exc

    if not isinstance(content, str):
        if content is None and message.get("tool_calls"):
            raise ValueError(
                "Tool-call-only responses are not yet supported by IORails (message contains tool_calls but no content)"
            )
        raise ValueError(f"Expected string content, got {type(content).__name__}")

    reasoning = message.get("reasoning_content") or None

    usage = _parse_usage(response["usage"]) if response.get("usage") else None

    return LLMResponse(
        content=content,
        reasoning=reasoning,
        model=response.get("model"),
        finish_reason=choice.get("finish_reason"),
        request_id=response.get("id"),
        usage=usage,
    )


def _parse_chat_completion_chunk(chunk: dict) -> Optional[LLMResponseChunk]:
    """Build an LLMResponseChunk from an SSE chunk dict.

    Returns None for chunks without one of: content delta, reasoning delta,
    or a usage payload.
    Role-only first events and finish-only events with empty deltas
    map to None.

    Last chunk from OpenAI-compatible providers has a ``usage`` field when
    ``stream_options.include_usage=true``. This is passed through to capture
    the token usage metadata.
    """
    choices = chunk.get("choices") or []
    usage_dict = chunk.get("usage")

    delta_content: Optional[str] = None
    delta_reasoning: Optional[str] = None
    finish_reason = None
    if choices:
        choice = choices[0]
        delta = choice.get("delta") or {}
        delta_content = delta.get("content")
        delta_reasoning = delta.get("reasoning_content") or None
        finish_reason = choice.get("finish_reason")

    if not delta_content and not delta_reasoning and not usage_dict:
        return None

    return LLMResponseChunk(
        delta_content=delta_content,
        delta_reasoning=delta_reasoning,
        model=chunk.get("model"),
        finish_reason=finish_reason,
        request_id=chunk.get("id"),
        usage=_parse_usage(usage_dict) if usage_dict else None,
    )


class ModelEngineError(Exception):
    """Raised when a model engine call fails."""

    def __init__(self, message: str, model_name: str, status: int | None = None) -> None:
        self.model_name = model_name
        self.status = status
        super().__init__(message)


class ModelEngine(BaseEngine):
    """Wraps a single Model config and makes HTTP calls to its endpoint.

    Each ModelEngine owns its own RetryClient with per-model timeout,
    retry, and connection pool settings.
    """

    def __init__(self, model_config: Model) -> None:
        """Resolve base URL, API key, and retry settings from the model config."""
        self.model_config = model_config
        self.model_name: str = model_config.model or ""
        self.base_url: str = self._resolve_base_url()
        self.api_key: Optional[str] = self._resolve_api_key(model_config.engine)

        params = model_config.parameters or {}
        super().__init__(
            timeout_total=float(params.get("timeout") or DEFAULT_TIMEOUT_TOTAL),
            timeout_connect=float(params.get("timeout_connect") or DEFAULT_TIMEOUT_CONNECT),
            max_attempts=int(params.get("max_attempts") or DEFAULT_MAX_ATTEMPTS),
        )

    def _resolve_base_url(self) -> str:
        """Resolve the base URL from model parameters or engine type.

        Strips an optional trailing "/v1" so users can follow the OpenAI / LLMRails
        convention of including "/v1" in base_url without producing a doubled
        "/v1/v1/chat/completions" path when _CHAT_COMPLETIONS_ENDPOINT is appended.
        """
        params = self.model_config.parameters or {}
        engine = self.model_config.engine

        if params.get("base_url"):
            base_url = params["base_url"]
        elif engine in _ENGINE_BASE_URLS:
            base_url = _ENGINE_BASE_URLS[engine]
        else:
            raise ValueError(
                f"No base_url in parameters and cannot infer from engine '{engine}' for model '{self.model_name}'"
            )

        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url

    def _get_environment_variable(self, variable_name: str) -> str | None:
        """Return the value stored in environment variable `variable_name`."""
        env_value = os.environ.get(variable_name)
        return env_value

    def _resolve_api_key(self, engine: str | None) -> Optional[str]:
        """Resolve the API key from model config or environment."""
        if self.model_config.api_key_env_var:
            env_value = self._get_environment_variable(self.model_config.api_key_env_var)

            # Only raise an exception if the user provided an API Key env var and it isn't set
            if not env_value:
                raise RuntimeError(f"Environment variable '{self.model_config.api_key_env_var}' not set")
            return env_value

        if engine == "nim":
            env_value = self._get_environment_variable("NVIDIA_API_KEY")
            return env_value

        if engine == "openai":
            env_value = self._get_environment_variable("OPENAI_API_KEY")
            return env_value

        # If no key is available, assume it's a local model that doesn't need one
        return None

    def _ensure_running(self) -> None:
        """Raise if the engine has not been started."""
        if not self._running:
            raise ModelEngineError(
                f"ModelEngine for '{self.model_name}' has not been started. Call start() first.",
                model_name=self.model_name,
            )

    def _prepare_request(self, messages: LLMMessages, **kwargs: Any) -> _RequestParams:
        """Build the client, URL, headers, and body common to every request."""
        client = cast(RetryClient, self._client)
        url = self.base_url + _CHAT_COMPLETIONS_ENDPOINT

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict[str, Any] = {"model": self.model_name, "messages": messages, **kwargs}
        return _RequestParams(client=client, url=url, headers=headers, body=body)

    async def _raise_for_status(self, response: aiohttp.ClientResponse, req_id: str, t0: float) -> None:
        """Raise ``ModelEngineError`` if the HTTP status indicates an error."""
        if response.status >= 400:
            elapsed_ms = (time.monotonic() - t0) * 1000
            error_body = await safe_read_body(response)
            log.warning(
                "[%s] HTTP %s from model '%s' time=%.1fms",
                req_id,
                response.status,
                self.model_name,
                elapsed_ms,
            )
            raise ModelEngineError(
                f"HTTP {response.status} from model '{self.model_name}': {error_body}",
                model_name=self.model_name,
                status=response.status,
            )

    def _wrap_exception(self, exc: Exception, req_id: str, t0: float, label: str = "Request") -> ModelEngineError:
        """Wrap an unexpected exception in a ``ModelEngineError``."""
        elapsed_ms = (time.monotonic() - t0) * 1000
        log.warning("[%s] %s to model '%s' failed time=%.1fms", req_id, label, self.model_name, elapsed_ms)
        return ModelEngineError(
            f"{label} to model '{self.model_name}' failed: {exc}",
            model_name=self.model_name,
        )

    async def call(
        self,
        messages: LLMMessages,
        **kwargs: Any,
    ) -> dict:
        """Make a POST request to the /v1/chat/completions endpoint.

        Retries on transient failures (429, 5xx, connection errors) are
        handled automatically by the RetryClient with exponential backoff.

        Args:
            messages: List of message dicts in OpenAI format.
            **kwargs: Additional parameters for the request body (temperature, max_tokens, etc.)

        Returns:
            The parsed JSON response dict from the API.

        Raises:
            ModelEngineError: If the request fails after all retries.
        """
        self._ensure_running()
        req = self._prepare_request(messages, **kwargs)

        req_id = get_request_id()
        log.info("[%s] HTTP POST %s model='%s'", req_id, req.url, self.model_name)
        log.debug("[%s] HTTP request body: %s", req_id, truncate(req.body))

        t0 = time.monotonic()
        try:
            async with req.client.post(req.url, json=req.body, headers=req.headers) as response:
                await self._raise_for_status(response, req_id, t0)

                elapsed_ms = (time.monotonic() - t0) * 1000
                result = await response.json()
                log.debug(
                    "[%s] HTTP response status=%s time=%.1fms body: %s",
                    req_id,
                    response.status,
                    elapsed_ms,
                    truncate(result),
                )
                return result

        except ModelEngineError:
            raise
        except Exception as exc:
            raise self._wrap_exception(exc, req_id, t0) from exc

    async def stream_call(
        self,
        messages: LLMMessages,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponseChunk, None]:
        """Make a streaming POST request to the /v1/chat/completions endpoint.

        Sends ``stream=True`` and yields one ``LLMResponseChunk`` per SSE
        event that carries a content delta, reasoning delta, OR a
        ``usage`` payload. Role-only, finish-only, and empty-choices
        events without usage are skipped. Retries are handled by the
        RetryClient (same as ``call()``).

        Note: when the upstream payload includes
        ``stream_options.include_usage=true`` (default for the
        OpenAI-compatible client), the provider sends a final
        usage-only chunk with empty ``choices`` after the last content
        chunk. That terminal chunk is yielded as
        ``LLMResponseChunk(usage=...)`` with both ``delta_content``
        and ``delta_reasoning`` unset — callers that only care about
        content should gate on ``chunk.delta_content`` rather than
        assuming every yielded chunk carries one.

        Args:
            messages: List of message dicts in OpenAI format.
            **kwargs: Additional parameters for the request body (temperature, max_tokens, etc.)

        Yields:
            ``LLMResponseChunk`` objects with ``delta_content``,
            ``delta_reasoning``, and/or ``usage`` populated.

        Raises:
            ModelEngineError: If the request fails after all retries.
        """
        self._ensure_running()
        req = self._prepare_request(messages, stream=True, **kwargs)

        # For streaming, disable the total timeout (response body streams
        # for the full generation duration) and use sock_read to detect stalls
        # between individual SSE chunks.
        stream_timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self._timeout.connect,
            sock_read=self._timeout.total,
        )

        req_id = get_request_id()
        log.info("[%s] HTTP POST (stream) %s model='%s'", req_id, req.url, self.model_name)
        log.debug("[%s] HTTP request body: %s", req_id, truncate(req.body))

        t0 = time.monotonic()
        try:
            async with req.client.post(req.url, json=req.body, headers=req.headers, timeout=stream_timeout) as response:
                await self._raise_for_status(response, req_id, t0)

                # Use readline() instead of iterating response.content directly.
                # response.content uses readany() which returns arbitrary byte
                # chunks — multiple SSE events in one TCP segment would be merged
                # into one unparseable blob.  readline() splits on \n correctly.
                while True:
                    raw_line = await response.content.readline()
                    if not raw_line:
                        break

                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue

                    payload = line[len("data: ") :]
                    if payload == "[DONE]":
                        break

                    try:
                        raw_chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        log.warning("[%s] Unparseable SSE chunk: %s", req_id, payload[:200])
                        continue

                    parsed_chunk = _parse_chat_completion_chunk(raw_chunk)
                    if parsed_chunk is not None:
                        yield parsed_chunk

                elapsed_ms = (time.monotonic() - t0) * 1000
                log.debug("[%s] Stream completed time=%.1fms", req_id, elapsed_ms)

        except ModelEngineError:
            raise
        except Exception as exc:
            raise self._wrap_exception(exc, req_id, t0, label="Stream request") from exc

    async def chat_completion(self, messages: LLMMessages, **kwargs: Any) -> LLMResponse:
        """Generate a chat completion and return a structured ``LLMResponse``.

        Calls the /v1/chat/completions endpoint and parses the OpenAI-format
        response into an ``LLMResponse`` carrying content, reasoning (when the
        provider exposes ``reasoning_content``), usage, finish reason, and
        request id.

        Raises:
            ModelEngineError: If the request fails or the response format is unexpected.
        """
        response = await self.call(messages, **kwargs)
        try:
            return _parse_chat_completion(response)
        except ValueError as exc:
            raise ModelEngineError(
                f"Unexpected response format from model '{self.model_name}': {exc}",
                model_name=self.model_name,
            ) from exc

    async def stream_chat_completion(
        self, messages: LLMMessages, **kwargs: Any
    ) -> AsyncGenerator[LLMResponseChunk, None]:
        """Stream a chat completion and yield ``LLMResponseChunk`` objects.

        Thin pass-through over ``stream_call`` — see that method's
        docstring for the contract, including the terminal usage-only
        chunk emitted when ``stream_options.include_usage`` is on.

        Raises:
            ModelEngineError: If the request fails after all retries.
        """
        async for chunk in self.stream_call(messages, **kwargs):
            yield chunk
