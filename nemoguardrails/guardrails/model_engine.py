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

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import nullcontext
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

import aiohttp
from aiohttp_retry import RetryClient

from nemoguardrails.exceptions import (
    LLMClientError,
    LLMConnectionError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
    merge_headers_case_insensitive,
    safe_read_body,
)
from nemoguardrails.guardrails.base_engine import BaseEngine
from nemoguardrails.guardrails.guardrails_types import LLMMessages, get_request_id, truncate
from nemoguardrails.guardrails.telemetry import (
    llm_call_span,
    set_llm_call_content,
    set_llm_request_attributes,
    set_llm_response_attributes,
)
from nemoguardrails.guardrails.tool_schema import Tool, ToolExchange, ToolResult, Toolset
from nemoguardrails.llm.clients._errors import ErrorContext, raise_for_sse_error, raise_for_status
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.tracing.constants import (
    llm_operation_duration,
    record_time_per_output_chunk,
    record_time_to_first_chunk,
    record_token_usage,
)
from nemoguardrails.types import ChatMessage, LLMResponse, LLMResponseChunk, ToolCall, ToolCallFunction, UsageInfo

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

log = logging.getLogger(__name__)

# The OTEL GenAI operation name for every call this engine makes; it only speaks
# /v1/chat/completions.
_OPERATION_NAME = "chat"

# Default base URLs by engine type
_ENGINE_BASE_URLS = {
    "nim": "https://integrate.api.nvidia.com",
    "openai": "https://api.openai.com",
}

_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"

# Standard top-level keys in a chat-completion chunk; everything else (e.g. NIM's `nvext`)
# is surfaced as provider_metadata, mirroring LLMRails' `_STANDARD_RESPONSE_KEYS`.
_STANDARD_CHUNK_KEYS = frozenset({"model", "choices", "usage", "id", "object", "created"})

# Parameter keys the engine reserves and handles itself, so they are NOT
# forwarded into the /v1/chat/completions request body.  Everything else in
# a model's ``parameters`` config block becomes a per-request default via
# ``ModelEngine.body_param_defaults``.
_RESERVED_LLM_PARAMETERS = frozenset(
    {
        # transport / retry — consumed by ModelEngine.__init__ and
        # _resolve_base_url, never part of the request body
        "base_url",
        "timeout",
        "timeout_connect",
        "max_attempts",
        # secret — carried in the Authorization header, never echoed into the body
        "api_key",
        # model identity / positional — set explicitly by _prepare_request;
        # would collide with the "model" body field set there.  After Model
        # validation these never appear in parameters, but exclude them
        # defensively against direct construction.
        "model",
        "model_name",
        "messages",
        # streaming control — owned by the engine.  "stream" in particular
        # collides with the explicit stream=True kwarg that stream_call()
        # passes into _prepare_request(), which would raise TypeError on a
        # duplicate keyword argument.
        "stream",
        # Can't set Model-level `stream_options` because the same model can
        # be used in streaming or non-streaming mode. `stream_call` sets the
        # streaming default (`include_usage`), overridable via `llm_params`.
        "stream_options",
        # client-only options — these configure transport, not the
        # chat-completion request body, so they are never forwarded as body
        # fields.
        "default_headers",
        "default_query",
    }
)


class _RequestParams(NamedTuple):
    """Pre-built parameters for an HTTP request to the completions endpoint."""

    client: RetryClient
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


# TODO: duplicates OpenAIChatModel._to_messages in
# nemoguardrails/llm/models/openai_chat.py . Needs consolidating.
def _to_wire_messages(prompt: str | list) -> LLMMessages:
    """Normalize an ``LLMModel`` prompt into OpenAI-format message dicts.

    This is the ``LLMModel`` protocol boundary. Library rail actions render a
    task prompt with ``LLMTaskManager.render_task_prompt``, which returns either
    a string or a list of ``{"type": ..., "content": ...}`` dicts depending on
    whether the task prompt was configured with ``content:`` or ``messages:``;
    ``llm_call`` then hands over a string or a list of ``ChatMessage``. Both
    shapes are converted here. A list of dicts that is already in wire form
    passes through untouched.

    ``ChatMessage`` conversion drops ``provider_metadata`` — it is an internal
    field the completions endpoint would reject — and re-encodes tool-call
    arguments as a JSON string, which is the wire shape the API expects. It also
    maps the task manager's ``type`` key onto ``role``, which the API requires.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not prompt or not isinstance(prompt[0], ChatMessage):
        return prompt

    messages: LLMMessages = []
    for message in prompt:
        wire_message = message.to_dict()
        wire_message.pop("provider_metadata", None)
        for tool_call in wire_message.get("tool_calls", []):
            function = tool_call.get("function", {})
            if isinstance(function.get("arguments"), dict):
                function["arguments"] = json.dumps(function["arguments"])
        messages.append(wire_message)
    return messages


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
    exposes it (NIM, DeepSeek-style). Tool calls are parsed from
    ``message.tool_calls`` (OpenAI shape) into ``LLMResponse.tool_calls`` via
    ``ChatMessage.from_dict``, which normalizes JSON-string arguments into a
    dict. ``content`` is ``None`` on a tool-call-only response and is
    normalized to an empty string; a ``None`` content with no tool calls is
    treated as a malformed response.
    """
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected /v1/chat/completions response shape: {exc}") from exc

    raw_tool_calls = message.get("tool_calls")

    if content is None:
        # A tool-call-only response legitimately carries content=None; a null
        # content with no tool calls is malformed.
        if not raw_tool_calls:
            raise ValueError("Expected string content, got NoneType")
        content = ""
    elif not isinstance(content, str):
        raise ValueError(f"Expected string content, got {type(content).__name__}")

    tool_calls = ChatMessage.from_dict(message).tool_calls if raw_tool_calls else None

    reasoning = message.get("reasoning_content") or None

    usage = _parse_usage(response["usage"]) if response.get("usage") else None

    return LLMResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        model=response.get("model"),
        finish_reason=choice.get("finish_reason"),
        request_id=response.get("id"),
        usage=usage,
    )


def _parse_chat_completion_chunk(chunk: dict) -> Optional[LLMResponseChunk]:
    """Build an LLMResponseChunk from an SSE chunk dict.

    Returns None for chunks without one of: content delta, reasoning delta,
    a usage payload, or a finish_reason.
    Role-only first events map to None.

    Finish-only frames are preserved: a delta with no content/reasoning
    (OpenAI sends ``delta: {}``, NIM sends ``delta: {"content": ""}``) and no
    usage, carrying only a ``finish_reason``. Dropping them would strip
    ``gen_ai.response.finish_reasons`` from the LLM span. (Some providers
    instead attach ``finish_reason`` to the final content chunk — that case is
    already captured, since content keeps the chunk alive.) When
    ``stream_options.include_usage=true`` the usage payload arrives in a
    *separate* later frame with empty ``choices`` — so finish_reason and usage
    do not share a frame.

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

    if not delta_content and not delta_reasoning and not usage_dict and not finish_reason:
        return None

    provider_metadata = {
        key: value for key, value in chunk.items() if key not in _STANDARD_CHUNK_KEYS and value is not None
    }

    return LLMResponseChunk(
        delta_content=delta_content,
        delta_reasoning=delta_reasoning,
        model=chunk.get("model"),
        finish_reason=finish_reason,
        request_id=chunk.get("id"),
        usage=_parse_usage(usage_dict) if usage_dict else None,
        provider_metadata=provider_metadata or None,
    )


def _accumulate_tool_call_delta(tool_calls: dict[int, dict], raw_chunk: dict) -> None:
    """Update the tool-call accumulator with any tool_call deltas from a raw SSE chunk.

    OpenAI streams argument JSON as fragments across many chunks; NIM delivers
    complete arguments in one delta. Both are handled uniformly: ``tool_calls``
    is keyed by the OpenAI ``index`` field and mutated in place on every call.
    Finalize with ``_finalize_tool_calls`` once ``finish_reason=="tool_calls"``.
    """
    choices = raw_chunk.get("choices") or []
    if not choices:
        return
    delta = choices[0].get("delta") or {}
    for tool_call_delta in delta.get("tool_calls") or []:
        # ``index`` ties fragmented argument deltas back to their tool call (only
        # the first delta per call carries id/name; later ones carry just args +
        # the same index). It defaults to 0 for single-call streams that omit it;
        # OpenAI/NIM always send it for parallel calls. The collision check below
        # flags the rare case where a provider omits it for parallel calls.
        index = tool_call_delta.get("index", 0)
        function = tool_call_delta.get("function") or {}
        if index not in tool_calls:
            tool_calls[index] = {
                "id": tool_call_delta.get("id") or "",
                "name": function.get("name") or "",
                "arguments_buffer": "",
            }
        else:
            incoming_id = tool_call_delta.get("id")
            if incoming_id:
                existing_id = tool_calls[index]["id"]
                if existing_id and incoming_id != existing_id:
                    # A distinct tool call (different id) landed on an existing
                    # accumulator slot — happens when a provider omits ``index`` for
                    # parallel calls, collapsing them into one slot (id/name
                    # overwritten, arguments concatenated into invalid JSON). Warn
                    # rather than silently corrupt; strict per-provider validation
                    # is the canonicalization layer's job.
                    log.warning(
                        "[%s] tool-call id %r collided with accumulator slot %d (existing id %r); "
                        "provider likely omitted 'index' for parallel tool calls — result may be corrupted",
                        get_request_id(),
                        incoming_id,
                        index,
                        existing_id,
                    )
                tool_calls[index]["id"] = incoming_id
            name = function.get("name")
            if name:
                tool_calls[index]["name"] = name
        arguments_fragment = function.get("arguments") or ""
        if arguments_fragment:
            tool_calls[index]["arguments_buffer"] += arguments_fragment


# TODO: duplicates _finalize_tool_calls in
# nemoguardrails/llm/models/openai_chat.py . Needs consolidating.
def _finalize_tool_calls(tool_calls: dict[int, dict]) -> list[ToolCall]:
    """Assemble accumulated tool-call fragments into ToolCall objects.

    Called once when the stream emits finish_reason='tool_calls'. An empty buffer (no
    argument fragments streamed) is a no-argument call and becomes ``{}``; a non-empty
    buffer that is not a valid JSON object (e.g. arguments truncated mid-stream) raises
    ``ValueError`` so the malformed call fails closed rather than silently degrading to
    empty arguments that could pass the tool-call rail. This mirrors the non-streaming
    parser (``ChatMessage.from_dict``), which raises on the same bytes; ``stream_call``
    wraps the error into ``ModelEngineError`` exactly as the non-streaming path does.
    """
    result = []
    for index in sorted(tool_calls.keys()):
        entry = tool_calls[index]
        raw_arguments = entry.get("arguments_buffer", "")
        if raw_arguments:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Tool call arguments are not valid JSON: {raw_arguments!r}") from exc
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Tool call arguments must be a JSON object, got {type(arguments).__name__}: {raw_arguments!r}"
                )
        else:
            arguments = {}
        result.append(
            ToolCall(
                id=entry.get("id", ""),
                type="function",
                function=ToolCallFunction(
                    name=entry.get("name", ""),
                    arguments=arguments,
                ),
            )
        )
    return result


def _parse_tools_openai(tools: list) -> list[Tool]:
    """Parse OpenAI Chat Completions tool definitions into ``Tool`` objects.

    Each entry has the nested shape ``{"type": "function", "function": {"name",
    "description", "parameters", "strict"}}``; ``function.parameters`` (the JSON
    Schema) maps to ``Tool.arguments_schema``. Entries that are not a dict, lack a
    ``function`` block, or whose function has no non-empty ``name`` are skipped.
    """
    parsed: list[Tool] = []
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parsed.append(
            Tool(
                name=name,
                type=entry.get("type", "function"),
                description=function.get("description"),
                arguments_schema=function.get("parameters"),
                strict=function.get("strict"),
            )
        )
    return parsed


def _parse_tools_nim(tools: list) -> list[Tool]:
    """Parse NIM tool definitions. NIM uses the OpenAI Chat Completions tool shape."""
    return _parse_tools_openai(tools)


_TOOL_PARSERS = {
    "openai": _parse_tools_openai,
    "nim": _parse_tools_nim,
}


def _tool_result_from_message(message: dict) -> ToolResult:
    """Normalize one OpenAI Chat Completions ``role:"tool"`` message into a ``ToolResult``.

    This shape has no error flag, so ``is_error`` is always ``False``.
    """
    return ToolResult(
        call_id=message.get("tool_call_id"),
        name=message.get("name"),
        content=message.get("content"),
    )


def _extract_tool_results_openai(messages: LLMMessages) -> list[ToolResult]:
    """Extract OpenAI Chat Completions tool results into ``ToolResult`` objects.

    Chat Completions carries each tool result as a top-level ``{"role": "tool",
    "tool_call_id", "content"}`` message (optionally ``name``).
    """
    return [
        _tool_result_from_message(message)
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]


def _extract_tool_results_nim(messages: LLMMessages) -> list[ToolResult]:
    """Extract NIM tool results. NIM uses the OpenAI Chat Completions shape."""
    return _extract_tool_results_openai(messages)


_RESULT_EXTRACTORS = {
    "openai": _extract_tool_results_openai,
    "nim": _extract_tool_results_nim,
}


def _tool_calls_from_message(message: dict) -> list[ToolCall]:
    """Extract the tool calls from one assistant message into ``ToolCall`` objects.
    Malformed tool calls fall back to just id, type, function"""
    try:
        return ChatMessage.from_dict(message).tool_calls or []
    except ValueError:
        return [
            ToolCall(
                id=raw.get("id", ""),
                type=raw.get("type", "function"),
                function=ToolCallFunction(name=(raw.get("function") or {}).get("name", ""), arguments={}),
            )
            for raw in message.get("tool_calls") or []
            if isinstance(raw, dict)
        ]


def _extract_tool_exchanges_openai(messages: LLMMessages) -> list[ToolExchange]:
    """Group an OpenAI Chat Completions conversation into per-turn ``ToolExchange``es."""
    exchanges: list[ToolExchange] = []
    # The exchange currently accepting tool results: the most recent assistant tool-call
    # turn, or an orphan opened by a leading/stray tool result. Any other message resets
    # it to None so the next tool result starts a fresh exchange instead of attaching to
    # a closed turn.
    open_exchange: ToolExchange | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            if open_exchange is None:
                open_exchange = ToolExchange(calls=[], results=[])
                exchanges.append(open_exchange)
            open_exchange.results.append(_tool_result_from_message(message))
        elif role == "assistant" and message.get("tool_calls"):
            open_exchange = ToolExchange(calls=_tool_calls_from_message(message), results=[])
            exchanges.append(open_exchange)
        else:
            open_exchange = None
    return exchanges


def _extract_tool_exchanges_nim(messages: LLMMessages) -> list[ToolExchange]:
    """Extract NIM tool exchanges. NIM uses the OpenAI Chat Completions shape."""
    return _extract_tool_exchanges_openai(messages)


# How much of a failed response to read for error classification. Large enough that any
# realistic provider error body parses as JSON and yields its message/code/param, bounded
# because an unparseable body (an HTML proxy error page) becomes the client-facing message
# verbatim, and must not be forwarded whole.
_ERROR_BODY_MAX_CHARS = 8192


_TOOL_EXCHANGE_EXTRACTORS = {
    "openai": _extract_tool_exchanges_openai,
    "nim": _extract_tool_exchanges_nim,
}


class ModelEngineError(Exception):
    """Raised when a model engine call fails."""

    def __init__(
        self,
        message: str,
        model_name: str,
        status: int | None = None,
        *,
        inner_exception: BaseException | None = None,
    ) -> None:
        self.model_name = model_name
        self.status = status
        # The typed LLMClientError this failure was classified into, when it came from a
        # provider response. ``nemoguardrails.llm.clients._errors.as_client_error`` reads
        # this attribute, which is how the server renders the provider's own message, code,
        # param, and Retry-After instead of falling back to ``str(exception)``.
        self.inner_exception = inner_exception
        super().__init__(message)

    @property
    def status_code(self) -> int | None:
        """The upstream HTTP status, under the name status extractors look for.

        A rail reaching the model through ``llm_call`` has this error wrapped into
        ``LLMCallException``, and ``nemoguardrails.llm.call._extract_http_status``
        duck-types on ``status_code`` — the OpenAI SDK and httpx spelling.  Without
        this alias the upstream status is dropped and the server reports 500 for
        what was really a 429 or 503, which SDKs retry differently.  Callers that
        hold the concrete type can keep reading ``status``.
        """
        return self.status


class ModelEngine(BaseEngine):
    """Wraps a single Model config and makes HTTP calls to its endpoint.

    Each ModelEngine owns its own RetryClient with per-model timeout,
    retry, and connection pool settings.

    Implements the :class:`~nemoguardrails.types.LLMModel` protocol through
    ``generate_async`` / ``stream_async``, so library rail actions can reach an
    IORails-configured model through ``nemoguardrails.llm.call.llm_call`` the
    same way they reach any other backend.
    """

    def __init__(
        self,
        model_config: Model,
        *,
        tracer: Optional["Tracer"] = None,
        metrics_enabled: bool = False,
        content_capture_enabled: bool = False,
    ) -> None:
        """Resolve base URL, API key, and retry settings from the model config.

        The telemetry arguments carry the OTEL instrumentation for every call
        this engine makes.  They live here rather than one level up in
        ``EngineRegistry`` because rails reach the model through
        ``generate_async``, not through ``EngineRegistry.model_call``.
        """
        self.model_config = model_config
        self.model_name: str = model_config.model or ""
        self.base_url: str = self._resolve_base_url()
        self.api_key: Optional[str] = self._resolve_api_key(model_config.engine)
        self._tracer = tracer
        self._metrics_enabled = metrics_enabled
        self._content_capture_enabled = content_capture_enabled

        params = model_config.parameters or {}
        super().__init__(
            timeout_total=float(params.get("timeout") or DEFAULT_TIMEOUT_TOTAL),
            timeout_connect=float(params.get("timeout_connect") or DEFAULT_TIMEOUT_CONNECT),
            max_attempts=int(params.get("max_attempts") or DEFAULT_MAX_ATTEMPTS),
        )

        self.default_headers: Mapping[str, str] = MappingProxyType(
            {str(key): str(value) for key, value in (params.get("default_headers") or {}).items()}
        )

        # Default `llm_params` used on inference are the subset of Model.parameters after
        # filtering out keys in _RESERVED_LLM_PARAMETERS.  Exposed as a read-only
        # MappingProxyType view so callers can't mutate the shared per-engine defaults.
        self.body_param_defaults: Mapping[str, Any] = MappingProxyType(
            {key: value for key, value in params.items() if key not in _RESERVED_LLM_PARAMETERS}
        )

    @property
    def provider_name(self) -> str:
        """The provider this model is served by ('nim', 'openai', ...).

        Falls back to ``"unknown"`` so telemetry always carries a
        ``gen_ai.provider.name`` label rather than dropping the attribute.
        """
        return self.model_config.engine or "unknown"

    @property
    def provider_url(self) -> str:
        """The OpenAI-compatible API root for this model.

        ``base_url`` is stored without the ``/v1`` suffix (see
        ``_resolve_base_url``), so it is re-appended here to match what
        ``OpenAIChatModel.provider_url`` reports for the same endpoint.
        """
        return f"{self.base_url}/v1"

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
                inner_exception=self._generic_client_error(0, "The model engine is not running."),
            )

    def _prepare_request(self, messages: LLMMessages, **kwargs: Any) -> _RequestParams:
        """Build the client, URL, headers, and body common to every request."""
        client = cast(RetryClient, self._client)
        url = self.base_url + _CHAT_COMPLETIONS_ENDPOINT

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers = merge_headers_case_insensitive(headers, self.default_headers)

        body: dict[str, Any] = {"model": self.model_name, "messages": messages, **kwargs}
        return _RequestParams(client=client, url=url, headers=headers, body=body)

    @property
    def _error_context(self) -> ErrorContext:
        """Identify this engine's model to the shared error classifier."""
        return ErrorContext(
            model_name=self.model_name,
            provider_name=self.provider_name,
            base_url=self.base_url,
        )

    def _generic_client_error(self, status_code: int, message: str) -> LLMClientError:
        """Build a caller-safe client error for a failure the shared classifier cannot type.

        Every ``ModelEngineError`` needs one of these, because ``client_facing_message`` falls
        back to ``str(exception)`` without it, and this engine's messages name the model.
        The model context still travels for diagnostics: it reaches logs through
        ``LLMClientError.__str__``, never the client envelope, which renders
        ``error_message`` alone.
        """
        return LLMClientError(status_code, message, **self._error_context.as_kwargs())

    def _wrap_client_error(self, client_error: LLMClientError, summary: str) -> ModelEngineError:
        """Carry a classified provider error inside the ``ModelEngineError`` this engine raises.

        The message keeps the model name for the server log; it is the inner error, not this
        text, that the server renders for the caller, so the raw upstream body is left out
        rather than concatenated in.
        """
        if client_error.status_code > 0:
            status = client_error.status_code
        else:
            status = None
        return ModelEngineError(
            f"{summary} from model '{self.model_name}': {client_error.error_message}",
            model_name=self.model_name,
            status=status,
            inner_exception=client_error,
        )

    def _classify_error_response(self, status: int, body: str, headers: Any, req_id: str) -> LLMClientError | None:
        """Return the typed provider error for a failed response, or None if it cannot be classified."""
        try:
            raise_for_status(status, body, headers, self._error_context)
        except LLMClientError as client_error:
            return client_error
        except Exception:
            # A classifier failure must not cost the caller the upstream status. Letting this
            # escape would strip it in ``call``'s catch-all, and the server would report 500
            # for what was really a 429 or 503, which SDKs retry differently.
            log.warning(
                "[%s] could not classify HTTP %s from model '%s'",
                req_id,
                status,
                self.model_name,
                exc_info=True,
            )
        return None

    def _raise_for_sse_error(self, raw_chunk: dict, headers: Any) -> None:
        """Raise ``ModelEngineError`` for a provider error frame carried inside a stream.

        Mirrors ``BaseClient._check_sse_error``. Anything ``raise_for_sse_error`` raises other
        than an ``LLMClientError`` still terminates the stream through ``stream_call``'s
        catch-all, so the frame is never silently dropped.
        """
        try:
            raise_for_sse_error(raw_chunk, headers, self._error_context)
        except LLMClientError as client_error:
            raise self._wrap_client_error(client_error, "Stream error") from client_error

    async def _raise_for_status(self, response: aiohttp.ClientResponse, req_id: str, t0: float) -> None:
        """Raise ``ModelEngineError`` if the HTTP status indicates an error."""
        if response.status >= 400:
            elapsed_ms = (time.monotonic() - t0) * 1000
            error_body = await safe_read_body(response, max_chars=_ERROR_BODY_MAX_CHARS)
            log.warning(
                "[%s] HTTP %s from model '%s' time=%.1fms",
                req_id,
                response.status,
                self.model_name,
                elapsed_ms,
            )
            client_error = self._classify_error_response(response.status, error_body, response.headers, req_id)
            if client_error is None:
                # The body is left out deliberately: unclassified is exactly the case where it
                # could be anything, and it must not reach the caller. The status still does.
                client_error = self._generic_client_error(response.status, f"HTTP {response.status}")
            raise self._wrap_client_error(client_error, f"HTTP {response.status}") from client_error

    def _classify_transport_failure(self, exc: Exception) -> LLMClientError | None:
        """Return the typed client error for a timeout or connection failure, or None for anything else.

        Mirrors the httpx mapping in ``nemoguardrails.llm.clients.base``, so the same
        transport failure reads the same to a caller whichever engine served the request.
        No HTTP response arrived, hence status 0. Exception details stay on the cause
        chain instead of entering the client-facing message.
        """
        # Order matters: aiohttp's ServerTimeoutError subclasses ClientConnectionError, so
        # testing for a connection failure first would report every timeout as one.
        if isinstance(exc, (asyncio.TimeoutError, aiohttp.ServerTimeoutError)):
            return LLMTimeoutError(0, "Request timed out.", **self._error_context.as_kwargs())
        if isinstance(exc, aiohttp.ClientConnectionError):
            return LLMConnectionError(0, "Connection error.", **self._error_context.as_kwargs())
        return None

    def _wrap_exception(self, exc: Exception, req_id: str, t0: float, label: str = "Request") -> ModelEngineError:
        """Wrap an unexpected exception in a ``ModelEngineError``."""
        elapsed_ms = (time.monotonic() - t0) * 1000
        log.warning(
            "[%s] %s to model '%s' failed time=%.1fms",
            req_id,
            label,
            self.model_name,
            elapsed_ms,
            exc_info=exc,
        )
        client_error = self._classify_transport_failure(exc)
        if client_error is None:
            client_error = self._generic_client_error(0, f"{label} failed.")
        return ModelEngineError(
            client_error.error_message,
            model_name=self.model_name,
            inner_exception=client_error,
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

        Tool calls (when the request declared ``tools``) are accumulated
        from streamed ``delta.tool_calls`` fragments and surfaced as a
        single ``LLMResponseChunk`` whose ``delta_tool_calls`` carries the
        COMPLETE finalized list exactly once — on the first chunk with a
        ``finish_reason`` (``"tool_calls"`` for a free choice, ``"stop"``
        for a forced ``tool_choice``), or via a post-loop safety net if the
        provider omits a parseable finish frame. No other chunk carries
        ``delta_tool_calls``, so consumers may treat it as last-write-wins.

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
        # Request usage on the terminal stream chunk (LLMRails parity); a caller can override
        # or disable it via stream_options in llm_params.
        kwargs.setdefault("stream_options", {"include_usage": True})
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

                # Surface response headers on every chunk as provider_metadata['response_headers'],
                # lowercased to match LLMRails' httpx client (aiohttp preserves the server's casing).
                response_headers = (
                    {key.lower(): value for key, value in response.headers.items()}
                    if isinstance(response.headers, Mapping)
                    else None
                )

                # Use readline() instead of iterating response.content directly.
                # response.content uses readany() which returns arbitrary byte
                # chunks — multiple SSE events in one TCP segment would be merged
                # into one unparseable blob.  readline() splits on \n correctly.
                tool_calls: dict[int, dict] = {}
                tool_calls_emitted = False
                last_chunk_id = None
                while True:
                    raw_line = await response.content.readline()
                    if not raw_line:
                        break

                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    log.debug("[%s] SSE line: %s", req_id, truncate(line))

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

                    # A provider can report a failure in a frame rather than in the HTTP
                    # status, and the stream would otherwise end as though generation had
                    # completed. The value must be non-null, not merely present: a gateway
                    # that emits "error": null on every frame would otherwise end a healthy
                    # stream. Only a top-level key counts, so model output that merely looks
                    # like an error object stays ordinary content.
                    if isinstance(raw_chunk, dict) and raw_chunk.get("error") is not None:
                        self._raise_for_sse_error(raw_chunk, response.headers)

                    last_chunk_id = raw_chunk.get("id") or last_chunk_id
                    _accumulate_tool_call_delta(tool_calls, raw_chunk)

                    parsed_chunk = _parse_chat_completion_chunk(raw_chunk)

                    # Finalize accumulated tool calls onto the terminating chunk.
                    # A streamed tool call ends with a finish_reason — "tool_calls"
                    # when the model chose freely, but "stop" when tool_choice forced
                    # a specific function. Gate on *any* finish_reason (with tool calls
                    # accumulated), not the literal "tool_calls", or forced calls never
                    # surface. finish_reason keeps that chunk alive in
                    # _parse_chat_completion_chunk, so parsed_chunk is non-None here; if a
                    # provider ever omits a parseable finish frame, the post-loop safety
                    # net below surfaces the calls instead.
                    choices = raw_chunk.get("choices") or []
                    finish_reason = choices[0].get("finish_reason") if choices else None
                    if tool_calls and not tool_calls_emitted and finish_reason and parsed_chunk is not None:
                        if finish_reason not in ("tool_calls", "stop"):
                            # Abnormal terminator (e.g. "length", "content_filter")
                            # while a tool call is mid-assembly: surface what we have
                            # but warn, because the JSON arguments are likely truncated
                            # and will degrade to {} in _finalize_tool_calls.
                            log.warning(
                                "[%s] tool-call stream ended with finish_reason=%r; "
                                "assembled tool-call arguments may be truncated or incomplete",
                                req_id,
                                finish_reason,
                            )
                        parsed_chunk.delta_tool_calls = _finalize_tool_calls(tool_calls)
                        tool_calls_emitted = True

                    if parsed_chunk is not None:
                        # Merge headers after the chunk's body-level provider_metadata (e.g. nvext), not over it.
                        if response_headers:
                            parsed_chunk.provider_metadata = {
                                **(parsed_chunk.provider_metadata or {}),
                                "response_headers": response_headers,
                            }
                        yield parsed_chunk

                # Safety net: a provider that ends the stream with [DONE]/EOF and no
                # finish_reason chunk still gets its accumulated tool calls surfaced.
                if tool_calls and not tool_calls_emitted:
                    yield LLMResponseChunk(
                        finish_reason="tool_calls",
                        request_id=last_chunk_id,
                        delta_tool_calls=_finalize_tool_calls(tool_calls),
                        provider_metadata={"response_headers": response_headers} if response_headers else None,
                    )

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
                inner_exception=LLMResponseValidationError(
                    f"Unexpected response format from the model: {exc}",
                    response_data=response,
                    **self._error_context.as_kwargs(),
                ),
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

    def _merge_params(self, stop: Optional[list[str]], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge the model's configured body defaults under the per-call kwargs.

        ``stop=None`` means "not specified by this call", so a model-level
        ``stop`` default survives rather than being overwritten with ``None``.
        """
        merged = {**self.body_param_defaults, **kwargs}
        if stop is not None:
            merged["stop"] = stop
        return merged

    def _duration_metric(self):
        """Return the operation-duration metric context, or a no-op when metrics are off."""
        if not self._metrics_enabled:
            return nullcontext()
        return llm_operation_duration(self.model_name, self.provider_name, _OPERATION_NAME)

    async def generate_async(
        self,
        prompt: str | list,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion, implementing the ``LLMModel`` protocol.

        Adapter only: it normalizes *prompt* — a plain string or a list of
        ``ChatMessage``, which is what ``llm_call`` passes on behalf of library
        rail actions — into wire messages and delegates.  All the work is in
        ``generate_from_messages``.

        Raises:
            ModelEngineError: If the request fails or the response format is unexpected.
        """
        return await self.generate_from_messages(_to_wire_messages(prompt), stop=stop, **kwargs)

    async def generate_from_messages(
        self,
        messages: LLMMessages,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from OpenAI-format messages.

        The instrumented entry point every caller ends up at: rails through
        ``generate_async``, main generation directly from
        ``EngineRegistry.model_call``, which already holds wire messages and so
        does not need the protocol adapter.  The model's configured
        ``parameters`` supply request-body defaults which *kwargs* override.

        Raises:
            ModelEngineError: If the request fails or the response format is unexpected.
        """
        params = self._merge_params(stop, kwargs)

        # Request params are known before the call, so they land on the span
        # even if the call raises.  Response, usage, and content attrs are set
        # inside the span context so the helpers see the live span; all three
        # are skipped on exception, which never reaches them.
        with llm_call_span(self._tracer, self.model_name, self.provider_name, _OPERATION_NAME) as span:
            set_llm_request_attributes(span, params)
            with self._duration_metric():
                result = await self.chat_completion(messages, **params)
            set_llm_response_attributes(
                span,
                model=result.model,
                response_id=result.request_id,
                finish_reason=result.finish_reason,
                usage=result.usage,
            )
            if self._content_capture_enabled:
                set_llm_call_content(span, messages, result.content)

        if self._metrics_enabled:
            record_token_usage(self.model_name, self.provider_name, _OPERATION_NAME, result.usage)

        return result

    async def stream_async(
        self,
        prompt: str | list,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponseChunk, None]:
        """Stream a completion, implementing the ``LLMModel`` protocol.

        The streaming counterpart of ``generate_async``: normalize *prompt* into
        wire messages, then delegate to ``stream_from_messages``.

        Raises:
            ModelEngineError: If the request fails after all retries.
        """
        stream = self.stream_from_messages(_to_wire_messages(prompt), stop=stop, **kwargs)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            # Close the delegate explicitly.  A consumer that abandons this
            # generator only unwinds *this* one; without the aclose the
            # instrumented core stays suspended at its yield until garbage
            # collection, so the duration metric's `finally` never runs.
            await stream.aclose()

    async def stream_from_messages(
        self,
        messages: LLMMessages,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponseChunk, None]:
        """Stream a completion from OpenAI-format messages.

        The streaming counterpart of ``generate_from_messages``.  Parameter
        merging matches it; the chunk contract is ``stream_call``'s, including
        the terminal usage-only chunk.

        Raises:
            ModelEngineError: If the request fails after all retries.
        """
        params = self._merge_params(stop, kwargs)

        # Providers spread the response fields across the SSE chunks —
        # OpenAI-compatible engines only populate ``usage`` on the terminal
        # chunk (when ``stream_options.include_usage=true``) and finish_reason
        # likewise arrives last — so each field keeps its latest non-None value.
        captured_usage: Optional[UsageInfo] = None
        captured_model: Optional[str] = None
        captured_response_id: Optional[str] = None
        captured_finish_reason: Optional[str] = None
        # Accumulate streamed delta_content here when content capture is on;
        # joined and recorded onto the LLM span at stream end.  The list is
        # allocated unconditionally (cost: one empty list per stream); the
        # per-chunk appends are gated on the flag so the disabled path
        # doesn't carry chunk strings in memory.
        content_parts: list[str] = []

        with llm_call_span(self._tracer, self.model_name, self.provider_name, _OPERATION_NAME) as span:
            set_llm_request_attributes(span, params, stream=True)
            with self._duration_metric():
                # Gate timing-state setup on ``_metrics_enabled`` so the cold
                # path skips ``time.monotonic()`` and the per-chunk bookkeeping
                # entirely.  ``t0`` defaults to ``0.0`` in the disabled path so
                # the type stays a plain ``float`` — it's never read there.
                t0 = time.monotonic() if self._metrics_enabled else 0.0
                last_chunk_time: Optional[float] = None
                async for chunk in self.stream_chat_completion(messages, **params):
                    if self._metrics_enabled:
                        # Per OTEL semconv, "first chunk" / "output chunk" mean
                        # content-bearing chunks — gate on ``delta_content`` /
                        # ``delta_reasoning`` to skip the terminal usage frame
                        # and any other cosmetic SSE events the parser leaves in
                        # place.
                        if chunk.delta_content or chunk.delta_reasoning:
                            now = time.monotonic()
                            if last_chunk_time is None:
                                record_time_to_first_chunk(
                                    self.model_name, self.provider_name, _OPERATION_NAME, now - t0
                                )
                            else:
                                record_time_per_output_chunk(
                                    self.model_name, self.provider_name, _OPERATION_NAME, now - last_chunk_time
                                )
                            last_chunk_time = now
                    # Captured regardless of ``_metrics_enabled`` because they
                    # feed the span's response/usage attrs as well as the metric.
                    if chunk.model is not None:
                        captured_model = chunk.model
                    if chunk.request_id is not None:
                        captured_response_id = chunk.request_id
                    if chunk.finish_reason is not None:
                        captured_finish_reason = chunk.finish_reason
                    if chunk.usage is not None:
                        captured_usage = chunk.usage
                    if self._content_capture_enabled and chunk.delta_content:
                        content_parts.append(chunk.delta_content)
                    yield chunk
            set_llm_response_attributes(
                span,
                model=captured_model,
                response_id=captured_response_id,
                finish_reason=captured_finish_reason,
                usage=captured_usage,
            )
            # Empty ``content_parts`` -> output_text=None so we don't claim an
            # empty assistant response (matches iorails.py's request-span
            # streaming path).
            if self._content_capture_enabled:
                output_text = "".join(content_parts) if content_parts else None
                set_llm_call_content(span, messages, output_text)

        if self._metrics_enabled:
            record_token_usage(self.model_name, self.provider_name, _OPERATION_NAME, captured_usage)

    def parse_tools(self, llm_params: Optional[dict]) -> Toolset:
        """Parse the provider tool block in ``llm_params`` into a ``Toolset``.

        Reads the opaque ``tools`` block forwarded via
        ``GenerationOptions.llm_params`` and normalizes it into the internal
        ``Toolset`` the tool rails validate against, keyed on the model's engine
        (``_TOOL_PARSERS``). OpenAI and NIM share the Chat Completions shape; an
        engine with no registered parser falls back to it. Returns an empty
        ``Toolset`` when no tools are declared.
        """
        tools = (llm_params or {}).get("tools")
        if not tools:
            return Toolset()
        parser = _TOOL_PARSERS.get(self.model_config.engine, _parse_tools_openai)
        return Toolset(tools=parser(tools))

    def extract_tool_results(self, messages: LLMMessages) -> list[ToolResult]:
        """Extract incoming tool results from ``messages`` into ``ToolResult`` objects.

        Pulls the provider's tool-result messages out of the conversation and
        normalizes them into the internal ``ToolResult`` shape the ToolResultRail
        consumes, keyed on the model's engine (``_RESULT_EXTRACTORS``). OpenAI and
        NIM share the Chat Completions shape (``role:"tool"`` messages); an engine
        with no registered extractor falls back to it. Returns an empty list when
        there are no tool results.
        """
        extractor = _RESULT_EXTRACTORS.get(self.model_config.engine, _extract_tool_results_openai)
        return extractor(messages)

    def extract_tool_exchanges(self, messages: LLMMessages) -> list[ToolExchange]:
        """Group ``messages`` into per-turn ``(tool_calls, tool_results)`` exchanges.

        Each exchange pairs one assistant turn's tool calls with the tool results that
        answer it, so ``RailsManager.are_tool_results_safe`` can validate ``call_id``
        linkage turn-locally rather than across the whole flattened history (the latter
        falsely flags ids reused across turns, which the OpenAI spec permits). Keyed on
        the model's engine (``_TOOL_EXCHANGE_EXTRACTORS``); OpenAI and NIM share the Chat
        Completions shape and an engine with no registered extractor falls back to it.
        """
        extractor = _TOOL_EXCHANGE_EXTRACTORS.get(self.model_config.engine, _extract_tool_exchanges_openai)
        return extractor(messages)
