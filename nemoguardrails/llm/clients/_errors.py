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

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Tuple, Union

from nemoguardrails.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMClientError,
    LLMContextWindowError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnsupportedParamsError,
)

_CONTEXT_WINDOW_KEYWORDS = [
    "context length",
    "context_length",
    "context window",
    "maximum token",
    "max_tokens",
    "too many tokens",
    "token limit",
]

# Bare "is not supported" was deliberately removed: it false-positives on
# non-param 400s ("model is not supported in your region", "image input is not
# supported for this model"). Real OpenAI param rejections always carry the
# "Unsupported parameter:" prefix matched above. A provider emitting bare
# "X is not supported" without that prefix will classify as LLMBadRequestError
# instead of LLMUnsupportedParamsError; if observed in the wild, add a tighter
# phrase here (e.g. "is not a supported parameter").
_UNSUPPORTED_PARAMS_KEYWORDS = [
    "unsupported parameter",
    "parameter not allowed",
    "unknown parameter",
    "unrecognized parameter",
    "unrecognized request argument",
    "' is unsupported",
    "extra inputs are not permitted",
]

_UNKNOWN_PARAM_HINT_TOKENS = (
    "unrecognized request argument",
    "unsupported parameter",
    "' is unsupported",
    "extra inputs are not permitted",
)

_MIGRATION_HINT_021 = (
    "(If you upgraded from 0.21: the default framework forwards `parameters` "
    "verbatim to the OpenAI-compatible endpoint, which rejected the field above. "
    "LangChain-only flags must be removed for the default framework. To keep "
    "0.21 LangChain behavior, set NEMOGUARDRAILS_LLM_FRAMEWORK=langchain.)"
)

_SECRET_PATTERN = re.compile(r"(sk-|nvapi-|AIza|bearer\s+)\S+", re.IGNORECASE)


@dataclass(frozen=True)
class ErrorContext:
    model_name: Optional[str] = None
    provider_name: Optional[str] = None
    base_url: Optional[str] = None

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "provider_name": self.provider_name,
            "base_url": self.base_url,
        }


_EMPTY_CONTEXT = ErrorContext()


def _redact_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub(lambda m: m.group(1) + "***", text)


_URL_PATTERN = re.compile(r"https?://[^\s)\"']+")


def _sanitize(message: str) -> str:
    """Scrub a client-facing error message: redact secrets and upstream URLs."""
    return _URL_PATTERN.sub("[redacted-url]", _redact_secrets(message))


# Outbound HTTP status -> OpenAI error ``type``. This is the response-formatting
# counterpart to ``_SSE_ERROR_TYPE_TO_STATUS`` below (which parses inbound
# provider SSE error types into statuses); keep the two in sync.
_STATUS_TO_ERROR_TYPE: Dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
}


def _error_type_for_status(status_code: Optional[int]) -> str:
    if status_code in _STATUS_TO_ERROR_TYPE:
        return _STATUS_TO_ERROR_TYPE[status_code]
    if status_code and status_code >= 500:
        return "server_error"
    return "api_error"


def normalize_error_status(status: Any) -> int:
    if isinstance(status, int) and 400 <= status < 600:
        return status
    return 500


def build_error_payload(
    message: str,
    *,
    status: Optional[int] = None,
    error_type: Optional[str] = None,
    code: Union[str, int, None] = None,
    param: Optional[str] = None,
) -> Dict[str, Any]:
    """Single source of truth for the OpenAI error envelope.

    Always sanitizes the message. ``type`` defaults to the OpenAI category for
    ``status``; ``code`` and ``param`` stay null unless the caller supplies them
    (the streaming paths pass the status/marker as ``code`` because an SSE
    response has no HTTP status line to carry it, and the rail-violation path
    passes the blocking rail family as ``param``).
    """
    return {
        "error": {
            "message": _sanitize(message),
            "type": error_type or _error_type_for_status(status),
            "param": param,
            "code": code,
        }
    }


# Internal stream error markers. A chunk only counts as a terminal error frame
# when its ``type`` is one of these, so model-generated content that merely
# looks like an OpenAI error object cannot truncate a stream or skip output
# rails. ``downstream_error`` carries an upstream HTTP status;
# ``generation_error`` is a status-less generation failure;
# ``guardrails_violation`` is emitted when a rail blocks.
GENERATION_ERROR_TYPE = "generation_error"
DOWNSTREAM_ERROR_TYPE = "downstream_error"
GUARDRAILS_VIOLATION_TYPE = "guardrails_violation"

STREAM_ERROR_TYPES = frozenset(
    {
        GENERATION_ERROR_TYPE,
        DOWNSTREAM_ERROR_TYPE,
        GUARDRAILS_VIOLATION_TYPE,
    }
)


def as_client_error(exception: BaseException) -> Optional[LLMClientError]:
    """Return the underlying :class:`LLMClientError`, unwrapping ``LLMCallException``."""
    inner = getattr(exception, "inner_exception", None)
    if isinstance(inner, LLMClientError):
        return inner
    if isinstance(exception, LLMClientError):
        return exception
    return None


def client_facing_message(exception: BaseException) -> str:
    """The message to show an API caller for a provider failure.

    Prefers ``LLMClientError.error_message`` over ``str(exception)``: the latter
    is prefixed with the internal rail model, provider, and endpoint, which must
    not be disclosed to the caller.
    """
    client_error = as_client_error(exception)
    return client_error.error_message if client_error is not None else str(exception)


def build_streaming_error_payload(exception: BaseException) -> str:
    """Build the JSON error chunk pushed into a stream when generation fails.

    Shared by both streaming backends so they emit the same envelope. A carried
    HTTP status marks a downstream failure; otherwise any provider-supplied
    ``type``/``code`` recovered from the message is preserved, falling back to
    the generation markers. The ``type`` is always one of
    :data:`STREAM_ERROR_TYPES` so the terminal-chunk detector recognizes it.
    """
    from nemoguardrails.utils import extract_error_json

    status = getattr(exception, "status", None)
    raw_message = client_facing_message(exception)
    extracted = extract_error_json(raw_message)
    inner = extracted.get("error") if isinstance(extracted, dict) else None

    if isinstance(inner, dict):
        message = inner.get("message") or raw_message
        provider_code = inner.get("code")
    else:
        message = inner if isinstance(inner, str) and inner else raw_message
        provider_code = None

    if status is not None:
        status = normalize_error_status(status)
        error_type = DOWNSTREAM_ERROR_TYPE
        code: Union[str, int, None] = status
    else:
        error_type = GENERATION_ERROR_TYPE
        code = provider_code if provider_code is not None else "generation_failed"

    return json.dumps(build_error_payload(message, status=status, error_type=error_type, code=code))


def _parse_retry_after_value(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(tz=timezone.utc)).total_seconds()


def _parse_retry_after(headers: Any) -> Optional[float]:
    raw = headers.get("retry-after") if headers else None
    if not raw:
        return None
    return _parse_retry_after_value(raw)


def _extract_from_parsed_body(parsed_body: Any) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    error_message = ""
    error_type = None
    error_code = None
    param = None
    if isinstance(parsed_body, dict):
        error_obj = parsed_body.get("error", {})
        if isinstance(error_obj, dict):
            error_message = error_obj.get("message", "") or ""
            error_type = error_obj.get("type")
            error_code = error_obj.get("code")
            param = error_obj.get("param")
        elif isinstance(error_obj, str):
            error_message = error_obj
        if not error_message:
            error_message = parsed_body.get("message") or parsed_body.get("detail") or ""
    return error_message, error_type, error_code, param


def _build_error_fields(parsed_body: Any, raw_body: str, headers: Any, ctx: ErrorContext) -> Tuple[str, Dict[str, Any]]:
    error_message, error_type, error_code, param = _extract_from_parsed_body(parsed_body)
    if not error_message:
        error_message = raw_body or ""
    error_message = _redact_secrets(error_message)
    kwargs = dict(
        error_type=error_type,
        error_code=error_code,
        param=param,
        body=parsed_body,
        response_headers=dict(headers) if headers else None,
        **ctx.as_kwargs(),
    )
    return error_message, kwargs


def _looks_like_unknown_param_400(error_message: str) -> bool:
    msg_lower = error_message.lower()
    return any(token in msg_lower for token in _UNKNOWN_PARAM_HINT_TOKENS)


def _maybe_append_migration_hint(error_message: str) -> str:
    if not _looks_like_unknown_param_400(error_message):
        return error_message
    return f"{error_message}\n\n{_MIGRATION_HINT_021}"


def _classify_bad_request(status_code: int, error_message: str, kwargs: Dict[str, Any]) -> LLMClientError:
    msg_lower = error_message.lower()
    if any(kw in msg_lower for kw in _CONTEXT_WINDOW_KEYWORDS):
        return LLMContextWindowError(status_code, error_message, **kwargs)
    if any(kw in msg_lower for kw in _UNSUPPORTED_PARAMS_KEYWORDS):
        if "stream_options" in msg_lower:
            error_message = (
                f"{error_message} (set include_usage_in_stream=False on the model "
                "or in config.yml parameters to remove this field from streaming requests)"
            )
        else:
            error_message = _maybe_append_migration_hint(error_message)
        return LLMUnsupportedParamsError(status_code, error_message, **kwargs)
    return LLMBadRequestError(status_code, error_message, **kwargs)


def raise_for_status(status_code: int, body: str, headers: Any, ctx: Optional[ErrorContext] = None) -> None:
    ctx = ctx or _EMPTY_CONTEXT
    try:
        parsed_body = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        parsed_body = None

    error_message, kwargs = _build_error_fields(parsed_body, body, headers, ctx)
    if not error_message:
        error_message = f"HTTP {status_code}"

    if status_code in (401, 403):
        raise LLMAuthenticationError(status_code, error_message, **kwargs)

    if status_code == 408:
        raise LLMTimeoutError(status_code, error_message, **kwargs)

    if status_code == 429:
        retry_after = _parse_retry_after(headers)
        raise LLMRateLimitError(status_code, error_message, **kwargs, retry_after_seconds=retry_after)

    if status_code == 400 or status_code == 422:
        raise _classify_bad_request(status_code, error_message, kwargs)

    if status_code >= 500:
        raise LLMServerError(status_code, error_message, **kwargs)

    raise LLMClientError(status_code, error_message, **kwargs)


# Inbound provider SSE error ``type`` -> HTTP status. Response-formatting
# counterpart is ``_STATUS_TO_ERROR_TYPE`` above; keep the two in sync.
_SSE_ERROR_TYPE_TO_STATUS: Dict[str, int] = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "rate_limit_error": 429,
    "api_error": 500,
    "server_error": 500,
    "overloaded_error": 503,
}


def raise_for_sse_error(parsed_payload: Dict[str, Any], headers: Any, ctx: Optional[ErrorContext] = None) -> None:
    ctx = ctx or _EMPTY_CONTEXT
    error_obj = parsed_payload.get("error")
    error_type = error_obj.get("type") if isinstance(error_obj, dict) else None
    error_code = error_obj.get("code") if isinstance(error_obj, dict) else None

    status: Optional[int] = None
    if isinstance(error_type, str) and error_type in _SSE_ERROR_TYPE_TO_STATUS:
        status = _SSE_ERROR_TYPE_TO_STATUS[error_type]
    elif isinstance(error_code, int) and 400 <= error_code < 600:
        status = error_code

    if status is not None:
        raise_for_status(status, json.dumps(parsed_payload), headers, ctx)

    error_message, kwargs = _build_error_fields(parsed_payload, json.dumps(parsed_payload), headers, ctx)
    if not error_message:
        error_message = "Streaming error"
    raise LLMClientError(0, error_message, **kwargs)
