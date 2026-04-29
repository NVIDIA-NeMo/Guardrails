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
from typing import Any, Dict, Optional, Tuple

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

_UNSUPPORTED_PARAMS_KEYWORDS = [
    "unsupported parameter",
    "is not supported",
    "parameter not allowed",
    "unknown parameter",
    "unrecognized parameter",
]

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
