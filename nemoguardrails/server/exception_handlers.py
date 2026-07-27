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

import logging
from typing import Dict, Optional, Union

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.utils import is_body_allowed_for_status_code
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from nemoguardrails.exceptions import (
    InvalidStateError,
    LLMCallException,
    LLMRateLimitError,
    StreamingNotSupportedError,
)
from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.llm.clients._errors import (
    as_client_error,
    build_error_payload,
    client_facing_message,
    normalize_error_status,
)
from nemoguardrails.llm.models.initializer import ModelInitializationError

log = logging.getLogger(__name__)


def _error_response(
    status_code: int,
    message: str,
    error_type: Optional[str] = None,
    *,
    code: Union[str, int, None] = None,
    param: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """Render the shared OpenAI error envelope as a JSON HTTP response."""
    return JSONResponse(
        status_code=status_code,
        content=build_error_payload(message, status=status_code, error_type=error_type, code=code, param=param),
        headers=headers,
    )


def _client_error_details(exc: BaseException) -> tuple[str, Union[str, int, None], Optional[str], Dict[str, str]]:
    """Extract the client-facing message and the OpenAI error fields from an exception.

    ``code`` and ``param`` come from the provider when it supplied them, and a
    rate limit forwards its ``Retry-After`` so SDK backoff is not blind.
    """
    client_error = as_client_error(exc)
    if client_error is None:
        return str(exc), None, None, {}

    headers: Dict[str, str] = {}
    retry_after = getattr(client_error, "retry_after_seconds", None)
    if isinstance(client_error, LLMRateLimitError) and retry_after is not None:
        headers["retry-after"] = str(int(max(retry_after, 0)))

    return client_facing_message(exc), client_error.error_code, client_error.param, headers


async def llm_call_exception_handler(
    request: Request, exc: Union[LLMCallException, ModelEngineError, APIEngineError]
) -> Response:
    """Map LLM and engine call failures to their upstream HTTP status."""
    log.exception(exc)
    status = normalize_error_status(getattr(exc, "status", None))
    message, code, param, headers = _client_error_details(exc)
    return _error_response(status, message, code=code, param=param, headers=headers or None)


async def model_initialization_error_handler(request: Request, exc: ModelInitializationError) -> Response:
    """Return 400 when a model fails to initialize from the configuration."""
    log.exception(exc)
    return _error_response(400, str(exc))


async def bad_request_error_handler(
    request: Request, exc: Union[StreamingNotSupportedError, InvalidStateError]
) -> Response:
    """Return 400 for request/config combinations the caller can correct.

    These carry an actionable message (for example "enable streaming output
    rails"), so they must not fall through to the 500 catch-all, which would
    both hide the message and invite an SDK retry.
    """
    log.warning("Bad request: %s", exc)
    return _error_response(400, str(exc))


async def validation_error_handler(request: Request, exc: RequestValidationError) -> Response:
    """Return 422 for request body validation failures.

    Only the field locations and messages are reported. ``str(exc)`` is the repr
    of pydantic's error list, which embeds the raw request body (prompts,
    tokens, PII), so it must reach neither the client nor the server log.
    """
    errors = exc.errors()
    summary = "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', '')}" for error in errors
    )
    log.error("Request validation failed: %s", summary)
    first_loc = errors[0].get("loc") if errors else None
    if first_loc and first_loc[0] == "body":
        # OpenAI reports the field name alone (`model`, not `body.model`).
        first_loc = first_loc[1:]
    param = ".".join(str(part) for part in first_loc) if first_loc else None
    return _error_response(422, summary or "Request validation failed", param=param)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """Render HTTPException (404, 422 guards, upstream 502, etc.) as the error envelope.

    ``exc.headers`` is forwarded because HTTP requires some of them (``Allow``
    on 405, ``WWW-Authenticate`` on 401), and statuses that disallow a body get
    an empty response rather than an envelope.
    """
    log.info("HTTP %s: %s", exc.status_code, exc.detail)
    headers = getattr(exc, "headers", None)
    if not is_body_allowed_for_status_code(exc.status_code):
        return Response(status_code=exc.status_code, headers=headers)
    return _error_response(exc.status_code, str(exc.detail), headers=headers)


async def internal_error_handler(request: Request, exc: Exception) -> Response:
    """Catch-all for unexpected errors."""
    log.exception(exc)
    return _error_response(500, "Internal server error")
