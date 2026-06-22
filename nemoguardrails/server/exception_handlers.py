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
import re
from typing import Optional, Union

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.llm.clients._errors import _redact_secrets
from nemoguardrails.llm.models.initializer import ModelInitializationError

log = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://[^\s)\"']+")


def _sanitize(message: str) -> str:
    """Strip secrets and upstream URLs from a client-facing error message."""
    return _URL_PATTERN.sub("[redacted-url]", _redact_secrets(message))


_STATUS_TO_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
}


def _error_type_for_status(status_code: int) -> str:
    if status_code in _STATUS_TO_ERROR_TYPE:
        return _STATUS_TO_ERROR_TYPE[status_code]
    if status_code >= 500:
        return "server_error"
    return "api_error"


def _error_response(
    status_code: int,
    message: str,
    error_type: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    """Build an OpenAI-compatible error response.

    Uses the ``{"error": {"message", "type", "param", "code"}}`` envelope that
    the OpenAI SDK reads on every endpoint. The HTTP status code is the source
    of truth; ``code`` stays null unless a specific error code is known.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type or _error_type_for_status(status_code),
                "param": None,
                "code": code,
            }
        },
    )


async def llm_call_exception_handler(
    request: Request, exc: Union[LLMCallException, ModelEngineError, APIEngineError]
) -> Response:
    """Map LLM and engine call failures to their upstream HTTP status."""
    log.exception(exc)
    status = getattr(exc, "status", None) or 500
    return _error_response(status, _sanitize(str(exc)))


async def model_initialization_error_handler(request: Request, exc: ModelInitializationError) -> Response:
    """Return 400 when a model fails to initialize from the configuration."""
    log.exception(exc)
    return _error_response(400, _sanitize(str(exc)))


async def validation_error_handler(request: Request, exc: RequestValidationError) -> Response:
    """Return 422 for request body validation failures."""
    log.error("Request validation failed: %s", exc)
    return _error_response(422, _sanitize(str(exc)))


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """Render HTTPException (404, 422 guards, upstream 502, etc.) as the error envelope."""
    log.info("HTTP %s: %s", exc.status_code, exc.detail)
    return _error_response(exc.status_code, _sanitize(str(exc.detail)))


async def internal_error_handler(request: Request, exc: Exception) -> Response:
    """Catch-all for unexpected errors."""
    log.exception(exc)
    return _error_response(500, "Internal server error")
