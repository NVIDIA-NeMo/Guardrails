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
from typing import Optional, Union

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.llm.clients._errors import build_error_payload
from nemoguardrails.llm.models.initializer import ModelInitializationError

log = logging.getLogger(__name__)


def _error_response(status_code: int, message: str, error_type: Optional[str] = None) -> JSONResponse:
    """Render the shared OpenAI error envelope as a JSON HTTP response.

    The HTTP status code is the source of truth, so ``code`` stays null here.
    """
    return JSONResponse(
        status_code=status_code,
        content=build_error_payload(message, status=status_code, error_type=error_type),
    )


async def llm_call_exception_handler(
    request: Request, exc: Union[LLMCallException, ModelEngineError, APIEngineError]
) -> Response:
    """Map LLM and engine call failures to their upstream HTTP status."""
    log.exception(exc)
    status = getattr(exc, "status", None) or 500
    return _error_response(status, str(exc))


async def model_initialization_error_handler(request: Request, exc: ModelInitializationError) -> Response:
    """Return 400 when a model fails to initialize from the configuration."""
    log.exception(exc)
    return _error_response(400, str(exc))


async def validation_error_handler(request: Request, exc: RequestValidationError) -> Response:
    """Return 422 for request body validation failures."""
    log.error("Request validation failed: %s", exc)
    return _error_response(422, str(exc))


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """Render HTTPException (404, 422 guards, upstream 502, etc.) as the error envelope."""
    log.info("HTTP %s: %s", exc.status_code, exc.detail)
    return _error_response(exc.status_code, str(exc.detail))


async def internal_error_handler(request: Request, exc: Exception) -> Response:
    """Catch-all for unexpected errors."""
    log.exception(exc)
    return _error_response(500, "Internal server error")
