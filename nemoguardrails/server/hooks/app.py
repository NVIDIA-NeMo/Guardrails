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

"""The hooks server: checks the tool call in each agent hook with one config's tool-call rails, on IORails."""

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.responses import JSONResponse, Response

from nemoguardrails.exceptions import ConfigurationError, LLMCallException, NonStreamingWorkQueueFullError
from nemoguardrails.guardrails.guardrails import Guardrails
from nemoguardrails.guardrails.guardrails_types import quoted_identity
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.guardrails.tool_schema import Toolset, validate_arguments
from nemoguardrails.http.errors import HTTPClientError
from nemoguardrails.llm.clients._errors import client_facing_message
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailType, ToolViolationType
from nemoguardrails.server.exception_handlers import (
    _error_response,
    http_exception_handler,
    internal_error_handler,
    queue_full_error_handler,
    validation_error_handler,
)
from nemoguardrails.server.hooks.adapters import ADAPTERS, HookAdapter, HookAgent, UnsupportedHookEventError
from nemoguardrails.server.hooks.checking import HookOutcome, ToolInvocation, check_messages, outcome_from_result
from nemoguardrails.server.hooks.schemas import HookErrorCode
from nemoguardrails.telemetry import DeploymentTypeEnum, set_deployment_type

log = logging.getLogger(__name__)

# What `llm_call_exception_handler` answers on the main server. A rail's model call that fails with an upstream
# HTTP status propagates out of `check_async()` rather than blocking, but here it is a failed rail all the same.
_MODEL_CALL_ERRORS = (LLMCallException, ModelEngineError, HTTPClientError)


class HooksConfigError(ConfigurationError):
    """A config the hooks server can't serve, found before it answers any request."""


def check_hooks_config(config: RailsConfig) -> None:
    """Raise `HooksConfigError` unless *config* has tool-call rails and a `main` model, and IORails can run it."""
    reason = IORails.unconfigured_rail_type_reason(config, RailType.TOOL_CALL)
    if reason is not None:
        raise HooksConfigError(f"The hooks server needs tool-call rails and a `main` model. {reason}")
    reason = IORails.unsupported_reason(config)
    if reason is not None:
        raise HooksConfigError(f"The hooks server runs on IORails, which can't serve this config: {reason}")


def create_hooks_app(config: RailsConfig) -> FastAPI:
    """Build the hooks server for *config*, with its own engine; raises `HooksConfigError` if it can't serve hooks."""
    check_hooks_config(config)
    # IORails reports usage when it is built, so the deployment type is claimed before that.
    set_deployment_type(DeploymentTypeEnum.API.value)
    engine = _build_engine(config)

    # No docs or OpenAPI routes: the server answers only the hooks and the health checks.
    app = FastAPI(
        title="NeMo Guardrails Hooks API",
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.engine = engine
    for exc_type, handler in _EXCEPTION_HANDLERS:
        # Handlers are typed with their specific exception; Starlette's stub expects
        # (Request, Exception), so ty flags the narrower signature as a false positive.
        app.add_exception_handler(exc_type, handler)  # ty: ignore[invalid-argument-type]
    app.add_middleware(ExceptionMiddleware, handlers={Exception: internal_error_handler})
    app.include_router(_router)
    return app


def _build_engine(config: RailsConfig) -> Guardrails:
    """The server's IORails engine for *config*, never LLMRails, which can't run tool checks."""
    try:
        engine = Guardrails(config, require_iorails=True)
    except ValueError as exc:
        raise HooksConfigError(f"The hooks server can't build its engine: {exc}") from exc
    rails_engine = engine.rails_engine
    if not isinstance(rails_engine, IORails):
        raise HooksConfigError("The hooks server's engine isn't IORails, the only engine that runs tool checks")
    declared = _declared_tools(rails_engine)
    _check_tool_schemas(declared)
    _warn_about_undeclared_tools(config, declared)
    return engine


def _declared_tools(engine: IORails) -> Toolset:
    """The tools the `main` model declares, as the tool-call rails read them."""
    try:
        return engine.engine_registry.parse_tools("main", None)
    except Exception as exc:
        # A check whose tools don't parse blocks every call, so the server refuses to start instead.
        raise HooksConfigError(f"The hooks server can't read the `main` model's tools: {exc}") from exc


def _check_tool_schemas(declared: Toolset) -> None:
    """Raise `HooksConfigError` for a declared tool whose schema isn't valid JSON Schema, which fails every call to it."""
    for tool in declared.tools:
        # The per-call check, so startup refuses exactly the schemas a call would fail on. Empty arguments are enough,
        # since the schema is checked before the arguments are.
        violation = validate_arguments(tool, {})
        if violation is not None and violation.violation_type == ToolViolationType.INVALID_TOOL_SCHEMA:
            raise HooksConfigError(f"The hooks server can't use the `main` model's tools: {violation.reason}")


def _warn_about_undeclared_tools(config: RailsConfig, declared: Toolset) -> None:
    """Warn about each tool-call rail that denies every call, because the `main` model doesn't declare the tool."""
    tool_call_rails = config.rails.tool_call
    # IORails accepts no global tool-call flow but `tool call validation`.
    if tool_call_rails.flows and not declared.tools:
        log.warning(
            "`tool call validation` is in rails.tool_call.flows, but the `main` model declares no tools, "
            "so every tool call is denied"
        )
    for tool_name, flows in tool_call_rails.per_tool.items():
        if flows and declared.get(tool_name) is None:
            log.warning(
                "rails.tool_call.per_tool has rails for %r, but the `main` model doesn't declare that tool, "
                "so every call to it is denied",
                tool_name,
            )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the engine before the server accepts requests, and stop it on shutdown."""
    engine: Guardrails = app.state.engine
    await engine.startup()
    try:
        yield
    finally:
        await engine.shutdown()


_router = APIRouter()


@_router.post(
    "/v1/hooks/claude-code",
    name="handleClaudeCodeHook",
    summary="Check a tool call from a Claude Code hook",
    tags=["Hooks"],
)
async def handle_claude_code_hook(request: Request) -> Response:
    """Answer a Claude Code hook in Claude Code's hook output format."""
    return await _answer_hook("claude-code", request)


@_router.post(
    "/v1/hooks/codex",
    name="handleCodexHook",
    summary="Check a tool call from a Codex hook",
    tags=["Hooks"],
)
async def handle_codex_hook(request: Request) -> Response:
    """Answer a Codex hook in Codex's hook output format."""
    return await _answer_hook("codex", request)


@_router.post(
    "/v1/hooks/nemo-relay",
    name="handleNemoRelayHook",
    summary="Check a tool call from a NeMo Relay guardrail",
    tags=["Hooks"],
)
async def handle_nemo_relay_hook(request: Request) -> Response:
    """Answer a NeMo Relay tool conditional-execution guardrail with its decision."""
    return await _answer_hook("nemo-relay", request)


@_router.get("/v1/health", summary="Liveness health check.", tags=["Health"])
@_router.get("/healthz", summary="Liveness health check.", tags=["Health"])
async def health() -> Response:
    """Return HTTP 200 while the server process is running and able to serve requests."""
    return JSONResponse(content={"status": "pass"}, media_type="application/health+json")


async def _answer_hook(agent: HookAgent, request: Request) -> Response:
    """Check the tool call the hook carries, and answer with the agent's allow or deny, or an error status."""
    adapter = ADAPTERS[agent]
    call = _parse_hook(adapter, await _read_json(request))
    outcome = await _check_tool_call(request.app.state.engine, adapter, call)
    if outcome.kind == "error":
        return _error_response(500, outcome.message or "", code=outcome.code)
    return JSONResponse(adapter.response_body(outcome))


async def _read_json(request: Request) -> object:
    """The request body as JSON, whatever its `Content-Type` says."""
    try:
        return await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Raised from None, because the decode error keeps the raw body.
        raise RequestValidationError(
            [{"type": "json_invalid", "loc": ("body",), "msg": "JSON decode error", "input": {}}]
        ) from None


def _parse_hook(adapter: HookAdapter, body: object) -> ToolInvocation:
    """The tool call *body* asks to check on *adapter*'s path, raising a failure for the shared `422` handler."""
    try:
        return adapter.parse(body)
    except ValidationError as exc:
        # Without the errors' `input`, and raised from None: the raw values, `tool_input` among them, stay out of logs.
        errors = exc.errors(include_url=False, include_input=False)
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in errors]) from None


async def _check_tool_call(engine: Guardrails, adapter: HookAdapter, call: ToolInvocation) -> HookOutcome:
    """Run the config's tool-call rails on the hook's call, and log the verdict."""
    try:
        result = await engine.check_async(check_messages(call), rail_types=[RailType.TOOL_CALL])
    except _MODEL_CALL_ERRORS as exc:
        outcome = HookOutcome.error(f"a tool-call rail failed: {client_facing_message(exc)}", HookErrorCode.RAIL_FAILED)
        _log_check(adapter, call, outcome, rail=None)
        return outcome
    outcome = outcome_from_result(result)
    _log_check(adapter, call, outcome, rail=result.rail)
    return outcome


def _log_check(adapter: HookAdapter, call: ToolInvocation, outcome: HookOutcome, rail: Optional[str]) -> None:
    """Log one line for a check, without the tool's input; an error, which no rail decided, as a warning."""
    log.log(
        logging.WARNING if outcome.kind == "error" else logging.INFO,
        "Hook check on /v1/hooks/%s: event=%s tool_name=%r tool_use_id=%r outcome=%s code=%s rail=%r",
        adapter.agent,
        adapter.event,
        quoted_identity(call.name),
        None if call.call_id is None else quoted_identity(call.call_id),
        outcome.kind,
        None if outcome.code is None else outcome.code.value,
        rail,
    )


async def _unsupported_hook_event_handler(request: Request, exc: UnsupportedHookEventError) -> Response:
    """Answer a `hook_event_name` the path doesn't handle with `422 unsupported_hook_event`."""
    log.warning("Unsupported hook event: %s", exc)
    return _error_response(422, str(exc), code=HookErrorCode.UNSUPPORTED_HOOK_EVENT, param="hook_event_name")


_EXCEPTION_HANDLERS = (
    (UnsupportedHookEventError, _unsupported_hook_event_handler),
    (NonStreamingWorkQueueFullError, queue_full_error_handler),
    (RequestValidationError, validation_error_handler),
    (StarletteHTTPException, http_exception_handler),
    (Exception, internal_error_handler),
)
