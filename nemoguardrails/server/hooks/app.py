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
from typing import AsyncIterator, Mapping, Optional

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
from nemoguardrails.http.errors import HTTPClientError
from nemoguardrails.llm.clients._errors import client_facing_message
from nemoguardrails.rails.llm.config import Model, RailsConfig, _get_flow_model
from nemoguardrails.rails.llm.options import RailType
from nemoguardrails.server.exception_handlers import (
    _error_response,
    http_exception_handler,
    internal_error_handler,
    queue_full_error_handler,
    validation_error_handler,
)
from nemoguardrails.server.hooks.adapters import ADAPTERS, HookAdapter, HookAgent, UnsupportedHookEventError
from nemoguardrails.server.hooks.checking import (
    HookOutcome,
    ToolInvocation,
    check_messages,
    check_tools,
    outcome_from_result,
)
from nemoguardrails.server.hooks.schemas import HookErrorCode
from nemoguardrails.server.hooks.tool_policy import ToolPolicy
from nemoguardrails.server.metrics import MetricsExporterSettings, running_metrics_exporter
from nemoguardrails.telemetry import DeploymentTypeEnum, set_deployment_type

log = logging.getLogger(__name__)

# What `llm_call_exception_handler` answers on the main server. A rail's model call that fails with an upstream
# HTTP status propagates out of `check_async()` rather than blocking, but here it is a failed rail all the same.
_MODEL_CALL_ERRORS = (LLMCallException, ModelEngineError, HTTPClientError)

# What the log line names as deciding a call the tool policy denies, in place of a rail.
_TOOL_POLICY = "tool policy"

# A port and service name of its own, so it exports beside the main server on one host, and loopback, like its API.
_METRICS_DEFAULTS = MetricsExporterSettings(host="127.0.0.1", port=9465, service_name="nemoguardrails-hooks-server")


def metrics_exporter_settings(environ: Optional[Mapping[str, str]] = None) -> MetricsExporterSettings:
    """The `NEMO_GUARDRAILS_SERVER_METRICS_*` exporter settings, with the hooks server's defaults for those unset."""
    return MetricsExporterSettings.from_env(environ, defaults=_METRICS_DEFAULTS)


class HooksConfigError(ConfigurationError):
    """A config the hooks server can't serve, found before it answers any request."""


def check_hooks_config(config: RailsConfig) -> None:
    """Raise `HooksConfigError` unless IORails can serve *config*, it can deny a call, and its rails and `main` run."""
    reason = IORails.unsupported_reason(config)
    if reason is not None:
        raise HooksConfigError(f"The hooks server runs on IORails, which can't serve this config: {reason}")
    _check_no_global_tool_call_rails(config)
    _check_something_can_deny(config)
    _check_per_tool_rails_can_run(config, ToolPolicy.from_config(config.hooks_server))
    _check_main_model(config)


def _check_no_global_tool_call_rails(config: RailsConfig) -> None:
    """Raise when rails.tool_call.flows lists a rail, which can't deny a call the tool policy allows."""
    # IORails' one global tool-call rail checks a call against the tools a check passes, and the server passes the
    # called tool, so the rail always finds it.
    if config.rails.tool_call.flows:
        raise HooksConfigError(
            f"rails.tool_call.flows lists '{config.rails.tool_call.flows[0]}', which can't deny a call on the hooks "
            "server: the server applies its tool policy and declares the called tool itself, so only per-tool rails "
            "check a call. Restrict tools with hooks_server.allowed_tools and hooks_server.allow_unlisted_tools: "
            "false, and remove the flow."
        )


def _check_something_can_deny(config: RailsConfig) -> None:
    """Raise unless the tool policy or a per-tool rail can deny a call, since otherwise every call is allowed."""
    if config.hooks_server.can_deny or _has_per_tool_rails(config):
        return
    if config.hooks_server.allowed_tools:
        raise HooksConfigError(
            "hooks_server.allowed_tools names tools by name only while hooks_server.allow_unlisted_tools is true, "
            "so the tool policy denies nothing, and with no per-tool rails every tool call would be allowed; set "
            "hooks_server.allow_unlisted_tools: false."
        )
    raise HooksConfigError(
        "The hooks server needs a tool policy in the hooks_server section, or per-tool rails under "
        "rails.tool_call.per_tool; this config has neither, so every tool call would be allowed."
    )


def _has_per_tool_rails(config: RailsConfig) -> bool:
    """Whether *config* lists a per-tool rail for any tool's calls."""
    return any(config.rails.tool_call.per_tool.values())


def _check_per_tool_rails_can_run(config: RailsConfig, policy: ToolPolicy) -> None:
    """Raise when the tool policy denies every call to a tool that per-tool rails are configured for."""
    for name, flows in config.rails.tool_call.per_tool.items():
        if not flows:
            continue
        if policy.blocked_by_name(name):
            raise HooksConfigError(
                f"rails.tool_call.per_tool has rails for tool '{name}', which hooks_server.blocked_tools blocks by "
                "name, so they never run; remove its per_tool entry, or unblock it."
            )
        if policy.denied_by_name(name):
            raise HooksConfigError(
                f"rails.tool_call.per_tool has rails for tool '{name}', which hooks_server.allowed_tools doesn't list "
                "while hooks_server.allow_unlisted_tools is false, so they never run; add it to "
                "hooks_server.allowed_tools."
            )


def _check_main_model(config: RailsConfig) -> None:
    """Raise unless a tool-call rail uses any `main` model, and tool lists replace the tools it declares."""
    main = _main_model(config)
    if main is None:
        return
    if not _tool_call_rails_use_model(config, "main"):
        raise HooksConfigError(
            "The config has a `main` model, which the hooks server doesn't use: it checks tool calls and never "
            "generates a response, and no tool-call rail names `$model=main`. Remove the `main` model, or give the "
            "hooks server a config of its own."
        )
    # Refused rather than warned about: a config that declared its allowlist on `main` would allow every tool here.
    if main.parameters.get("tools") and not config.hooks_server.can_deny:
        raise HooksConfigError(
            "The `main` model declares tools in parameters.tools, which the hooks server ignores, and the hooks_server "
            "section sets no tool policy that can deny a call, so every tool call would be allowed. List the tools in "
            "hooks_server.allowed_tools with allow_unlisted_tools: false, or remove parameters.tools."
        )


def create_hooks_app(config: RailsConfig) -> FastAPI:
    """Build the hooks server for *config*, with its own engine; raises `HooksConfigError` if it can't serve hooks."""
    check_hooks_config(config)
    _warn_about_ignored_main_model_tools(config)
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
    app.state.tool_policy = ToolPolicy.from_config(config.hooks_server)
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
    if not isinstance(engine.rails_engine, IORails):
        raise HooksConfigError("The hooks server's engine isn't IORails, the only engine that runs tool checks")
    return engine


def _warn_about_ignored_main_model_tools(config: RailsConfig) -> None:
    """Warn that the hooks server ignores the tools a `main` model declares, since the tool policy replaces them."""
    main = _main_model(config)
    if main is not None and main.parameters.get("tools"):
        log.warning(
            "The `main` model declares tools in parameters.tools, which the hooks server ignores: it checks each "
            "tool call against hooks_server.allowed_tools, blocked_tools and allow_unlisted_tools only."
        )


def _main_model(config: RailsConfig) -> Optional[Model]:
    """*config*'s `main` model, or None without one."""
    return next((model for model in config.models if model.type == "main"), None)


def _tool_call_rails_use_model(config: RailsConfig, model_type: str) -> bool:
    """Whether a global or per-tool tool-call rail names *model_type* with `$model=`."""
    tool_call = config.rails.tool_call
    flows = [*tool_call.flows, *(flow for tool_flows in tool_call.per_tool.values() for flow in tool_flows)]
    return any(_get_flow_model(flow) == model_type for flow in flows)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the metrics exporter and the engine before the server accepts requests, and stop them on shutdown."""
    engine: Guardrails = app.state.engine
    # Under the CLI this reuses the CLI's exporter, which the CLI stops; served any other way, the app owns it.
    with running_metrics_exporter(metrics_exporter_settings()) as exporter:
        if exporter is not None:
            _warn_if_the_config_records_no_metrics(engine.config)
        await engine.startup()
        try:
            yield
        finally:
            await engine.shutdown()


def _warn_if_the_config_records_no_metrics(config: RailsConfig) -> None:
    """Warn that the scrape endpoint will show no guardrails metrics, when the config doesn't turn them on."""
    if not config.metrics.enabled:
        log.warning(
            "The metrics exporter is running, but the config doesn't set `metrics.enabled: true`, so the engine "
            "records no metrics, and the scrape endpoint shows none."
        )


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
    outcome = await _check_tool_call(request.app.state.engine, request.app.state.tool_policy, adapter, call)
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


async def _check_tool_call(
    engine: Guardrails, policy: ToolPolicy, adapter: HookAdapter, call: ToolInvocation
) -> HookOutcome:
    """Decide the hook's call with the tool policy, then run its tool's per-tool rails on it, and log the verdict."""
    if not isinstance(call.arguments, dict):
        outcome = HookOutcome.deny(f"tool call '{quoted_identity(call.name)}' has arguments that aren't a JSON object")
        _log_check(adapter, call, outcome, rail=None)
        return outcome
    reason = policy.decide(call.name, call.arguments)
    if reason is not None:
        outcome = HookOutcome.deny(reason)
        _log_check(adapter, call, outcome, rail=_TOOL_POLICY)
        return outcome
    # Only a tool with per-tool rails needs the check: it would run nothing else, and would take a queue slot.
    if not engine.config.rails.tool_call.per_tool.get(call.name):
        outcome = HookOutcome.allow()
        _log_check(adapter, call, outcome, rail=None)
        return outcome
    try:
        result = await engine.check_async(
            check_messages(call), rail_types=[RailType.TOOL_CALL], tools=check_tools(call, policy)
        )
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
