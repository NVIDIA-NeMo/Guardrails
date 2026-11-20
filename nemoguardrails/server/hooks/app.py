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
from collections import OrderedDict
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Mapping, Optional, Union

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
from nemoguardrails.server.hooks import decision_record
from nemoguardrails.server.hooks.adapters import ADAPTERS, HookAdapter, HookAgent, UnsupportedHookEventError
from nemoguardrails.server.hooks.agent_start import AgentStartPolicy
from nemoguardrails.server.hooks.cedar_policy import (
    CedarAgentStartRules,
    CedarPolicyError,
    CedarPolicySet,
    CedarToolRules,
)
from nemoguardrails.server.hooks.checking import (
    HookOutcome,
    ToolInvocation,
    check_messages,
    check_tools,
    outcome_from_result,
)
from nemoguardrails.server.hooks.decision_record import DecisionRecord, DecisionRecordError
from nemoguardrails.server.hooks.dogwood_engine import DogwoodToolRules, Event, tool_action
from nemoguardrails.server.hooks.quarantine import QuarantineEvent, QuarantineStore, QuarantineStoreError
from nemoguardrails.server.hooks.schemas import (
    AgentStartRequest,
    ClaudeCodeToolResultRequest,
    HookBody,
    HookErrorCode,
    ModelCallRequest,
    QuarantineReleaseRequest,
    QuarantineRequest,
)
from nemoguardrails.server.hooks.tool_policy import ToolPolicy
from nemoguardrails.server.metrics import MetricsExporterSettings, running_metrics_exporter
from nemoguardrails.telemetry import DeploymentTypeEnum, set_deployment_type

log = logging.getLogger(__name__)

# What `llm_call_exception_handler` answers on the main server. A rail's model call that fails with an upstream
# HTTP status propagates out of `check_async()` rather than blocking, but here it is a failed rail all the same.
_MODEL_CALL_ERRORS = (LLMCallException, ModelEngineError, HTTPClientError)

# What the log line names as deciding a call the tool policy denies, in place of a rail.
_TOOL_POLICY = "tool policy"
# The same, when the config's Cedar policies decide the call.
_CEDAR_POLICY = "cedar policy"
# The same, when the config's Dogwood policies decide the call.
_DOGWOOD_POLICY = "dogwood policy"

# The one event `/v1/hooks/agent-start` handles.
_AGENT_START_EVENT = "AgentStart"
# The events a Claude Code hook sends after a tool ran, which a Dogwood rule reads as `::response` and `::error`.
_TOOL_RESULT_EVENTS = {"PostToolUse": "response", "PostToolUseFailure": "error"}
# The one event `/v1/hooks/model-call` handles.
_MODEL_CALL_EVENT = "ModelCall"
# What the decision record names as deciding an agent start or a model call.
_AGENT_START_RULES = "hooks_server.agent_start"
_QUARANTINE = "quarantine"
# Who the record and the run stops name as giving the decision.
_BY = "nemoguardrails hooks-server"
# The rule or rail that decided the tool call being checked, as its log line names it, for the decision record.
_decided_by: ContextVar[Optional[str]] = ContextVar("_decided_by", default=None)

# The headers that name the agent a tool call comes from, so a quarantine of it can deny the call. Whatever runs the
# agent sets them, such as the Relay plugin's `headers` or a hook's command; a call without them can't be matched.
AGENT_NAME_HEADER = "x-nemo-agent-name"
AGENT_DIGEST_HEADER = "x-nemo-agent-digest"
# The header that names the run a tool call belongs to, so the owner of the run can ask whether a rule stopped it.
RUN_ID_HEADER = "x-nemo-run-id"
# The most runs whose stop the server keeps; the oldest is forgotten first.
_MAX_RUN_STOPS = 1024

# A port and service name of its own, so it exports beside the main server on one host, and loopback, like its API.
_METRICS_DEFAULTS = MetricsExporterSettings(host="127.0.0.1", port=9465, service_name="nemoguardrails-hooks-server")


def metrics_exporter_settings(environ: Optional[Mapping[str, str]] = None) -> MetricsExporterSettings:
    """The `NEMO_GUARDRAILS_SERVER_METRICS_*` exporter settings, with the hooks server's defaults for those unset."""
    return MetricsExporterSettings.from_env(environ, defaults=_METRICS_DEFAULTS)


class HooksConfigError(ConfigurationError):
    """A config the hooks server can't serve, found before it answers any request."""


def check_hooks_config(config: RailsConfig) -> None:
    """Raise `HooksConfigError` unless IORails can serve *config*, it can deny a call, its rails and `main` run, and
    an allowlist says whether unlisted tools are allowed."""
    reason = IORails.unsupported_reason(config)
    if reason is not None:
        raise HooksConfigError(f"The hooks server runs on IORails, which can't serve this config: {reason}")
    _check_no_global_tool_call_rails(config)
    _check_stop_rails_are_configured(config)
    _check_quarantine_has_a_store(config)
    _check_agent_start_can_deny(config)
    if config.hooks_server.agent_start is None or _checks_tool_calls(config):
        # A config for agent starts alone checks no tool calls; its tool-call paths answer an error, never an allow.
        _check_something_can_deny(config)
    _check_per_tool_rails_can_run(config, ToolPolicy.from_config(config.hooks_server))
    _check_main_model(config)
    _check_allowlist_sets_unlisted_tools_choice(config)


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
    if config.hooks_server.can_deny or _has_per_tool_rails(config) or config.hooks_server.quarantine_store:
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


def _check_stop_rails_are_configured(config: RailsConfig) -> None:
    """Raise when hooks_server.stop_rails or quarantine_rails names a rail no per-tool flow runs, since it would never
    select its outcome."""
    rails = {flow.split(" $", 1)[0] for flows in config.rails.tool_call.per_tool.values() for flow in flows}
    for list_name in ("stop_rails", "quarantine_rails"):
        for rail in getattr(config.hooks_server, list_name):
            if rail not in rails:
                raise HooksConfigError(
                    f"hooks_server.{list_name} names rail '{rail}', which no rails.tool_call.per_tool flow runs, so it "
                    "would never apply; name a per-tool rail without its $ arguments, such as 'regex check tool call'."
                )


def _check_quarantine_has_a_store(config: RailsConfig) -> None:
    """Raise when a rule can quarantine an agent but no store keeps the quarantine."""
    hooks_server = config.hooks_server
    if (hooks_server.quarantine_tools or hooks_server.quarantine_rails) and not hooks_server.quarantine_store:
        raise HooksConfigError(
            "hooks_server.quarantine_tools or quarantine_rails can quarantine an agent, but no "
            "hooks_server.quarantine_store keeps the quarantine; set quarantine_store to a file."
        )


def _check_agent_start_can_deny(config: RailsConfig) -> None:
    """Raise when hooks_server.agent_start is set but sets no rule, since then every agent would start."""
    agent_start = config.hooks_server.agent_start
    if agent_start is not None and not agent_start.can_deny:
        raise HooksConfigError(
            "hooks_server.agent_start sets no rule that can deny a start, so every agent would be allowed to start; "
            "set allowed_sources, allowed_models, allow_local_sources: false, require_digests: true, "
            "require_policy_enforcers: true or cedar, or remove the section."
        )


def _checks_tool_calls(config: RailsConfig) -> bool:
    """Whether *config*'s tool policy or a per-tool rail can deny a tool call."""
    # The decision record keeps decisions; it decides none, so it doesn't make a config check tool calls.
    tool_lists_set = bool(config.hooks_server.model_fields_set - {"agent_start", "decision_record"})
    return config.hooks_server.can_deny or _has_per_tool_rails(config) or tool_lists_set


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
    cedar_tool_rules, agent_start_policy = _load_rules(config)
    dogwood_tool_rules = _load_dogwood(config)
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
    app.state.cedar_tool_rules = cedar_tool_rules
    app.state.dogwood_tool_rules = dogwood_tool_rules
    app.state.checks_tool_calls = config.hooks_server.agent_start is None or _checks_tool_calls(config)
    app.state.agent_start_policy = agent_start_policy
    app.state.quarantine = _quarantine_store(config)
    app.state.run_stops = OrderedDict()
    app.state.decision_record = _decision_record(config)
    for exc_type, handler in _EXCEPTION_HANDLERS:
        # Handlers are typed with their specific exception; Starlette's stub expects
        # (Request, Exception), so ty flags the narrower signature as a false positive.
        app.add_exception_handler(exc_type, handler)  # ty: ignore[invalid-argument-type]
    app.add_middleware(ExceptionMiddleware, handlers={Exception: internal_error_handler})
    app.include_router(_router)
    return app


AgentStartRules = Union[AgentStartPolicy, CedarAgentStartRules]


def _load_rules(config: RailsConfig) -> tuple[Optional[CedarToolRules], Optional[AgentStartRules]]:
    """The tool-call Cedar rules and the agent-start rules the config sets, read once; a Cedar policy set that can't be
    read or parsed, or that needs `cedarpy` where it isn't installed, stops the server from starting."""
    hooks_server = config.hooks_server
    base = _config_folder(config)
    try:
        cedar_tool_rules = None
        if hooks_server.cedar is not None:
            policies = _load_cedar(hooks_server.cedar, base, "hooks_server", outcomes_allowed=True)
            if policies.quarantines and not hooks_server.quarantine_store:
                raise HooksConfigError(
                    'hooks_server.cedar has a forbid with @outcome("quarantine"), but no hooks_server.quarantine_store '
                    "keeps the quarantine; set quarantine_store to a file."
                )
            cedar_tool_rules = CedarToolRules(policies)
        agent_start = hooks_server.agent_start
        agent_start_policy: Optional[AgentStartRules] = None
        if agent_start is not None and agent_start.cedar is not None:
            policies = _load_cedar(agent_start.cedar, base, "hooks_server.agent_start", outcomes_allowed=False)
            agent_start_policy = CedarAgentStartRules(policies)
        elif agent_start is not None:
            agent_start_policy = AgentStartPolicy.from_config(agent_start)
    except CedarPolicyError as exc:
        raise HooksConfigError(str(exc)) from None
    return cedar_tool_rules, agent_start_policy


def _load_dogwood(config: RailsConfig) -> Optional[DogwoodToolRules]:
    """The Dogwood tool-call rules the config sets, read once; a policy outside the supported subset stops the server
    from starting."""
    rules = config.hooks_server.dogwood
    if rules is None:
        return None
    try:
        base = _config_folder(config)
        loaded = DogwoodToolRules.load(
            base / rules.policies,
            base / rules.schema_file,
            rules.gateway,
            rules.denied_calls,
            providers=None if rules.providers is None else base / rules.providers,
            macros=None if rules.macros is None else base / rules.macros,
        )
    except CedarPolicyError as exc:
        raise HooksConfigError(str(exc)) from None
    if loaded.quarantines and not config.hooks_server.quarantine_store:
        raise HooksConfigError(
            'hooks_server.dogwood has a forbid with @outcome("quarantine"), but no hooks_server.quarantine_store keeps '
            "the quarantine; set quarantine_store to a file."
        )
    return loaded


def _config_folder(config: RailsConfig) -> Path:
    """The folder a config's relative Cedar paths start from: the config's own, else the directory the server runs
    in."""
    if not config.config_path or "," in config.config_path:
        return Path.cwd()
    path = Path(config.config_path)
    return path if path.is_dir() else path.parent


def _load_cedar(rules, base: Path, section: str, *, outcomes_allowed: bool) -> CedarPolicySet:
    """The policy set one `cedar` section names."""
    schema = None if rules.schema_file is None else base / rules.schema_file
    return CedarPolicySet.load(base / rules.policies, schema, section=section, outcomes_allowed=outcomes_allowed)


def _quarantine_store(config: RailsConfig) -> Optional[QuarantineStore]:
    """The store hooks_server.quarantine_store names, read once to refuse a store that can't be read."""
    path = config.hooks_server.quarantine_store
    if path is None:
        return None
    try:
        return QuarantineStore(Path(path))
    except QuarantineStoreError as exc:
        raise HooksConfigError(str(exc)) from exc


def _decision_record(config: RailsConfig) -> Optional[DecisionRecord]:
    """The record hooks_server.decision_record names, checked once to refuse a record whose chain is broken."""
    path = config.hooks_server.decision_record
    if path is None:
        return None
    try:
        return DecisionRecord(Path(path))
    except DecisionRecordError as exc:
        raise HooksConfigError(str(exc)) from exc


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


def _check_allowlist_sets_unlisted_tools_choice(config: RailsConfig) -> None:
    """Raise when hooks_server.allowed_tools lists tools but leaves allow_unlisted_tools to its default."""
    # The default, true, suits a blocklist or per-tool rails alone, but beside an allowlist it silently allows the
    # tools the allowlist leaves out, so an allowlist must choose.
    hooks_server = config.hooks_server
    if hooks_server.allowed_tools and "allow_unlisted_tools" not in hooks_server.model_fields_set:
        raise HooksConfigError(
            "hooks_server.allowed_tools lists tools while hooks_server.allow_unlisted_tools is unset, and its "
            "default, true, allows every tool allowed_tools doesn't list; set allow_unlisted_tools: false to allow "
            "only the listed tools, or true to allow the rest as well."
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
    """Answer a Claude Code hook in Claude Code's hook output format: a tool call, or with Dogwood rules a tool's
    result, which is recorded."""
    body = await _read_json(request)
    if (
        isinstance(body, dict)
        and body.get("hook_event_name") in _TOOL_RESULT_EVENTS
        and request.app.state.dogwood_tool_rules is not None
    ):
        return _record_tool_result(request, body)
    return await _answer_hook("claude-code", request)


def _record_tool_result(request: Request, body: dict) -> Response:
    """Record that a tool call completed (`PostToolUse`) or failed (`PostToolUseFailure`), for the Dogwood rules that
    read `::response` and `::error` events. Nothing is decided; Claude Code is answered with no decision."""
    try:
        result = ClaudeCodeToolResultRequest.model_validate(body)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in errors]) from None
    rules: DogwoodToolRules = request.app.state.dogwood_tool_rules
    arguments = result.tool_input if isinstance(result.tool_input, dict) else {}
    record: DecisionRecord = request.app.state.decision_record
    try:
        record.append(
            {
                "kind": "tool_result",
                "agent_name": request.headers.get(AGENT_NAME_HEADER),
                "agent_digest": request.headers.get(AGENT_DIGEST_HEADER),
                "run_id": request.headers.get(RUN_ID_HEADER),
                # As for a tool call: the names of its arguments, and only the values a Dogwood policy reads.
                "asked": {
                    "hook": "claude-code",
                    "event": result.hook_event_name,
                    "tool_name": result.tool_name,
                    "argument_names": sorted(arguments),
                    "call_id": result.tool_use_id,
                    "inputs": rules.kept_inputs(result.tool_name, arguments),
                },
                "outcome": _TOOL_RESULT_EVENTS[result.hook_event_name],
                "reason": None,
                "rule": None,
                "by": _BY,
            }
        )
    except DecisionRecordError as exc:
        log.error("%s", exc)
        return _error_response(500, str(exc), code=HookErrorCode.DECISION_RECORD_FAILED)
    return JSONResponse({})


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


@_router.post(
    "/v1/hooks/agent-start",
    name="handleAgentStartHook",
    summary="Check whether an agent may start",
    tags=["Hooks"],
)
async def handle_agent_start_hook(request: Request) -> Response:
    """Answer whether the agent the facts describe may start, with the same decision body as NeMo Relay's path."""
    policy: Optional[AgentStartRules] = request.app.state.agent_start_policy
    quarantine: Optional[QuarantineStore] = request.app.state.quarantine
    if policy is None and quarantine is None:
        return _error_response(
            404,
            "this hooks server's config has no hooks_server.agent_start section, so it checks no agent starts",
            code=HookErrorCode.CHECK_NOT_CONFIGURED,
        )
    start = _parse_agent_start(await _read_json(request))
    try:
        outcome, rule = _decide_start(policy, quarantine, start)
    except QuarantineStoreError as exc:
        return _error_response(500, str(exc), code=HookErrorCode.QUARANTINE_STORE_FAILED)
    _log_agent_start(start, outcome)
    failed = _record_decision(
        request,
        kind="agent_start",
        agent=(start.agent_name, start.agent_digest),
        asked=start.model_dump(exclude={"hook_event_name"}, exclude_none=True),
        outcome=outcome,
        rule=rule,
    )
    if failed is not None:
        return failed
    if outcome.kind == "allow":
        return JSONResponse({"decision": "allow"})
    return JSONResponse({"decision": "deny", "reason": outcome.reason})


def _decide_start(
    policy: Optional["AgentStartRules"], quarantine: Optional[QuarantineStore], start: AgentStartRequest
) -> tuple[HookOutcome, Optional[str]]:
    """The answer to an agent start, and the rule that gave it: a quarantine first, then the agent-start rules."""
    held = None if quarantine is None else quarantine.find(start.agent_name, start.agent_digest)
    if held is not None:
        return HookOutcome.deny(_quarantined_reason(held)), _QUARANTINE
    rule = _CEDAR_POLICY if isinstance(policy, CedarAgentStartRules) else _AGENT_START_RULES
    reason = None if policy is None else policy.decide(start)
    return (HookOutcome.allow() if reason is None else HookOutcome.deny(reason)), (None if policy is None else rule)


@_router.post(
    "/v1/hooks/model-call",
    name="handleModelCallHook",
    summary="Check whether an agent may call a model, during its run",
    tags=["Hooks"],
)
async def handle_model_call_hook(request: Request) -> Response:
    """Answer whether the agent may call the model, by the same rules that decide the models it may start with.

    The call is decided as an agent start that names only this model: a quarantine of the agent, then
    `allowed_models` or the agent-start Cedar policies on its `Model`. So a model outside the list is refused during
    the run, not only at start."""
    policy: Optional[AgentStartRules] = request.app.state.agent_start_policy
    quarantine: Optional[QuarantineStore] = request.app.state.quarantine
    if policy is None and quarantine is None:
        return _error_response(
            404,
            "this hooks server's config has no hooks_server.agent_start section, so it checks no model calls",
            code=HookErrorCode.CHECK_NOT_CONFIGURED,
        )
    body = await _read_json(request)
    try:
        event = HookBody.model_validate(body).hook_event_name
        if event != _MODEL_CALL_EVENT:
            raise UnsupportedHookEventError("model-call", event, _MODEL_CALL_EVENT)
        call = ModelCallRequest.model_validate(body)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in errors]) from None
    start = AgentStartRequest(
        hook_event_name=_AGENT_START_EVENT,
        agent_name=call.agent_name,
        agent_digest=call.agent_digest,
        parts=[],
        models=[call.model],
    )
    try:
        outcome, rule = _decide_start(policy, quarantine, start)
    except QuarantineStoreError as exc:
        return _error_response(500, str(exc), code=HookErrorCode.QUARANTINE_STORE_FAILED)
    log.info(
        "Hook check on /v1/hooks/model-call: event=%s agent_name=%r model=%r outcome=%s reason=%r",
        _MODEL_CALL_EVENT,
        quoted_identity(call.agent_name),
        quoted_identity(call.model),
        outcome.kind,
        outcome.reason,
    )
    failed = _record_decision(
        request,
        kind="model_call",
        agent=(call.agent_name, call.agent_digest),
        asked=call.model_dump(exclude={"hook_event_name", "agent_name", "agent_digest"}, exclude_none=True),
        outcome=outcome,
        rule=rule,
    )
    if failed is not None:
        return failed
    if outcome.kind == "allow":
        return JSONResponse({"decision": "allow", "rule": rule})
    return JSONResponse({"decision": "deny", "reason": outcome.reason, "rule": rule})


@_router.get("/v1/decisions", name="listDecisions", summary="The decision record", tags=["Decisions"])
async def list_decisions(request: Request) -> Response:
    """Every recorded decision, oldest first, filtered by `kind`, `outcome`, `run_id` and `agent_name`."""
    record: Optional[DecisionRecord] = request.app.state.decision_record
    if record is None:
        return _no_decision_record()
    query = request.query_params
    try:
        entries = decision_record.read(record.path)
    except DecisionRecordError as exc:
        return _error_response(500, str(exc), code=HookErrorCode.DECISION_RECORD_FAILED)
    selected = decision_record.select(
        entries,
        kind=query.get("kind"),
        outcome=query.get("outcome"),
        run_id=query.get("run_id"),
        agent_name=query.get("agent_name"),
    )
    return JSONResponse({"decisions": selected})


@_router.get(
    "/v1/decisions/check",
    name="checkDecisions",
    summary="Whether the decision record's chain holds",
    tags=["Decisions"],
)
async def check_decisions(request: Request) -> Response:
    """Whether every entry's hash is its own and chains to the one before; with `head`, that the last one is it."""
    record: Optional[DecisionRecord] = request.app.state.decision_record
    if record is None:
        return _no_decision_record()
    return JSONResponse(decision_record.check(record.path, head=request.query_params.get("head")))


def _no_decision_record() -> Response:
    return _error_response(
        404,
        "this hooks server's config sets no hooks_server.decision_record, so it keeps no record",
        code=HookErrorCode.CHECK_NOT_CONFIGURED,
    )


def _record_decision(
    request: Request,
    *,
    kind: str,
    agent: tuple[Optional[str], Optional[str]],
    asked: dict,
    outcome: HookOutcome,
    rule: Optional[str],
) -> Optional[Response]:
    """Add the decision to the record, when the config keeps one; an error response when it can't be written, so no
    decision is given that the record doesn't hold."""
    record: Optional[DecisionRecord] = request.app.state.decision_record
    if record is None:
        return None
    try:
        record.append(
            {
                "kind": kind,
                "agent_name": agent[0],
                "agent_digest": agent[1],
                "run_id": request.headers.get(RUN_ID_HEADER),
                "asked": asked,
                "outcome": outcome.kind,
                "reason": outcome.reason if outcome.kind != "error" else outcome.message,
                "rule": rule,
                "by": _BY,
            }
        )
    except DecisionRecordError as exc:
        log.error("%s", exc)
        return _error_response(500, str(exc), code=HookErrorCode.DECISION_RECORD_FAILED)
    return None


@_router.get("/v1/runs/{run_id}", name="getRunStop", summary="Whether a rule stopped a run", tags=["Runs"])
async def get_run_stop(run_id: str, request: Request) -> Response:
    """The stop or quarantine a rule chose for a tool call of the run, or `null`. The owner of the run, which started
    the harness, reads it and ends the run with the agents it called. Kept in memory: a restart forgets it, while a
    quarantine stays in its store."""
    return JSONResponse({"run_id": run_id, "stop": request.app.state.run_stops.get(run_id)})


_NO_QUARANTINE_STORE = "this hooks server's config sets no hooks_server.quarantine_store, so it quarantines nothing"


@_router.post("/v1/quarantines", name="quarantineAgent", summary="Quarantine an agent", tags=["Quarantine"])
async def quarantine_agent(request: Request) -> Response:
    """Quarantine an agent at a person's or the platform's request, with no violation. Its runs still alive are
    denied every tool call that names it, and it can't start until it is released."""
    store: Optional[QuarantineStore] = request.app.state.quarantine
    if store is None:
        return _error_response(404, _NO_QUARANTINE_STORE, code=HookErrorCode.CHECK_NOT_CONFIGURED)
    body = _parse_model(QuarantineRequest, await _read_json(request))
    try:
        event = store.quarantine(body.agent_name, body.agent_digest, body.reason, body.requested_by, source="request")
    except QuarantineStoreError as exc:
        return _error_response(500, str(exc), code=HookErrorCode.QUARANTINE_STORE_FAILED)
    log.info("Quarantine: agent_name=%r agent_digest=%r by=%r", event.agent_name, event.agent_digest, event.by)
    return JSONResponse({"quarantine": asdict(event)})


@_router.get("/v1/quarantines", name="listQuarantines", summary="List quarantined agents", tags=["Quarantine"])
async def list_quarantines(request: Request) -> Response:
    """Every quarantine in force. Released ones stay in the store's file, which is the history."""
    store: Optional[QuarantineStore] = request.app.state.quarantine
    if store is None:
        return _error_response(404, _NO_QUARANTINE_STORE, code=HookErrorCode.CHECK_NOT_CONFIGURED)
    try:
        return JSONResponse({"quarantines": [asdict(event) for event in store.active()]})
    except QuarantineStoreError as exc:
        return _error_response(500, str(exc), code=HookErrorCode.QUARANTINE_STORE_FAILED)


@_router.post(
    "/v1/quarantines/release", name="releaseAgent", summary="Release a quarantined agent", tags=["Quarantine"]
)
async def release_agent(request: Request) -> Response:
    """Release a quarantined agent, recording who released it and why. Nothing else releases one."""
    store: Optional[QuarantineStore] = request.app.state.quarantine
    if store is None:
        return _error_response(404, _NO_QUARANTINE_STORE, code=HookErrorCode.CHECK_NOT_CONFIGURED)
    body = _parse_model(QuarantineReleaseRequest, await _read_json(request))
    try:
        event = store.release(body.agent_name, body.agent_digest, body.reason, body.released_by)
    except QuarantineStoreError as exc:
        return _error_response(500, str(exc), code=HookErrorCode.QUARANTINE_STORE_FAILED)
    if event is None:
        return _error_response(
            404,
            f"agent '{quoted_identity(body.agent_name)}' with that digest is not quarantined",
            code=HookErrorCode.NOT_QUARANTINED,
        )
    log.info("Quarantine released: agent_name=%r agent_digest=%r by=%r", event.agent_name, event.agent_digest, event.by)
    return JSONResponse({"released": asdict(event)})


def _parse_model(model, body: object):
    """*body* as *model*, raising a failure for the shared `422` handler."""
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in errors]) from None


@_router.get("/v1/health", summary="Liveness health check.", tags=["Health"])
@_router.get("/healthz", summary="Liveness health check.", tags=["Health"])
async def health() -> Response:
    """Return HTTP 200 while the server process is running and able to serve requests."""
    return JSONResponse(content={"status": "pass"}, media_type="application/health+json")


async def _answer_hook(agent: HookAgent, request: Request) -> Response:
    """Check the tool call the hook carries, and answer with the agent's allow or deny, or an error status."""
    adapter = ADAPTERS[agent]
    if not request.app.state.checks_tool_calls:
        return _error_response(
            404,
            "this hooks server's config sets only hooks_server.agent_start, so it checks no tool calls",
            code=HookErrorCode.CHECK_NOT_CONFIGURED,
        )
    call = _parse_hook(adapter, await _read_json(request))
    identity = (request.headers.get(AGENT_NAME_HEADER), request.headers.get(AGENT_DIGEST_HEADER))
    _decided_by.set(None)
    dogwood: Optional[DogwoodToolRules] = request.app.state.dogwood_tool_rules
    try:
        outcome = await _check_tool_call(
            request.app.state.engine,
            request.app.state.tool_policy,
            adapter,
            call,
            request.app.state.quarantine,
            identity,
            request.app.state.cedar_tool_rules,
            dogwood=None if dogwood is None else (dogwood, *_run_history(request, identity)),
        )
    except QuarantineStoreError as exc:
        return _error_response(500, str(exc), code=HookErrorCode.QUARANTINE_STORE_FAILED)
    run_id = request.headers.get(RUN_ID_HEADER)
    if run_id and outcome.kind in ("stop", "quarantine"):
        _record_run_stop(request.app.state.run_stops, run_id, outcome, call, identity)
    arguments = sorted(call.arguments) if isinstance(call.arguments, dict) else []
    failed = _record_decision(
        request,
        kind="tool_call",
        agent=identity,
        # The tool's input stays out of the record, as it stays out of the log: only the names of its arguments.
        asked={
            "hook": agent,
            "tool_name": call.name,
            "argument_names": arguments,
            "call_id": call.call_id,
            # A Dogwood policy that reads an input of earlier calls, in a sum or a match, needs its value: only those.
            **(
                {"inputs": dogwood.kept_inputs(call.name, call.arguments)}
                if dogwood is not None and isinstance(call.arguments, dict)
                else {}
            ),
        },
        outcome=outcome,
        # The rule or rail that decided; none for a call nothing denied.
        rule=_decided_by.get(),
    )
    if failed is not None:
        return failed
    if outcome.kind == "error":
        return _error_response(500, outcome.message or "", code=outcome.code)
    return JSONResponse(adapter.response_body(outcome))


def _run_history(request: Request, identity: tuple[Optional[str], Optional[str]]) -> tuple[Optional[list[Event]], str]:
    """The events of the run the request belongs to, from the decision record, and why there are none when they can't
    be had. A run is its `x-nemo-run-id` under one agent name and digest."""
    run_id = request.headers.get(RUN_ID_HEADER)
    if not run_id:
        return None, f"the request has no {RUN_ID_HEADER} header, so it belongs to no run"
    record: DecisionRecord = request.app.state.decision_record
    try:
        entries = decision_record.read(record.path)
    except DecisionRecordError as exc:
        log.error("%s", exc)
        return None, "the decision record, which holds the run's history, can't be read"
    events: list[Event] = []
    for entry in entries:
        if (entry.get("run_id"), entry.get("agent_name"), entry.get("agent_digest")) != (run_id, *identity):
            continue
        asked = entry.get("asked") or {}
        if entry.get("kind") == "tool_call":
            # A denied call is an attempt, a `::request`, as the Dogwood language keeps it, unless the config asks for
            # AgentCore's reading, an `::error`.
            kind = "request" if entry.get("outcome") == "allow" else request.app.state.dogwood_tool_rules.denied_calls
        elif entry.get("kind") == "tool_result":
            kind = entry.get("outcome")
        else:
            continue
        try:
            at = datetime.fromisoformat(entry["at"])
        except (KeyError, TypeError, ValueError):
            return None, "the decision record has an entry without a time"
        if isinstance(asked.get("tool_name"), str):
            events.append(Event(str(kind), tool_action(asked["tool_name"]), at, asked.get("inputs") or {}))
    return events, ""


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


def _parse_agent_start(body: object) -> AgentStartRequest:
    """The facts an agent-start body carries, raising a failure for the shared `422` handler."""
    try:
        event = HookBody.model_validate(body).hook_event_name
        if event != _AGENT_START_EVENT:
            raise UnsupportedHookEventError("agent-start", event, _AGENT_START_EVENT)
        return AgentStartRequest.model_validate(body)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in errors]) from None


def _log_agent_start(start: AgentStartRequest, outcome: HookOutcome) -> None:
    """Log one line for an agent-start check: the agent, how many facts it carried, and the verdict."""
    log.info(
        "Hook check on /v1/hooks/agent-start: event=%s agent_name=%r parts=%d models=%d policies=%d outcome=%s "
        "reason=%r",
        _AGENT_START_EVENT,
        quoted_identity(start.agent_name),
        len(start.parts),
        len(start.models),
        len(start.policies),
        outcome.kind,
        outcome.reason,
    )


async def _check_tool_call(
    engine: Guardrails,
    policy: ToolPolicy,
    adapter: HookAdapter,
    call: ToolInvocation,
    quarantine: Optional[QuarantineStore] = None,
    identity: tuple[Optional[str], Optional[str]] = (None, None),
    cedar: Optional[CedarToolRules] = None,
    dogwood: Optional[tuple[DogwoodToolRules, Optional[list[Event]], str]] = None,
) -> HookOutcome:
    """Decide the hook's call with the tool policy, then run its tool's per-tool rails on it, and log the verdict.

    A call from a quarantined agent is answered with the quarantine before anything else. A rule that quarantines
    stores the quarantine under the agent the headers name; without them the run is still ended, and nothing is
    stored."""
    agent_name, agent_digest = identity
    if quarantine is not None and agent_name:
        held = quarantine.find(agent_name, agent_digest)
        if held is not None:
            outcome = HookOutcome.quarantine(_quarantined_reason(held))
            _log_check(adapter, call, outcome, rail="quarantine")
            return outcome
    if not isinstance(call.arguments, dict):
        outcome = HookOutcome.deny(f"tool call '{quoted_identity(call.name)}' has arguments that aren't a JSON object")
        _log_check(adapter, call, outcome, rail=None)
        return outcome
    if cedar is not None or dogwood is not None:
        # Cedar or Dogwood alone decides the call; a forbid's @outcome selects a stop or a quarantine.
        if dogwood is not None:
            rules, history, unavailable = dogwood
            now = datetime.now(timezone.utc)
            kind, reason = rules.decide(call.name, call.arguments, *identity, history, now, unavailable)
            rail = _DOGWOOD_POLICY
        else:
            assert cedar is not None
            kind, reason = cedar.decide(call.name, call.arguments, *identity)
            rail = _CEDAR_POLICY
        if kind != "allow":
            reason = reason or kind
            if kind == "quarantine":
                outcome = _quarantine_by_rule(quarantine, identity, reason)
            else:
                outcome = HookOutcome.stop(reason) if kind == "stop" else HookOutcome.deny(reason)
            _log_check(adapter, call, outcome, rail=rail)
            return outcome
        return await _run_per_tool_rails(engine, policy, adapter, call, quarantine, identity)
    # A quarantine, then a stop, are checked first: a call that ends the run is denied too, whatever else the policy
    # says about it.
    reason = policy.quarantines(call.name, call.arguments)
    if reason is not None:
        outcome = _quarantine_by_rule(quarantine, identity, reason)
        _log_check(adapter, call, outcome, rail=_TOOL_POLICY)
        return outcome
    reason = policy.stops(call.name, call.arguments)
    if reason is not None:
        outcome = HookOutcome.stop(reason)
        _log_check(adapter, call, outcome, rail=_TOOL_POLICY)
        return outcome
    reason = policy.decide(call.name, call.arguments)
    if reason is not None:
        outcome = HookOutcome.deny(reason)
        _log_check(adapter, call, outcome, rail=_TOOL_POLICY)
        return outcome
    return await _run_per_tool_rails(engine, policy, adapter, call, quarantine, identity)


async def _run_per_tool_rails(
    engine: Guardrails,
    policy: ToolPolicy,
    adapter: HookAdapter,
    call: ToolInvocation,
    quarantine: Optional[QuarantineStore],
    identity: tuple[Optional[str], Optional[str]],
) -> HookOutcome:
    """Run the per-tool rails of a call the rules allowed, and log the verdict."""
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
    if outcome.kind == "deny" and result.rail in engine.config.hooks_server.quarantine_rails:
        outcome = _quarantine_by_rule(quarantine, identity, outcome.reason or result.rail)
    elif outcome.kind == "deny" and result.rail in engine.config.hooks_server.stop_rails:
        outcome = HookOutcome.stop(outcome.reason or result.rail)
    _log_check(adapter, call, outcome, rail=result.rail)
    return outcome


def _record_run_stop(
    run_stops: "OrderedDict[str, dict]",
    run_id: str,
    outcome: HookOutcome,
    call: ToolInvocation,
    identity: tuple[Optional[str], Optional[str]],
) -> None:
    """Keep the first stop or quarantine of a run, for the owner of the run to read at `/v1/runs/{run_id}`."""
    if run_id in run_stops:
        return
    run_stops[run_id] = {
        "outcome": outcome.kind,
        "reason": outcome.reason,
        "tool_name": quoted_identity(call.name),
        "agent_name": identity[0],
        "agent_digest": identity[1],
        "by": _BY,
        "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    while len(run_stops) > _MAX_RUN_STOPS:
        run_stops.popitem(last=False)


def _quarantine_by_rule(
    quarantine: Optional[QuarantineStore], identity: tuple[Optional[str], Optional[str]], reason: str
) -> HookOutcome:
    """The quarantine a rule selected, stored under the agent the call's headers name, when they name one."""
    agent_name, agent_digest = identity
    if quarantine is not None and agent_name and agent_digest:
        quarantine.quarantine(
            agent_name, agent_digest, reason, by="nemoguardrails hooks-server", source="rule", rule=reason
        )
    else:
        log.warning(
            "A rule quarantined the agent of a tool call that names no agent in the %s and %s headers: the run is "
            "ended, but no quarantine is stored.",
            AGENT_NAME_HEADER,
            AGENT_DIGEST_HEADER,
        )
    return HookOutcome.quarantine(reason)


def _quarantined_reason(held: QuarantineEvent) -> str:
    """The reason a quarantined agent's start or call is denied: the quarantine's own reason."""
    return f"agent '{quoted_identity(held.agent_name)}' is quarantined: {held.reason}"


def _log_check(adapter: HookAdapter, call: ToolInvocation, outcome: HookOutcome, rail: Optional[str]) -> None:
    """Log one line for a check, without the tool's input; an error, which no rail decided, as a warning."""
    _decided_by.set(rail)
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
