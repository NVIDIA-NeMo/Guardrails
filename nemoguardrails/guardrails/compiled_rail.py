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

"""Manifest-driven rail execution for IORails.

A ``CompiledRail`` is the executable unit behind one configured flow string. It is built
once, at engine construction: it resolves the flow's ``RailSurface`` from the manifest
catalog, imports the library action the surface declares, and freezes a plan for filling
that action's parameters. Thereafter each request is one ``await action(**kwargs)`` and a
translation of the returned ``RailOutcome`` into IORails' ``RailResult``.

This replaces the hand-written ``RailAction`` hierarchy. The rail logic itself lives in
``nemoguardrails/library/``, shared with LLMRails, so there is one implementation of each
rail rather than two.

**No Colang runtime is involved.** Executing a rail needs the manifest, the action module,
and a parameter binder. The Colang *vocabulary* appears in one place — ``messages_to_events``
emits event shapes for actions that consume conversation history — but no dispatcher,
runtime, flow, or event loop is required.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from nemoguardrails.actions.rail_outcome import RailOutcome, require_rail_outcome
from nemoguardrails.guardrails.guardrails_types import LLMMessages, RailResult
from nemoguardrails.guardrails.rail_guard import rail_error_result
from nemoguardrails.guardrails.telemetry import action_span
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.manifests import (
    RailDirection,
    RailSurface,
    default_rail_catalog,
    parse_configured_surface,
    resolve_import_ref,
)
from nemoguardrails.manifests.surface_reference import normalize_configured_surface_name

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from nemoguardrails.logging.explain import LLMCallInfo
    from nemoguardrails.manifests import RailCatalog

log = logging.getLogger(__name__)


class RailCompilationError(Exception):
    """A configured flow cannot be turned into an executable rail.

    Raised only during compilation, never while serving a request. ``IORails`` treats it as
    the reason a config must fall back to LLMRails, so its message is user-facing and
    should name the flow and what is wrong with it.
    """


@dataclass(frozen=True)
class RailDependencies:
    """Runtime collaborators a rail action may declare as parameters.

    Injection is by parameter *name*, matching how the Colang runtimes supply the same
    values to the same actions. An action receives only what its signature declares.
    """

    llms: Mapping[str, Any]
    llm_task_manager: Any
    config: Any
    http_client: Any = None
    model_caches: Optional[Mapping[str, Any]] = None
    tracer: Optional["Tracer"] = None


@dataclass(frozen=True)
class RailExecution:
    """One rail run: its verdict, plus every model call the action made.

    ``llm_calls`` comes from a sink installed around the action, so it holds exactly this
    rail's calls — empty for a rail that reached no model.
    """

    result: RailResult
    llm_calls: tuple["LLMCallInfo", ...] = ()


_USER_MESSAGE_EVENT = "UserMessage"
_BOT_UTTERANCE_EVENT = "StartUtteranceBotAction"
_SYSTEM_MESSAGE_EVENT = "SystemMessage"


def messages_to_events(messages: LLMMessages) -> list[dict[str, Any]]:
    """Convert IORails messages into the event shapes conversation-history actions read.

    ``topic_safety_check_input`` and its kin reach history through
    ``nemoguardrails.llm.filters.to_chat_messages``, which recognises exactly these three
    event types. Emitting them here gives IORails the same history LLMRails passes, without
    changing a shipped library signature.

    The coupling runs both ways and is pinned by tests at both ends: a change to the event
    vocabulary in ``llm/filters.py`` would otherwise silently drop a rail's history rather
    than fail. Turns with no content — an assistant tool-call turn, for instance — are
    skipped, matching what ``to_chat_messages`` emits for them.
    """
    events: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not content:
            continue
        role = message.get("role")
        if role == "user":
            events.append({"type": _USER_MESSAGE_EVENT, "text": content})
        elif role == "assistant":
            events.append({"type": _BOT_UTTERANCE_EVENT, "script": content})
        elif role == "system":
            events.append({"type": _SYSTEM_MESSAGE_EVENT, "content": content})
    return events


def _last_user_content(messages: LLMMessages) -> str:
    """Return the most recent user message's content, or "" when there is none.

    Empty rather than raising: the library actions read ``context.get(...)`` with a default
    and call the model with empty text, and IORails now matches that.
    """
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return message["content"]
    return ""


def _llm_calls_from(sink: list[dict[str, Any]]) -> tuple["LLMCallInfo", ...]:
    """Pull the LLMCallInfo records out of a processing-log sink."""
    return tuple(entry["data"] for entry in sink if entry.get("type") == "llm_call_info")


def _outcome_to_result(outcome: RailOutcome) -> RailResult:
    """Translate an engine-neutral verdict into IORails' rail result.

    Field for field, inventing nothing: ``decision`` gates, ``reason`` passes through
    verbatim (``None`` stays ``None`` — rendering a display string is the caller's job),
    and ``metadata`` becomes the structured verdict the generation log records.
    """
    if outcome.is_transform:
        raise RailCompilationError("transform outcomes are not supported yet; this surface should not have compiled")
    allowed = not outcome.is_blocked
    return RailResult(
        is_safe=allowed,
        reason=outcome.reason,
        return_value={"allowed": allowed, **outcome.metadata},
    )


@dataclass(frozen=True)
class _BoundParameter:
    """One action parameter and the value the manifest says fills it."""

    action_param: str
    value: Any


class CompiledRail:
    """One configured flow, resolved to a library action and ready to run."""

    def __init__(
        self,
        *,
        flow: str,
        surface: RailSurface,
        action: Callable[..., Any],
        bound: tuple[_BoundParameter, ...],
        deps: RailDependencies,
    ) -> None:
        """Store the frozen execution plan. Build through :func:`compile_rail`."""
        self.flow = flow
        self.surface = surface
        self._action = action
        self._bound = bound
        self._deps = deps
        self._accepted = _accepted_parameters(action)

    @property
    def surface_name(self) -> str:
        """The manifest surface name, without any ``$param=`` suffix."""
        return self.surface.name

    async def run(self, messages: LLMMessages, bot_response: Optional[str] = None) -> RailResult:
        """Execute the rail and return just its verdict."""
        return (await self.execute(messages, bot_response)).result

    async def execute(self, messages: LLMMessages, bot_response: Optional[str] = None) -> RailExecution:
        """Execute the rail, returning its verdict and the model calls it made.

        Model calls are collected by installing a fresh ``processing_log_var`` list for the
        duration of the action. Every path that produces an ``LLMCallInfo`` appends to that
        var — a live call through ``track_llm_call``, a cache hit, and jailbreak's NIM call
        — so the sink sees them all, and it sees only this rail's.

        Do not replace this with a read of ``llm_call_info_var`` after the action returns.
        Library actions set that var inside themselves, so once it is given a scope the read
        yields the *caller's* value for every rail. A sink is written during the call and is
        indifferent to the var's lifetime.
        """
        sink: list[dict[str, Any]] = []
        token = processing_log_var.set(sink)
        try:
            with action_span(self._deps.tracer, self.surface_name) as span:
                try:
                    outcome = require_rail_outcome(await self._action(**self._call_kwargs(messages, bot_response)))
                    result = _outcome_to_result(outcome)
                except Exception as exc:
                    result = rail_error_result(span, self.surface_name, exc)
        finally:
            processing_log_var.reset(token)

        return RailExecution(result=result, llm_calls=_llm_calls_from(sink))

    def _call_kwargs(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        """Assemble the action's arguments from its declared parameters and the manifest."""
        kwargs = {
            name: value
            for name, value in self._request_dependencies(messages, bot_response).items()
            if name in self._accepted
        }
        for bound in self._bound:
            kwargs[bound.action_param] = bound.value
        return kwargs

    def _request_dependencies(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        """Every value injectable by parameter name for this request.

        Filtered against the action's signature by the caller, so building an entry an
        action does not declare costs nothing but a dict slot.
        """
        return {
            "llms": self._deps.llms,
            "llm": self._deps.llms.get("main"),
            "llm_task_manager": self._deps.llm_task_manager,
            "config": self._deps.config,
            "http_client": self._deps.http_client,
            "model_caches": self._deps.model_caches,
            "context": {
                "user_message": _last_user_content(messages),
                "bot_message": bot_response or "",
            },
            "events": messages_to_events(messages),
        }


def _accepted_parameters(action: Callable[..., Any]) -> frozenset[str]:
    """Return the parameter names *action* accepts by name.

    ``**kwargs`` is excluded deliberately: a catch-all would otherwise look like a
    parameter called ``kwargs`` and be handed the wrong value.
    """
    parameters = inspect.signature(action).parameters
    return frozenset(
        name
        for name, parameter in parameters.items()
        if parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )


def _resolve_surface(flow: str, direction: RailDirection, catalog: "RailCatalog") -> tuple[RailSurface, dict[str, str]]:
    """Find the manifest surface for *flow*, or explain why there is not one."""
    try:
        name, params = parse_configured_surface(flow)
    except ValueError as exc:
        raise RailCompilationError(f"{flow!r} is not a valid flow reference: {exc}") from exc

    normalized = normalize_configured_surface_name(name)
    surfaces = catalog.surfaces()
    surface = surfaces.get((direction, normalized))
    if surface is not None:
        return surface, params

    other_directions = sorted(key[0].value for key in surfaces if key[1] == normalized and key[0] is not direction)
    if other_directions:
        raise RailCompilationError(
            f"{flow!r} declares direction {direction.value!r} but {normalized!r} "
            f"is only available as {', '.join(other_directions)}"
        )
    raise RailCompilationError(f"{flow!r} has no surface named {normalized!r} in the rail catalog")


def _bind_parameters(surface: RailSurface, params: Mapping[str, str], flow: str) -> tuple[_BoundParameter, ...]:
    """Freeze the manifest's bindings into concrete values, failing now if one cannot be."""
    bound: list[_BoundParameter] = []
    for binding in surface.bindings:
        if binding.kind == "literal":
            bound.append(_BoundParameter(binding.action_param, binding.value))
            continue

        key = binding.key
        if key is None:
            raise RailCompilationError(
                f"{flow!r} declares a {binding.kind} binding for {binding.action_param!r} with no source key"
            )

        if binding.kind == "surface_param":
            if key in params:
                bound.append(_BoundParameter(binding.action_param, params[key]))
            elif binding.required:
                raise RailCompilationError(f"{flow!r} is missing required parameter ${key}=")
        # Context bindings name a conversation variable, which exists only per request, so
        # they are filled from the request context rather than frozen here.
    return tuple(bound)


def compile_rail(
    flow: str,
    direction: RailDirection,
    deps: RailDependencies,
    catalog: Optional["RailCatalog"] = None,
) -> CompiledRail:
    """Compile one configured flow string into an executable rail.

    Every way a config can be unservable surfaces here as ``RailCompilationError``: an
    unknown surface name, a surface declared for another direction, a missing required
    ``$param=``, or an action that cannot be imported. ``IORails.unsupported_reason``
    attempts this compilation to decide whether it can serve a config at all, so the same
    code path decides both, and construction can never fail after the check has passed.

    The action module is imported here, which is why an optional integration stays optional:
    a config with no GLiNER rail never imports GLiNER.
    """
    catalog = catalog if catalog is not None else default_rail_catalog()
    surface, params = _resolve_surface(flow, direction, catalog)

    try:
        action = resolve_import_ref(surface.action)
    except (ImportError, AttributeError) as exc:
        raise RailCompilationError(
            f"{flow!r} declares action {surface.action.name!r}, which cannot be imported: {exc}"
        ) from exc

    if not callable(action):
        raise RailCompilationError(f"{flow!r} resolved action {surface.action.name!r} to a non-callable")

    return CompiledRail(
        flow=flow,
        surface=surface,
        action=action,
        bound=_bind_parameters(surface, params, flow),
        deps=deps,
    )
