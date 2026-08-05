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
once, at engine construction. It resolves the flow's ``RailSurface`` from the manifest
catalog, imports the library action the surface declares, and freezes a plan for filling
that action's parameters. Thereafter each request is one ``await action(**kwargs)`` and
the returned ``RailOutcome`` is passed back to the caller unchanged.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from nemoguardrails.actions.rail_outcome import RailOutcome, require_rail_outcome
from nemoguardrails.guardrails.guardrails_types import LLMMessages
from nemoguardrails.guardrails.rail_guard import rail_error_outcome
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

    Raised while compiling, never while serving a request: a rail that fails mid-request
    produces a blocking outcome through ``rail_guard`` instead. The message is user-facing —
    it is why a config is not servable — so name the flow and what is wrong with it.
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
    """One rail run: its engine-neutral verdict, plus every model call the action made.

    ``llm_calls`` comes from a sink installed around the action, so it holds exactly this
    rail's calls — empty for a rail that reached no model.

    The caller (``RailsManager._run_rail``) converts ``outcome`` to a ``RailResult`` and
    attaches ``triggered_rail`` and ``records``. ``CompiledRail`` deliberately never does
    that conversion; it knows nothing about IORails' result type.
    """

    outcome: RailOutcome
    llm_calls: tuple["LLMCallInfo", ...] = ()


_USER_MESSAGE_EVENT = "UserMessage"
_BOT_UTTERANCE_EVENT = "StartUtteranceBotAction"
_SYSTEM_MESSAGE_EVENT = "SystemMessage"


def messages_to_events(messages: LLMMessages) -> list[dict[str, Any]]:
    """Convert IORails messages into the event shapes conversation-history actions read.
    Used by actions which are tightly-coupled with colang event definitions for backwards-compatibility.
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
        accepted: frozenset[str],
    ) -> None:
        """Store the frozen execution plan. Build through :func:`compile_rail`.

        *accepted* is computed and validated by ``compile_rail`` and passed in rather than
        recomputed here, so the parameter set the bindings were checked against is by
        construction the one request-time injection filters on.
        """
        self.flow = flow
        self.surface = surface
        self._action = action
        self._bound = bound
        self._deps = deps
        self._accepted = accepted

    @property
    def surface_name(self) -> str:
        """The manifest surface name, without any ``$param=`` suffix."""
        return self.surface.name

    async def run(self, messages: LLMMessages, bot_response: Optional[str] = None) -> RailOutcome:
        """Execute the rail and return its engine-neutral verdict."""
        return (await self.execute(messages, bot_response)).outcome

    async def execute(self, messages: LLMMessages, bot_response: Optional[str] = None) -> RailExecution:
        """Execute the rail, returning its verdict and the model calls it made.

        Model calls are collected by installing a fresh ``processing_log_var`` list for the
        duration of the action. Every path that produces an ``LLMCallInfo`` appends to that
        var — a live call through ``track_llm_call``, a cache hit, and jailbreak's NIM call
        — so the sink sees them all, and it sees only this rail's.
        """
        sink: list[dict[str, Any]] = []
        token = processing_log_var.set(sink)
        try:
            with action_span(self._deps.tracer, self.surface_name) as span:
                try:
                    outcome = require_rail_outcome(await self._action(**self._call_kwargs(messages, bot_response)))
                except Exception as exc:
                    outcome = rail_error_outcome(span, self.surface_name, exc)
        finally:
            processing_log_var.reset(token)

        return RailExecution(outcome=outcome, llm_calls=_llm_calls_from(sink))

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
        # Context bindings are rejected before this point by
        # _reject_unfillable_binding_kinds, so there is nothing to freeze for them here.
    return tuple(bound)


def _reject_unfillable_binding_kinds(surface: RailSurface, flow: str) -> None:
    """Fail compilation for a binding kind request-time injection cannot fill yet.

    A ``context`` binding maps a conversation variable onto a *specific* action parameter —
    ``user_message`` into ``text`` for most vendor rails. Injection does not do that: it
    supplies the whole ``context`` dict under the name ``context`` and nothing else. So a
    surface declaring one would call its action without a required argument on every
    request, and the fail-closed envelope would report that ``TypeError`` as a block —
    a config-level gap disguised as a rail verdict.

    Refusing to compile routes the config to LLMRails, which can run it. None of the
    surfaces reachable today declares one; this is the tripwire for when PR 4 widens the
    tier to the vendor rails, which use them heavily.
    """
    unfillable = sorted({binding.action_param for binding in surface.bindings if binding.kind == "context"})
    if not unfillable:
        return
    raise RailCompilationError(
        f"{flow!r} declares context binding(s) for {', '.join(repr(p) for p in unfillable)}, "
        f"which manifest-driven execution does not fill yet"
    )


def _accepts_arbitrary_keywords(action: Callable[..., Any]) -> bool:
    """Whether *action* has a ``**kwargs`` catch-all, so any keyword can be passed to it."""
    parameters = inspect.signature(action).parameters
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _reject_unaccepted_bindings(
    surface: RailSurface,
    action: Callable[..., Any],
    bound: tuple[_BoundParameter, ...],
    accepted: frozenset[str],
    flow: str,
) -> None:
    """Fail compilation when the manifest binds a parameter the action cannot be passed.

    Bindings are applied by keyword, so one the action cannot take raises ``TypeError`` on
    every request, which the fail-closed envelope turns into a silent block. Catching it
    here makes a manifest/action mismatch a loud configuration error instead, and is what
    lets ``unsupported_reason`` decide servability by attempting compilation.

    Note the asymmetry with injection, which is deliberate and easy to get wrong. Injection
    ignores ``**kwargs`` because "should this value be *offered*?" must be answered from
    declared parameters — a catch-all would otherwise be handed every dependency. This asks
    the narrower question "can this keyword be *passed*?", and a catch-all genuinely accepts
    anything, so it is nothing to reject. Reusing the injection set here would refuse actions
    that work.

    Manifests binding a parameter that only lands in ``**kwargs`` is a separate concern —
    a typo would be silently swallowed rather than raising — and is covered across the whole
    catalog by ``test_every_manifest_binding_names_a_declared_parameter``.
    """
    if _accepts_arbitrary_keywords(action):
        return

    unaccepted = sorted(param.action_param for param in bound if param.action_param not in accepted)
    if not unaccepted:
        return
    raise RailCompilationError(
        f"{flow!r} binds {', '.join(repr(p) for p in unaccepted)}, which action "
        f"{surface.action.name!r} does not accept; it declares {sorted(accepted)}"
    )


def compile_rail(
    flow: str,
    direction: RailDirection,
    deps: RailDependencies,
    catalog: Optional["RailCatalog"] = None,
) -> CompiledRail:
    """Compile one configured flow string into an executable rail.

    Unservable rails raise a ``RailCompilationError``, validated at compile time.
    """
    catalog = catalog if catalog is not None else default_rail_catalog()
    surface, params = _resolve_surface(flow, direction, catalog)

    # Don't import dependencies if an unsupported surface is compiled
    _reject_unfillable_binding_kinds(surface, flow)

    try:
        action = resolve_import_ref(surface.action)
    except (ImportError, AttributeError) as exc:
        raise RailCompilationError(
            f"{flow!r} declares action {surface.action.name!r}, which cannot be imported: {exc}"
        ) from exc

    if not callable(action):
        raise RailCompilationError(f"{flow!r} resolved action {surface.action.name!r} to a non-callable")

    accepted = _accepted_parameters(action)
    bound = _bind_parameters(surface, params, flow)
    _reject_unaccepted_bindings(surface, action, bound, accepted, flow)

    return CompiledRail(
        flow=flow,
        surface=surface,
        action=action,
        bound=bound,
        deps=deps,
        accepted=accepted,
    )
