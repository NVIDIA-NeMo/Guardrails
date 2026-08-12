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

import importlib.metadata
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

from nemoguardrails.actions.rail_outcome import RailOutcome, require_rail_outcome
from nemoguardrails.guardrails.guardrails_types import LLMMessages
from nemoguardrails.guardrails.rail_guard import rail_error_outcome
from nemoguardrails.guardrails.telemetry import action_span
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.manifests import (
    Binding,
    RailDirection,
    RailSurface,
    default_rail_catalog,
    parse_configured_surface,
    resolve_import_ref,
)

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
    model_caches: Optional[Mapping[str, Any]] = None
    tracer: Optional["Tracer"] = None


@dataclass(frozen=True)
class RailExecution:
    """One rail run: its engine-neutral verdict, plus every model call the action made.

    The caller converts ``outcome`` into its own result type; ``CompiledRail`` never does.
    """

    outcome: RailOutcome
    llm_calls: tuple["LLMCallInfo", ...] = ()


_USER_MESSAGE_EVENT = "UserMessage"
_BOT_UTTERANCE_EVENT = "StartUtteranceBotAction"
_SYSTEM_MESSAGE_EVENT = "SystemMessage"


def _current_turn_index(messages: LLMMessages) -> Optional[int]:
    """Position of the turn being checked: the last user message that carries content."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and message.get("content"):
            return index
    return None


def _history_before_current_turn(messages: LLMMessages) -> LLMMessages:
    """The turns preceding the one being checked.

    Actions append the checked turn themselves, from ``context["user_message"]``, and always
    at the end. So the history stops short of it: emitting it here would hand the model the
    same turn twice, and emitting what follows it — an assistant reply in a ``check()``
    transcript, say — would place a later turn ahead of it and reorder the conversation.
    With no user turn to check, every message is history.
    """
    index = _current_turn_index(messages)
    return messages if index is None else messages[:index]


def messages_to_events(messages: LLMMessages) -> list[dict[str, Any]]:
    """Convert IORails messages into the event shapes conversation-history actions read.
    Used by actions which are tightly-coupled with colang event definitions for backwards-compatibility.
    """
    events: list[dict[str, Any]] = []
    for message in _history_before_current_turn(messages):
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
    index = _current_turn_index(messages)
    return "" if index is None else messages[index]["content"]


def _llm_calls_from(sink: list[dict[str, Any]]) -> tuple["LLMCallInfo", ...]:
    """Pull the LLMCallInfo records out of a processing-log sink."""
    return tuple(entry["data"] for entry in sink if entry.get("type") == "llm_call_info")


@dataclass(frozen=True)
class _BoundParameter:
    """One action parameter and the value the manifest says fills it."""

    action_param: str
    value: Any


@dataclass(frozen=True)
class _ContextParameter:
    """One action parameter and the conversation variable that fills it, per request.

    Unlike a literal or a surface parameter this cannot be frozen at compile time: its value
    is the request's own text, so freezing it would pin every later request to the first one.
    """

    action_param: str
    key: str


# The conversation variables a context binding may name. Also the keys of the ``context``
# dict injected wholesale into actions that declare it, so the two cannot drift.
_CONTEXT_KEYS = ("user_message", "bot_message")


def _request_context(messages: LLMMessages, bot_response: Optional[str]) -> dict[str, str]:
    """Build the conversation variables for one request."""
    return {"user_message": _last_user_content(messages), "bot_message": bot_response or ""}


class CompiledRail:
    """One configured flow, resolved to a library action and ready to run."""

    def __init__(
        self,
        *,
        flow: str,
        surface: RailSurface,
        action: Callable[..., Any],
        bound: tuple[_BoundParameter, ...],
        context_bound: tuple[_ContextParameter, ...],
        deps: RailDependencies,
        accepted: frozenset[str],
        http_client: Any = None,
    ) -> None:
        """Store the frozen execution plan. Build through :func:`compile_rail`.

        *accepted* is passed in rather than recomputed, so the set the bindings were validated
        against is by construction the one injection filters on.
        """
        self.flow = flow
        self.surface = surface
        self._action = action
        self._bound = bound
        self._context_bound = context_bound
        self._deps = deps
        self._accepted = accepted
        self._http_client = http_client

    @property
    def surface_name(self) -> str:
        """The manifest surface name, without any ``$param=`` suffix."""
        return self.surface.name

    async def run(self, messages: LLMMessages, bot_response: Optional[str] = None) -> RailOutcome:
        """Execute the rail and return its engine-neutral verdict."""
        return (await self.execute(messages, bot_response)).outcome

    async def execute(self, messages: LLMMessages, bot_response: Optional[str] = None) -> RailExecution:
        """Execute the rail, returning its verdict and the model calls it made.

        A fresh ``processing_log_var`` sink is installed around the action, so ``llm_calls``
        holds this rail's calls and only this rail's — live calls, cache hits and jailbreak's
        NIM call all append there, and a rail that reaches no model appends nothing.
        """
        sink: list[dict[str, Any]] = []
        token = processing_log_var.set(sink)
        try:
            with action_span(self._deps.tracer, self.surface_name) as span:
                try:
                    outcome = require_rail_outcome(await self._invoke(messages, bot_response))
                except Exception as exc:
                    outcome = rail_error_outcome(span, self.surface_name, exc)
        finally:
            processing_log_var.reset(token)

        return RailExecution(outcome=outcome, llm_calls=_llm_calls_from(sink))

    async def _invoke(self, messages: LLMMessages, bot_response: Optional[str]) -> Any:
        """Call the action, awaiting it only if it is asynchronous.

        Two shipped library actions are plain ``def`` (the guardrails_ai validators), so an
        unconditional await would raise TypeError on every request and the fail-closed
        envelope would report a working rail as a block. ``ActionDispatcher`` has always made
        the same allowance for LLMRails.
        """
        result = self._action(**self._call_kwargs(messages, bot_response))
        return await result if inspect.isawaitable(result) else result

    def _call_kwargs(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        """Assemble the action's arguments from its declared parameters and the manifest."""
        context = _request_context(messages, bot_response)
        kwargs = {
            name: value
            for name, value in self._request_dependencies(messages, context).items()
            if name in self._accepted
        }
        for bound in self._bound:
            kwargs[bound.action_param] = bound.value
        for context_bound in self._context_bound:
            kwargs[context_bound.action_param] = context[context_bound.key]
        return kwargs

    def _request_dependencies(self, messages: LLMMessages, context: Mapping[str, str]) -> dict[str, Any]:
        """Every value injectable by parameter name; the caller filters against the signature."""
        return {
            "llms": self._deps.llms,
            "llm": self._deps.llms.get("main"),
            "llm_task_manager": self._deps.llm_task_manager,
            "config": self._deps.config,
            "http_client": self._http_client,
            "model_caches": self._deps.model_caches,
            "context": context,
            "events": messages_to_events(messages),
        }

    async def close(self) -> None:
        """Close the rail's HTTP client, if it owns one.

        Failures propagate: releasing one client is the whole job here, and the caller
        closing a whole set of rails is the only layer that can decide a leak is
        survivable. Repeat calls are safe, as a closed client's ``close()`` is a no-op.
        """
        if self._http_client is not None:
            await self._http_client.close()


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
        surface_name, params = parse_configured_surface(flow)
    except ValueError as exc:
        raise RailCompilationError(f"{flow!r} is not a valid flow reference: {exc}") from exc

    surfaces = catalog.surfaces()
    surface = surfaces.get((direction, surface_name))
    if surface is not None:
        return surface, params

    # A surface configured in the wrong section is a likelier mistake than a misspelled name,
    # and the two are indistinguishable from the message alone, so name the sections it does
    # have. Appended rather than raised separately: one refusal, one place to change it.
    available = sorted({key[0].name for key in surfaces if key[1] == surface_name})
    hint = f"; it is available with direction {', '.join(available)}" if available else ""
    raise RailCompilationError(
        f"{flow!r} has no surface named {surface_name!r} with direction {direction.name} in the rail catalog{hint}"
    )


def _binding_source_key(binding: Binding, flow: str) -> str:
    """Return a non-literal binding's source key, or fail compilation naming the parameter."""
    if binding.key is None:
        raise RailCompilationError(
            f"{flow!r} declares a {binding.kind} binding for {binding.action_param!r} with no source key"
        )
    return binding.key


def _frozen_parameters(surface: RailSurface, params: Mapping[str, str], flow: str) -> tuple[_BoundParameter, ...]:
    """Freeze the bindings whose values are known at compile time: literals and $params."""
    bound: list[_BoundParameter] = []
    for binding in surface.bindings:
        if binding.kind == "literal":
            bound.append(_BoundParameter(binding.action_param, binding.value))
            continue

        key = _binding_source_key(binding, flow)
        if binding.kind == "surface_param":
            if key in params:
                bound.append(_BoundParameter(binding.action_param, params[key]))
            elif binding.required:
                raise RailCompilationError(f"{flow!r} is missing required parameter ${key}=")
            continue

        if binding.kind != "context":
            raise RailCompilationError(
                f"{flow!r} declares an unsupported {binding.kind!r} binding for {binding.action_param!r}"
            )
    return tuple(bound)


def _context_parameters(surface: RailSurface, flow: str) -> tuple[_ContextParameter, ...]:
    """Plan the bindings filled from the request's own conversation variables.

    A context binding maps one variable onto a specific action parameter, which is not the
    same as injecting the whole ``context`` dict: ``user_message`` reaches an action that
    calls the parameter ``text`` or ``user_prompt``.
    """
    bound: list[_ContextParameter] = []
    for binding in surface.bindings:
        if binding.kind != "context":
            continue
        key = _binding_source_key(binding, flow)
        if key not in _CONTEXT_KEYS:
            # Refused rather than passed an empty value: the rail would otherwise return a
            # verdict computed over evidence IORails never supplied.
            raise RailCompilationError(
                f"{flow!r} binds {binding.action_param!r} to context variable {key!r}, "
                f"which IORails does not supply; it has {list(_CONTEXT_KEYS)}"
            )
        bound.append(_ContextParameter(binding.action_param, key))
    return tuple(bound)


def _transform_target_reason(surface: RailSurface) -> Optional[str]:
    """Report a surface that rewrites content, which IORails cannot apply yet."""
    if surface.transform_target is None:
        return None
    return f"transforms {surface.transform_target.value!r}"


# Surfaces whose actions read retrieval evidence out of the request context: ``relevant_chunks``,
# ``relevant_chunks_sep``, or the Colang-internal ``_last_bot_prompt``. Keyed by direction as well
# as name, because one rail can surface in both directions.
_RETRIEVAL_CONTEXT_SURFACES: frozenset[tuple[RailDirection, str]] = frozenset(
    {
        (RailDirection.OUTPUT, "alignscore check facts"),
        (RailDirection.OUTPUT, "autoalign groundedness output"),
        (RailDirection.OUTPUT, "fiddler bot faithfulness"),
        (RailDirection.OUTPUT, "patronus api check output"),
        (RailDirection.OUTPUT, "patronus lynx check output hallucination"),
        (RailDirection.OUTPUT, "self check facts"),
        (RailDirection.OUTPUT, "self check hallucination"),
    }
)


def _retrieval_context_reason(surface: RailSurface) -> Optional[str]:
    """Report a surface needing retrieval evidence IORails has no source for."""
    if (surface.direction, surface.name) not in _RETRIEVAL_CONTEXT_SURFACES:
        return None
    return "needs retrieval evidence, which manifest-driven execution does not supply yet"


def _unsupported_rail_reason(surface: RailSurface) -> Optional[str]:
    """Manually edited blocklist of unsupported rails with reasons why"""

    blocked = {
        # TODO: Github Issue https://github.com/NVIDIA-NeMo/Guardrails/issues/2285
        (RailDirection.INPUT, "jailbreak detection heuristics"): (
            "Conflates dependencies with 'jailbreak detection model', so IORails "
            "cannot tell whether it needs 'torch' and 'transformers' installed"
        ),
    }
    return blocked.get((surface.direction, surface.name))


# Ordered so the cheapest, most structural check reports first. Each entry is removed by the
# work that lifts its limitation: the context-binding refusal went when resolution was built,
# and the transform refusal goes when IORails can apply a rewrite.
_SURFACE_SUPPORT_CHECKS: tuple[Callable[[RailSurface], Optional[str]], ...] = (
    _transform_target_reason,
    _retrieval_context_reason,
    _unsupported_rail_reason,
)


def unsupported_surface_reason(surface: RailSurface) -> Optional[str]:
    """Why manifest-driven execution cannot run *surface*, or None when it can."""
    for check in _SURFACE_SUPPORT_CHECKS:
        reason = check(surface)
        if reason is not None:
            return reason
    return None


def _accepts_arbitrary_keywords(action: Callable[..., Any]) -> bool:
    """Whether *action* has a ``**kwargs`` catch-all, so any keyword can be passed to it."""
    parameters = inspect.signature(action).parameters
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _reject_unaccepted_bindings(
    surface: RailSurface,
    action: Callable[..., Any],
    bound_params: Sequence[str],
    accepted: frozenset[str],
    flow: str,
) -> None:
    """Fail compilation when the manifest binds a parameter the action cannot be passed.

    Note the asymmetry with injection, which is easy to get wrong. Injection ignores
    ``**kwargs`` because "should this be *offered*?" must come from declared parameters, or a
    catch-all gets handed every dependency. This asks "can this be *passed*?", which a
    catch-all always can — so reusing the injection set here refuses actions that work.
    """
    if _accepts_arbitrary_keywords(action):
        return

    unaccepted = sorted(param for param in bound_params if param not in accepted)
    if not unaccepted:
        return
    raise RailCompilationError(
        f"{flow!r} binds {', '.join(repr(p) for p in unaccepted)}, which action "
        f"{surface.action.name!r} does not accept; it declares {sorted(accepted)}"
    )


# The parameter library actions resolve against ``llms``, by convention: an action needing a
# model declares ``model_name`` and indexes ``llms[model_name]``.
_MODEL_NAME_PARAM = "model_name"


def _reject_unconfigured_models(bound: tuple[_BoundParameter, ...], deps: RailDependencies, flow: str) -> None:
    """Fail compilation when a rail names a model type the configuration does not declare.

    The live gap is the *literal* binding. ``RailsConfig`` already rejects a ``$model=``
    naming an undeclared type (``check_model_exists_for_input_rails``), but it finds the model
    by parsing that suffix — so a rail whose model is baked into the manifest, such as
    ``llama guard check input``, passes config validation and then fails per request, where
    the fail-closed envelope reports the missing model as a rail block.
    """
    missing = sorted(
        {
            str(param.value)
            for param in bound
            if param.action_param == _MODEL_NAME_PARAM and param.value not in deps.llms
        }
    )
    if not missing:
        return
    raise RailCompilationError(
        f"{flow!r} needs model type(s) {', '.join(repr(m) for m in missing)}, "
        f"which the configuration does not define; it declares {sorted(deps.llms)}"
    )


def _owning_manifest(surface: RailSurface, catalog: "RailCatalog") -> Optional[Any]:
    """Return the manifest declaring *surface*, or None if the catalog has no owner for it."""
    for manifest in catalog.manifests.values():
        if surface in manifest.surfaces:
            return manifest
    return None


def _is_installed(distribution: str) -> bool:
    """Whether *distribution* is installed, without importing it."""
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _rail_config_section(config: Any, name: str) -> Any:
    """Return ``rails.config.<name>`` from a RailsConfig, or None when it is not configured."""
    rails = getattr(config, "rails", None)
    section = getattr(rails, "config", None) if rails is not None else None
    return getattr(section, name, None)


def _hf_classifier_runs_locally(config: Any, params: Mapping[str, str]) -> bool:
    """Whether the selected classifier runs in-process, per ``backends.get_backend``."""
    classifiers = _rail_config_section(config, "hf_classifier")
    if not classifiers:
        return True
    selected = classifiers.get(params.get("classifier", ""))
    if selected is None:
        return True
    return getattr(selected, "engine", "local") == "local"


def _jailbreak_detection_runs_locally(config: Any, params: Mapping[str, str]) -> bool:
    """Whether jailbreak detection runs in-process, which it does with no endpoint configured."""
    section = _rail_config_section(config, "jailbreak_detection")
    if section is None:
        return True
    return not (getattr(section, "server_endpoint", None) or getattr(section, "nim_base_url", None))


# TODO: replace with a backend selector declared in the manifest (#2279), which would remove
# the per-rail knowledge this table holds.
_LOCAL_BACKEND_CHECKS: dict[str, Callable[[Any, Mapping[str, str]], bool]] = {
    "hf_classifier": _hf_classifier_runs_locally,
    "jailbreak_detection": _jailbreak_detection_runs_locally,
}


def _reject_missing_dependencies(
    surface: RailSurface,
    catalog: "RailCatalog",
    flow: str,
    config: Any,
    params: Mapping[str, str],
) -> None:
    """Fail compilation when a rail's optional dependency is declared but not installed.

    Library actions import their optional dependency lazily, inside the function, as
    ``nemoguardrails/AGENTS.md`` requires. Nothing therefore fails until a request arrives, and
    the fail-closed envelope turns the ImportError into a *block* -- so a config missing an
    extra is indistinguishable, to the caller, from one whose rail genuinely tripped. Refusing
    at compile time reports it once, as the configuration error it is.

    A manifest offering both an in-process and a remote backend is enforced only when the
    configuration selects the in-process one; see ``_LOCAL_BACKEND_CHECKS``.

    Checked by distribution rather than by import, so it costs no import and needs no mapping
    from a package name to a module name -- ``guardrails-ai`` imports as ``guardrails``.
    """
    manifest = _owning_manifest(surface, catalog)
    if manifest is None:
        return

    runs_locally = _LOCAL_BACKEND_CHECKS.get(manifest.name)
    if runs_locally is not None and not runs_locally(config, params):
        return

    missing = sorted(dep for dep in manifest.requirements.optional_dependencies if not _is_installed(dep))
    if not missing:
        return

    extras = manifest.requirements.extras
    hint = f"; install the {extras[0]!r} extra" if extras else ""
    raise RailCompilationError(f"{flow!r} needs {', '.join(missing)}, which the environment does not have{hint}")


def unservable_reason(flow: str, direction: RailDirection, catalog: Optional["RailCatalog"] = None) -> Optional[str]:
    """Why *flow* cannot run under manifest-driven execution, or None when it can."""
    # Stops at the surface-level checks, so it never imports an action module.
    catalog = catalog if catalog is not None else default_rail_catalog()
    try:
        surface, _ = _resolve_surface(flow, direction, catalog)
    except RailCompilationError as exc:
        return str(exc)
    reason = unsupported_surface_reason(surface)
    return f"{flow!r} {reason}" if reason is not None else None


def compile_rail(
    flow: str,
    direction: RailDirection,
    deps: RailDependencies,
    *,
    http_client: Any = None,
    catalog: Optional["RailCatalog"] = None,
) -> CompiledRail:
    """Compile one configured flow string into an executable rail.

    Unservable rails raise a ``RailCompilationError``, validated at compile time.
    """
    catalog = catalog if catalog is not None else default_rail_catalog()
    surface, params = _resolve_surface(flow, direction, catalog)

    # Ahead of resolve_import_ref, so a refused surface never pulls in an optional dependency.
    unsupported = unsupported_surface_reason(surface)
    if unsupported is not None:
        raise RailCompilationError(f"{flow!r} {unsupported}")

    try:
        action = resolve_import_ref(surface.action)
    except (ImportError, AttributeError) as exc:
        raise RailCompilationError(
            f"{flow!r} declares action {surface.action.name!r}, which cannot be imported: {exc}"
        ) from exc

    if not callable(action):
        raise RailCompilationError(f"{flow!r} resolved action {surface.action.name!r} to a non-callable")

    accepted = _accepted_parameters(action)
    bound = _frozen_parameters(surface, params, flow)
    context_bound = _context_parameters(surface, flow)
    _reject_unaccepted_bindings(
        surface,
        action,
        [param.action_param for param in bound] + [param.action_param for param in context_bound],
        accepted,
        flow,
    )
    _reject_unconfigured_models(bound, deps, flow)
    _reject_missing_dependencies(surface, catalog, flow, deps.config, params)

    return CompiledRail(
        flow=flow,
        surface=surface,
        action=action,
        bound=bound,
        context_bound=context_bound,
        deps=deps,
        accepted=accepted,
        http_client=http_client,
    )
