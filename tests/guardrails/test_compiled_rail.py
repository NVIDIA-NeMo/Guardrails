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

"""Unit tests for CompiledRail, the manifest-driven rail unit that replaces RailAction.

A ``CompiledRail`` is built once per configured flow string: it reads the flow's
``RailSurface`` from the manifest catalog, resolves the library action, freezes a binding
plan, and thereafter turns a request into ``action(**kwargs)`` and passes the returned
``RailOutcome`` back to the caller. Converting to ``RailResult`` is ``RailsManager``'s job.

Two properties are load-bearing and are pinned here rather than left to review:

*Compilation fails at construction, not at request time.* An unknown surface, a
wrong-direction surface, or a missing required ``$param=`` must raise
``RailCompilationError`` while the engine is being built, because ``unsupported_reason``
decides whether IORails can serve a config by attempting exactly this compilation.

*Rail model calls are captured from a sink, never by reading a contextvar afterwards.*
``CompiledRail`` installs a fresh ``processing_log_var`` list around the action call.
A rail that makes no model call contributes nothing to its own empty list, so it cannot
inherit a previous rail's record — structurally, not by clearing anything. Do not
"simplify" this into a post-hoc read of ``llm_call_info_var``: library actions open and
close their own scope *inside* the action, so such a read yields the outer value.

Deferred to later commits, deliberately not covered here: transform outcomes (PR 5),
``model_caches`` (PR 6), and the ``RailsManager``/``unsupported_reason`` rewiring.
"""

import inspect
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.compiled_rail import (
    CompiledRail,
    RailCompilationError,
    RailDependencies,
    compile_rail,
    messages_to_events,
)
from nemoguardrails.library.content_safety.actions import (
    content_safety_check_input,
    content_safety_check_output,
)
from nemoguardrails.library.jailbreak_detection.actions import jailbreak_detection_model
from nemoguardrails.library.topic_safety.actions import topic_safety_check_input
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    RailCatalog,
    RailDirection,
    RailSurface,
    default_rail_catalog,
    resolve_import_ref,
)
from nemoguardrails.testing.fake_model import FakeLLMModel

CONTENT_SAFETY_INPUT = "content safety check input $model=content_safety"
CONTENT_SAFETY_ACTION = "nemoguardrails.library.content_safety.actions.content_safety_check_input"
TOPIC_SAFETY_INPUT = "topic safety check input $model=topic_control"
TOPIC_SAFETY_ACTION = "nemoguardrails.library.topic_safety.actions.topic_safety_check_input"
JAILBREAK_INPUT = "jailbreak detection model"

USER_MESSAGES = [{"role": "user", "content": "hello there"}]

SYNTHETIC_FLOW = "synthetic rail"
CONTENT_SAFETY_ACTION_REF = ActionRef(
    name="content_safety_check_input",
    target="nemoguardrails.library.content_safety.actions:content_safety_check_input",
)


class StubCatalog(RailCatalog):
    """A catalog holding one synthetic surface.

    ``compile_rail`` takes the catalog as a parameter precisely so a test can hand it a
    surface no shipped manifest produces — an unimportable action, a literal binding, a
    binding the schema forbids. Without this seam those compilation paths are unreachable.

    Subclasses the real catalog rather than duck-typing it: an empty record list constructs
    fine, and inheriting keeps the parameter's declared type honest instead of casting.
    """

    def __init__(self, surface: RailSurface, direction: RailDirection = RailDirection.INPUT):
        super().__init__(())
        self._stub_surfaces: dict[tuple[RailDirection, str], RailSurface] = {(direction, surface.name): surface}

    def surfaces(self, direction: Optional[RailDirection] = None) -> dict[tuple[RailDirection, str], RailSurface]:
        """Return the synthetic surface, filtered by *direction* as the real catalog does."""
        if direction is None:
            return self._stub_surfaces
        return {key: surface for key, surface in self._stub_surfaces.items() if key[0] is direction}


def synthetic_surface(
    action: ActionRef,
    bindings: tuple[Binding, ...] = (),
    *,
    bypass_validation: bool = False,
) -> RailSurface:
    """Build a one-off input surface for a compilation path the real catalog cannot reach.

    ``bypass_validation`` uses ``model_construct`` to skip Pydantic, which is the only way to
    build a manifest the schema rejects. Reach for it only when the branch under test exists
    to defend against exactly that.
    """
    if bypass_validation:
        return RailSurface.model_construct(
            name=SYNTHETIC_FLOW,
            direction=RailDirection.INPUT,
            action=action,
            bindings=bindings,
            transform_target=None,
        )
    return RailSurface(
        name=SYNTHETIC_FLOW,
        direction=RailDirection.INPUT,
        action=action,
        bindings=bindings,
        transform_target=None,
    )


class RecordingAction:
    """Stand-in for a library action that records how it was called.

    Copies *signature_of*'s parameters, so ``inspect.signature`` reports the same named
    parameters the real action declares. That is load-bearing rather than cosmetic:
    ``CompiledRail`` injects by parameter name and deliberately ignores ``**kwargs``, so a
    double declaring only ``**kwargs`` accepts everything at runtime while advertising
    nothing — it is handed its manifest bindings and no request dependencies at all. Pass
    the action being replaced, so injection is driven by a real shipped signature.
    """

    def __init__(self, outcome: Any = None, *, signature_of: Optional[Callable[..., Any]] = None):
        self.outcome = outcome if outcome is not None else RailOutcome.allow()
        self.kwargs: dict = {}
        if signature_of is not None:
            self.__signature__ = inspect.signature(signature_of)

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.outcome


@pytest.fixture
def deps() -> RailDependencies:
    """The dependency bundle CompiledRail injects, with inert stand-ins for live engines.

    The models are real ``FakeLLMModel`` instances rather than mocks so an action that
    genuinely calls one behaves as it would in production; nothing here reaches a network.
    """
    return RailDependencies(
        llms={
            "main": FakeLLMModel(responses=["main"]),
            "content_safety": FakeLLMModel(responses=["safe"]),
            "topic_control": FakeLLMModel(responses=["on-topic"]),
        },
        llm_task_manager=MagicMock(),
        config=MagicMock(),
        http_client=MagicMock(),
        model_caches=None,
        tracer=None,
    )


@pytest.fixture
def record_llm_call():
    """Append an LLMCallInfo to the active processing-log sink, as track_llm_call does.

    Simulates a rail's model call without standing up an engine, so capture behavior can be
    asserted directly. Mirrors ``logging/llm_tracker.py:59-61``; if that append changes
    shape, this fixture and the production reader must change together.
    """

    def _record(task: str) -> None:
        processing_log = processing_log_var.get()
        assert processing_log is not None, "no processing-log sink is installed; CompiledRail should install one"
        processing_log.append({"type": "llm_call_info", "timestamp": 0.0, "data": LLMCallInfo(task=task)})

    return _record


class TestCompilation:
    """Compilation resolves the surface and freezes a binding plan, or fails loudly."""

    def test_compiles_a_shipped_surface(self, deps):
        """A configured flow string resolves to its manifest surface and library action."""
        rail = compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps)

        assert isinstance(rail, CompiledRail)
        assert rail.flow == CONTENT_SAFETY_INPUT
        assert rail.surface_name == "content safety check input"

    def test_unknown_surface_name_raises(self, deps):
        """A flow with no catalog surface fails at compile time with the name in the message."""
        with pytest.raises(RailCompilationError, match="no surface"):
            compile_rail("not a real rail", RailDirection.INPUT, deps)

    def test_unparseable_flow_string_raises(self, deps):
        """A flow string that is not valid surface-reference syntax fails as a compilation error.

        The parser raises ``ValueError``; compilation must report it as a rail problem naming
        the flow, not let a bare parser error escape to the caller.
        """
        with pytest.raises(RailCompilationError, match="not a valid flow reference"):
            compile_rail("$model=orphaned", RailDirection.INPUT, deps)

    def test_wrong_direction_raises(self, deps):
        """An output-only surface configured as an input rail fails at compile time."""
        with pytest.raises(RailCompilationError, match="direction"):
            compile_rail("content safety check output $model=content_safety", RailDirection.INPUT, deps)

    def test_missing_required_surface_param_raises_at_compile_time(self, deps):
        """A surface needing $model= fails when the config omits it, not on the first request."""
        with pytest.raises(RailCompilationError, match="model"):
            compile_rail("content safety check input", RailDirection.INPUT, deps)

    def test_binding_a_parameter_the_action_rejects_raises_at_compile_time(self, deps, monkeypatch):
        """A manifest binding the action cannot accept fails compilation, not every request.

        Bindings are applied by keyword, so a mismatch would raise TypeError on each call and
        the fail-closed envelope would report it as a block — a configuration error wearing a
        rail verdict as a disguise.
        """

        async def action_without_model_name(llms, llm_task_manager, context):
            return RailOutcome.allow()

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, action_without_model_name)

        with pytest.raises(RailCompilationError, match="model_name"):
            compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps)

    def test_context_binding_surfaces_do_not_compile_yet(self, deps):
        """A surface using a context binding is refused rather than silently misbehaving.

        Context bindings map a conversation variable onto a named action parameter
        (``user_message`` into ``text``), which request-time injection does not do. Refusing
        to compile routes the config to LLMRails; PR 4 replaces this with real support.
        """
        with pytest.raises(RailCompilationError, match="context binding"):
            compile_rail("detect pii on input", RailDirection.INPUT, deps)

    def test_an_action_with_a_kwargs_catch_all_accepts_any_binding(self, deps, monkeypatch):
        """A ``**kwargs`` action is not refused, because it genuinely accepts the keyword.

        Guards the asymmetry that made the first version of this check wrong: the injection
        filter excludes ``**kwargs`` on purpose, and reusing that set to decide what can be
        *passed* refuses actions that work.
        """

        async def catch_all_action(**kwargs):
            return RailOutcome.allow()

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, catch_all_action)

        assert compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps) is not None


class TestMalformedManifest:
    """Compilation refuses a manifest that is well-formed on paper but cannot be executed.

    None of these arise from a shipped manifest, so each drives compilation through an
    injected catalog. They are worth pinning anyway: the catalog is populated by rglobbing
    ``library/**/rail.py``, so a third-party or in-progress manifest reaches this code, and
    the failure has to name the flow rather than surface as an import error from deep inside
    ``resolve_import_ref``.
    """

    def test_action_that_cannot_be_imported_raises(self, deps):
        """A manifest naming a module that does not exist fails compilation, not at import."""
        surface = synthetic_surface(ActionRef(name="ghost", target="nemoguardrails.no_such_module:action"))

        with pytest.raises(RailCompilationError, match="cannot be imported"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, StubCatalog(surface))

    def test_action_resolving_to_a_non_callable_raises(self, deps):
        """A manifest pointing at a module attribute that is not callable fails compilation.

        Resolution succeeds, so nothing raises until the rail is invoked — by which point the
        fail-closed envelope would report a config error as a rail block.
        """
        surface = synthetic_surface(
            ActionRef(name="not_a_function", target="nemoguardrails.guardrails.compiled_rail:_USER_MESSAGE_EVENT")
        )

        with pytest.raises(RailCompilationError, match="non-callable"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, StubCatalog(surface))

    def test_binding_with_no_source_key_raises(self, deps):
        """A non-literal binding carrying no source key fails compilation.

        Requires bypassing Pydantic at both the ``Binding`` and ``RailSurface`` level, because
        both reject it — so this guard is defence in depth against a manifest built through
        ``model_construct`` rather than a state the schema permits. It is retained rather than
        deleted because the alternative is a confusing ``$None=`` message further down.
        """
        keyless = Binding.model_construct(
            kind="surface_param", action_param="model_name", key=None, value=None, required=True
        )
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (keyless,), bypass_validation=True)

        with pytest.raises(RailCompilationError, match="no source key"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, StubCatalog(surface))


class TestManifestBindingContract:
    """Every manifest binding must name a parameter its action really declares.

    ``compile_rail`` can only refuse a binding the action cannot be *passed*, and a
    ``**kwargs`` catch-all can be passed anything — so a mistyped ``action_param`` would land
    silently in ``**kwargs`` and the action would quietly use its default. That gap is closed
    here instead: statically, across the whole catalog, where no monkeypatched double is
    involved and every surface is covered whether or not IORails can execute it yet.
    """

    def test_every_manifest_binding_names_a_declared_parameter(self):
        """No surface binds a parameter its action does not declare by name."""
        catalog = default_rail_catalog()
        mismatches: list[str] = []
        unimportable: list[str] = []
        checked: set[str] = set()

        for (direction, name), surface in catalog.surfaces().items():
            try:
                action = resolve_import_ref(surface.action)
            except Exception as exc:
                # An optional integration that is not installed in this environment.
                unimportable.append(f"{name} ({type(exc).__name__})")
                continue

            declared = {
                param
                for param, spec in inspect.signature(action).parameters.items()
                if spec.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            }
            checked.add(name)
            for binding in surface.bindings:
                if binding.action_param not in declared:
                    mismatches.append(
                        f"{direction.value} {name!r} binds {binding.action_param!r}, "
                        f"but {surface.action.name!r} declares {sorted(declared)}"
                    )

        assert not mismatches, "manifest bindings naming undeclared parameters:\n" + "\n".join(mismatches)

        # Guard against passing vacuously if a future environment cannot import anything:
        # the four rails IORails executes have no optional dependencies, so they are always
        # importable and must always have been checked.
        always_importable = {
            "content safety check input",
            "content safety check output",
            "topic safety check input",
            "jailbreak detection model",
        }
        assert always_importable <= checked, f"expected these to be checked, skipped: {sorted(unimportable)}"


class TestBindingResolution:
    """Each BindingKind fills its action parameter from the right source."""

    @pytest.mark.asyncio
    async def test_surface_param_binding_supplies_the_configured_value(self, deps, monkeypatch):
        """$model=content_safety reaches the action as model_name."""
        action = RecordingAction(signature_of=content_safety_check_input)
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, action)

        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert action.kwargs is not None
        assert action.kwargs["model_name"] == "content_safety"

    @pytest.mark.asyncio
    async def test_context_carries_the_request_messages(self, deps, monkeypatch):
        """The per-request context dict exposes user_message for context-bound actions."""
        action = RecordingAction(signature_of=content_safety_check_input)
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, action)

        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert action.kwargs["context"]["user_message"] == "hello there"

    @pytest.mark.asyncio
    async def test_literal_binding_supplies_a_constant(self, deps, monkeypatch):
        """A literal binding reaches the action as the value baked into the manifest.

        Uses a synthetic surface because every shipped surface carrying a literal binding
        belongs to an optional integration, and a unit test must not depend on which extras
        happen to be installed.
        """
        action = RecordingAction(signature_of=content_safety_check_input)
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, action)
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (Binding.literal("model_name", "baked_in"),))

        await compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, StubCatalog(surface)).run(USER_MESSAGES)

        assert action.kwargs["model_name"] == "baked_in"

    @pytest.mark.asyncio
    async def test_user_message_is_empty_when_the_request_has_no_user_turn(self, deps, monkeypatch):
        """A request with no user turn yields an empty user_message instead of raising.

        Matches the library actions, which read ``context.get(...)`` with a default and call
        the model with empty text. The hand-written rails raised here and failed closed.
        """
        action = RecordingAction(signature_of=content_safety_check_input)
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, action)

        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run([{"role": "system", "content": "hi"}])

        assert action.kwargs["context"]["user_message"] == ""

    @pytest.mark.asyncio
    async def test_bot_response_reaches_an_output_rail_as_bot_message(self, deps, monkeypatch):
        """An output rail's context carries the generated response under bot_message."""
        action = RecordingAction(signature_of=content_safety_check_output)
        monkeypatch.setattr("nemoguardrails.library.content_safety.actions.content_safety_check_output", action)

        rail = compile_rail("content safety check output $model=content_safety", RailDirection.OUTPUT, deps)
        await rail.run(USER_MESSAGES, bot_response="the reply")

        assert action.kwargs["context"]["bot_message"] == "the reply"


class TestDependencyInjection:
    """Injection is driven by inspect.signature, so an action gets only what it declares."""

    @pytest.mark.asyncio
    async def test_action_receives_only_the_parameters_it_declares(self, deps, monkeypatch):
        """An action declaring only llm_task_manager and model_name gets nothing else.

        That it is not handed llms, context or events is proved by the call succeeding:
        an undeclared keyword would raise TypeError, which the envelope would turn into a
        block. ``model_name`` is declared because the manifest binds it for this surface.
        """
        captured = {}

        async def narrow_action(llm_task_manager, model_name):
            captured["llm_task_manager"] = llm_task_manager
            captured["model_name"] = model_name
            return RailOutcome.allow()

        monkeypatch.setattr(TOPIC_SAFETY_ACTION, narrow_action)

        outcome = await compile_rail(TOPIC_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert not outcome.is_blocked
        assert captured == {"llm_task_manager": deps.llm_task_manager, "model_name": "topic_control"}

    @pytest.mark.asyncio
    async def test_events_are_supplied_to_actions_that_declare_them(self, deps, monkeypatch):
        """topic_safety_check_input declares events, so it receives synthesized ones."""
        action = RecordingAction(signature_of=topic_safety_check_input)
        monkeypatch.setattr(TOPIC_SAFETY_ACTION, action)

        await compile_rail(TOPIC_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert action.kwargs["events"] == [{"type": "UserMessage", "text": "hello there"}]

    @pytest.mark.asyncio
    async def test_http_client_is_supplied_to_vendor_actions(self, deps, monkeypatch):
        """jailbreak_detection_model declares http_client and receives the managed one."""
        action = RecordingAction(signature_of=jailbreak_detection_model)
        monkeypatch.setattr("nemoguardrails.library.jailbreak_detection.actions.jailbreak_detection_model", action)

        await compile_rail(JAILBREAK_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert action.kwargs["http_client"] is deps.http_client


class TestOutcomePassthrough:
    """The action's RailOutcome is returned unmodified; CompiledRail invents nothing."""

    @pytest.mark.asyncio
    async def test_allow_outcome_passes_through(self, deps, monkeypatch):
        """An ALLOW outcome is returned with its metadata intact."""
        expected = RailOutcome.allow(metadata={"policy_violations": []})
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction(expected, signature_of=content_safety_check_input))

        outcome = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert not outcome.is_blocked
        assert outcome.metadata == {"policy_violations": []}

    @pytest.mark.asyncio
    async def test_block_outcome_passes_through(self, deps, monkeypatch):
        """A BLOCK outcome is returned with its evidence intact."""
        expected = RailOutcome.block(metadata={"policy_violations": ["S1: Violence"]})
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction(expected, signature_of=content_safety_check_input))

        outcome = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert outcome.is_blocked
        assert outcome.metadata == {"policy_violations": ["S1: Violence"]}

    @pytest.mark.asyncio
    async def test_reason_is_passed_through_untouched(self, deps, monkeypatch):
        """A rail that supplies a reason keeps it verbatim; the engine never rewrites it."""
        monkeypatch.setattr(
            CONTENT_SAFETY_ACTION,
            RecordingAction(RailOutcome.block(reason="policy 4 tripped"), signature_of=content_safety_check_input),
        )

        outcome = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert outcome.reason == "policy 4 tripped"

    @pytest.mark.asyncio
    async def test_absent_reason_stays_absent(self, deps, monkeypatch):
        """A rail that supplies no reason yields reason=None rather than invented text."""
        monkeypatch.setattr(
            CONTENT_SAFETY_ACTION, RecordingAction(RailOutcome.block(), signature_of=content_safety_check_input)
        )

        outcome = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert outcome.reason is None

    @pytest.mark.asyncio
    async def test_non_outcome_return_is_rejected(self, deps, monkeypatch):
        """An action returning something other than RailOutcome fails closed, not silently."""
        monkeypatch.setattr(
            CONTENT_SAFETY_ACTION, RecordingAction(outcome="not an outcome", signature_of=content_safety_check_input)
        )

        outcome = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert outcome.is_blocked


class TestMessagesToEvents:
    """The message-to-event mapper feeds actions that read conversation history.

    Pinned from both ends: these shapes are what ``llm/filters.py:263-281`` consumes, so a
    change to that vocabulary must break here rather than silently drop topic safety's
    history.
    """

    def test_user_message_becomes_a_user_event(self):
        """A user turn maps to the UserMessage shape to_chat_messages reads."""
        assert messages_to_events([{"role": "user", "content": "hi"}]) == [{"type": "UserMessage", "text": "hi"}]

    def test_assistant_message_becomes_an_utterance_event(self):
        """An assistant turn maps to StartUtteranceBotAction with its script."""
        events = messages_to_events([{"role": "assistant", "content": "hello"}])

        assert events == [{"type": "StartUtteranceBotAction", "script": "hello"}]

    def test_system_message_becomes_a_system_event(self):
        """A system turn maps to SystemMessage with its content."""
        events = messages_to_events([{"role": "system", "content": "be brief"}])

        assert events == [{"type": "SystemMessage", "content": "be brief"}]

    def test_contentless_turn_is_skipped(self):
        """An assistant tool-call turn has no content and is dropped rather than crashing."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
        ]

        assert messages_to_events(messages) == [{"type": "UserMessage", "text": "hi"}]

    def test_order_is_preserved(self):
        """Turns keep conversation order so history reads correctly."""
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]

        assert [event["type"] for event in messages_to_events(messages)] == [
            "SystemMessage",
            "UserMessage",
            "StartUtteranceBotAction",
            "UserMessage",
        ]


class TestModelCallCapture:
    """Model calls are captured from a per-rail sink, so attribution cannot leak."""

    @pytest.mark.asyncio
    async def test_a_rail_that_makes_no_model_call_reports_none(self, deps, monkeypatch):
        """A vendor rail that never reaches a model produces no captured call."""
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction(signature_of=content_safety_check_input))

        execution = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).execute(USER_MESSAGES)

        assert execution.llm_calls == ()

    @pytest.mark.asyncio
    async def test_a_model_call_is_captured_with_its_task_label(self, deps, monkeypatch, record_llm_call):
        """A rail that calls a model captures that call's LLMCallInfo."""

        async def calling_action(**kwargs):
            record_llm_call(task="content_safety_check_input $model=content_safety")
            return RailOutcome.allow()

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, calling_action)

        execution = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).execute(USER_MESSAGES)

        assert len(execution.llm_calls) == 1
        assert execution.llm_calls[0].task == "content_safety_check_input $model=content_safety"

    @pytest.mark.asyncio
    async def test_a_model_free_rail_does_not_inherit_the_previous_rails_call(self, deps, monkeypatch, record_llm_call):
        """Running a model-free rail straight after a model-backed one captures nothing.

        This is the misattribution regression the whole sink design exists to prevent: with
        a post-hoc contextvar read the second rail would report the first rail's task,
        token counts, and model name as its own.
        """

        async def calling_action(**kwargs):
            record_llm_call(task="content_safety_check_input $model=content_safety")
            return RailOutcome.allow()

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, calling_action)
        monkeypatch.setattr(TOPIC_SAFETY_ACTION, RecordingAction(signature_of=topic_safety_check_input))

        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).execute(USER_MESSAGES)
        second = await compile_rail(TOPIC_SAFETY_INPUT, RailDirection.INPUT, deps).execute(USER_MESSAGES)

        assert second.llm_calls == ()


class TestFailsClosed:
    """A rail that raises is handled by the shared envelope, not by CompiledRail."""

    @pytest.mark.asyncio
    async def test_action_error_blocks(self, deps, monkeypatch):
        """An exception inside the action becomes a blocking outcome with a redacted reason."""

        async def exploding_action(**kwargs):
            raise RuntimeError("parser blew up")

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, exploding_action)

        outcome = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert outcome == RailOutcome.block(reason="content safety check input error: parser blew up")

    @pytest.mark.asyncio
    async def test_status_bearing_error_propagates(self, deps, monkeypatch):
        """A provider 503 leaves the rail rather than being reported as a guardrail block."""
        from nemoguardrails.exceptions import LLMCallException

        async def failing_action(**kwargs):
            raise LLMCallException("upstream refused", status=503)

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, failing_action)

        with pytest.raises(LLMCallException):
            await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)


# Invariant 8 (no Colang runtime in the rail path) is asserted in
# tests/llm/test_call_import_graph.py, next to the static import walker that can actually
# observe it. A sys.modules check here would pass vacuously: importing nemoguardrails at all
# loads both Colang runtimes through its __init__, so every module trivially "has" them.
