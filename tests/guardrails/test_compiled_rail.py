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

"""Unit tests for CompiledRail, the manifest-driven rail unit.

Two properties are load-bearing and pinned here rather than left to review: compilation
fails at construction rather than at request time, and rail model calls are captured from a
sink rather than read from a contextvar afterwards.
"""

import inspect
from dataclasses import replace
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.guardrails.compiled_rail import (
    _RETRIEVAL_CONTEXT_SURFACES,
    CompiledRail,
    RailCompilationError,
    RailDependencies,
    _hf_classifier_runs_locally,
    _is_installed,
    _jailbreak_detection_runs_locally,
    _unapplicable_transform_reason,
    compile_rail,
    messages_to_events,
    unservable_reason,
    unsupported_surface_reason,
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
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.testing.fake_model import FakeLLMModel

CONTENT_SAFETY_INPUT = "content safety check input $model=content_safety"
CONTENT_SAFETY_ACTION = "nemoguardrails.library.content_safety.actions.content_safety_check_input"
TOPIC_SAFETY_INPUT = "topic safety check input $model=topic_control"
TOPIC_SAFETY_ACTION = "nemoguardrails.library.topic_safety.actions.topic_safety_check_input"
JAILBREAK_INPUT = "jailbreak detection model"

USER_MESSAGES = [{"role": "user", "content": "hello there"}]

SYNTHETIC_FLOW = "synthetic rail"
TAKES_TEXT_TARGET = "tests.guardrails.test_compiled_rail:takes_text"
CONTENT_SAFETY_ACTION_REF = ActionRef(
    name="content_safety_check_input",
    target="nemoguardrails.library.content_safety.actions:content_safety_check_input",
)

EXECUTABLE_SURFACES = {
    "content safety check input",
    "content safety check output",
    "topic safety check input",
    "jailbreak detection model",
}

# Surfaces whose actions read retrieval evidence from the request context —
# ``relevant_chunks``, ``relevant_chunks_sep``, or the Colang-internal ``_last_bot_prompt``.
# Keyed by direction as well as name, because one rail can surface in both directions.
# Held here independently of the production deny-list so the two cross-check each other.
# Blocked by decision rather than by anything the manifest declares. Held here independently
# of the production blocklist so the two cross-check each other.
UNSUPPORTED_RAIL_SURFACES = {(RailDirection.INPUT, "jailbreak detection heuristics")}

RETRIEVAL_DEPENDENT_SURFACES = {
    (RailDirection.OUTPUT, "alignscore check facts"),
    (RailDirection.OUTPUT, "autoalign groundedness output"),
    (RailDirection.OUTPUT, "fiddler bot faithfulness"),
    (RailDirection.OUTPUT, "patronus api check output"),
    (RailDirection.OUTPUT, "patronus lynx check output hallucination"),
    (RailDirection.OUTPUT, "self check facts"),
    (RailDirection.OUTPUT, "self check hallucination"),
}


def _declared_parameters(action: Callable[..., Any]) -> set[str]:
    """Parameter names *action* declares, excluding ``*args`` and ``**kwargs``."""
    return {
        name
        for name, spec in inspect.signature(action).parameters.items()
        if spec.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


class StubCatalog(RailCatalog):
    """A catalog holding one synthetic surface no shipped manifest would produce.

    Subclasses the real catalog rather than duck-typing it, so the declared parameter type
    stays honest without a cast.
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
    direction: RailDirection = RailDirection.INPUT,
    transform_target: Optional[TransformTarget] = None,
) -> RailSurface:
    """Build a one-off surface for a compilation path the real catalog cannot reach.

    ``bypass_validation`` skips Pydantic, the only way to build a manifest the schema rejects.
    """
    if bypass_validation:
        return RailSurface.model_construct(
            name=SYNTHETIC_FLOW,
            direction=direction,
            action=action,
            bindings=bindings,
            transform_target=transform_target,
        )
    return RailSurface(
        name=SYNTHETIC_FLOW,
        direction=direction,
        action=action,
        bindings=bindings,
        transform_target=transform_target,
    )


def uncompiled_rail(surface: RailSurface, deps: RailDependencies) -> CompiledRail:
    """Build a ``CompiledRail`` around *surface* without going through compilation.

    The only way to hold a rail for a surface compilation still refuses, which is what a
    rewriting surface is until IORails can apply the rewrite.
    """
    return CompiledRail(
        flow=surface.name,
        surface=surface,
        action=takes_text,
        bound=(),
        context_bound=(),
        deps=deps,
        accepted=frozenset(),
    )


async def takes_text(text, config, **kwargs):
    """Signature reference: a vendor action receiving its content under ``text``.

    Thirteen shipped surfaces bind ``user_message`` onto a parameter named something other
    than ``user_message``; this is the shape they share.
    """
    return RailOutcome.allow()


async def takes_only_text(text):
    """Signature reference with no catch-all, so an unbindable parameter is refused."""
    return RailOutcome.allow()


class RecordingAction:
    """Stand-in for a library action that records how it was called.

    Always pass *signature_of*. Injection is by parameter name and ignores ``**kwargs``, so a
    double declaring only ``**kwargs`` advertises nothing and receives no dependencies at all
    — every injection assertion then fails for a reason unrelated to the code under test.
    """

    def __init__(self, outcome: Any = None, *, signature_of: Optional[Callable[..., Any]] = None):
        self.outcome = outcome if outcome is not None else RailOutcome.allow()
        self.kwargs: dict = {}
        if signature_of is not None:
            self.__signature__ = inspect.signature(signature_of)

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.outcome


_HF_FLOW = 'hf classifier check input $classifier="toxicity"'
_HF_LOCAL_CONFIG = {"hf_classifier": {"toxicity": {"engine": "local", "model": "m", "blocked_labels": ["toxic"]}}}
_HF_OTHER_CLASSIFIER_CONFIG = {
    "hf_classifier": {
        "profanity": {"engine": "vllm", "model": "m", "base_url": "http://vllm:8000", "blocked_labels": ["p"]}
    }
}
_REGEX_CONFIG = {"regex": {"input": {"patterns": [{"name": "secret", "pattern": "SECRET-[0-9]+"}]}}}
_HF_REMOTE_CONFIG = {
    "hf_classifier": {
        "toxicity": {"engine": "vllm", "model": "m", "base_url": "http://vllm:8000", "blocked_labels": ["toxic"]}
    }
}


def _deps_with_rails_config(rails_config: dict) -> RailDependencies:
    """Dependencies carrying a real RailsConfig, which the dependency check reads.

    A MagicMock config would answer every backend question truthily and exempt every rail,
    so these cases need the real model.
    """
    config = RailsConfig.from_content(
        config={
            "models": [{"type": "main", "engine": "openai", "model": "placeholder"}],
            "rails": {"config": rails_config},
        }
    )
    return RailDependencies(llms={"main": None, "content_safety": None}, llm_task_manager=MagicMock(), config=config)


@pytest.fixture
def deps() -> RailDependencies:
    """The dependency bundle CompiledRail injects, with inert stand-ins for live engines.

    The models are real ``FakeLLMModel`` instances rather than mocks so an action that
    genuinely calls one behaves as it would in production; nothing here reaches a network.
    ``llms`` also names the model types compilation validates model bindings against.
    """
    return RailDependencies(
        llms={
            "main": FakeLLMModel(responses=["main"]),
            "content_safety": FakeLLMModel(responses=["safe"]),
            "topic_control": FakeLLMModel(responses=["on-topic"]),
        },
        llm_task_manager=MagicMock(),
        config=MagicMock(),
        model_caches=None,
        tracer=None,
    )


@pytest.fixture
def content_safety_action(monkeypatch) -> RecordingAction:
    """Patch the content-safety action with a recording double and hand it back."""
    action = RecordingAction(signature_of=content_safety_check_input)
    monkeypatch.setattr(CONTENT_SAFETY_ACTION, action)
    return action


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

    def test_a_context_binding_surface_compiles(self, deps):
        """A surface using a context binding compiles, where it was refused before PR 4.

        Inverts the 3a tripwire deliberately: ``_unfillable_bindings_reason`` existed to keep
        these surfaces off IORails until request-time resolution was built, and this is the
        assertion that says it now is.
        """
        assert compile_rail("detect pii on input", RailDirection.INPUT, deps) is not None

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

    None arise from a shipped manifest, so each drives compilation through an injected
    catalog. Still worth pinning: the catalog rglobs ``library/**/rail.py``, so a third-party
    manifest reaches this code and the failure has to name the flow.
    """

    def test_action_that_cannot_be_imported_raises(self, deps):
        """A manifest naming a module that does not exist fails compilation, not at import."""
        surface = synthetic_surface(ActionRef(name="ghost", target="nemoguardrails.no_such_module:action"))

        with pytest.raises(RailCompilationError, match="cannot be imported"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface))

    def test_action_resolving_to_a_non_callable_raises(self, deps):
        """A manifest pointing at a module attribute that is not callable fails compilation.

        Resolution succeeds, so nothing raises until the rail is invoked — by which point the
        fail-closed envelope would report a config error as a rail block.
        """
        surface = synthetic_surface(
            ActionRef(name="not_a_function", target="nemoguardrails.guardrails.compiled_rail:_USER_MESSAGE_EVENT")
        )

        with pytest.raises(RailCompilationError, match="non-callable"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface))

    def test_binding_with_no_source_key_raises(self, deps):
        """A non-literal binding carrying no source key fails compilation.

        Needs a Pydantic bypass at both the ``Binding`` and ``RailSurface`` level, since both
        reject it: the guard is defence in depth, not a state the schema permits.
        """
        keyless = Binding.model_construct(
            kind="surface_param", action_param="model_name", key=None, value=None, required=True
        )
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (keyless,), bypass_validation=True)

        with pytest.raises(RailCompilationError, match="no source key"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface))

    def test_binding_kind_the_binder_cannot_fill_raises(self, deps):
        """A binding kind the binder does not handle fails compilation rather than vanishing.

        Unreachable for the same reasons as the keyless case, so it needs the same bypass. The
        alternative is a binding silently dropped, leaving the action short a parameter.
        """
        unhandled = Binding.model_construct(
            kind="conversation_state", action_param="model_name", key="model", value=None, required=True
        )
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (unhandled,), bypass_validation=True)

        with pytest.raises(RailCompilationError, match="unsupported"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface))


class TestUnrunnableSurfaces:
    """A surface the catalog can compile but IORails cannot execute is refused, not run."""

    @pytest.mark.parametrize("name", sorted(EXECUTABLE_SURFACES))
    def test_an_executable_surface_reports_no_reason(self, name):
        """Each shipped rail passes every support check, so compilation proceeds to the action."""
        surfaces = default_rail_catalog().surfaces()
        surface = next(s for (_, surface_name), s in surfaces.items() if surface_name == name)

        assert unsupported_surface_reason(surface) is None

    def test_a_rewriting_surface_compiles(self, deps):
        """A rail that rewrites its own direction's variable is runnable, which is the point of it."""
        assert compile_rail("autoalign check input", RailDirection.INPUT, deps) is not None

    @pytest.mark.parametrize(
        "direction, target",
        [
            (RailDirection.INPUT, TransformTarget.BOT_MESSAGE),
            (RailDirection.OUTPUT, TransformTarget.USER_MESSAGE),
            (RailDirection.INPUT, TransformTarget.RELEVANT_CHUNKS),
        ],
        ids=["input-rewrites-bot", "output-rewrites-user", "input-rewrites-chunks"],
    )
    def test_a_surface_rewriting_the_wrong_variable_does_not_compile(self, deps, direction, target):
        """A rewrite this engine has no variable for is refused rather than applied to the wrong one.

        Nothing in the manifest schema forbids the mismatch, so the refusal is what keeps it out.
        """
        surface = synthetic_surface(
            ActionRef(name="takes_text", target=TAKES_TEXT_TARGET), direction=direction, transform_target=target
        )

        with pytest.raises(RailCompilationError, match="cannot apply"):
            compile_rail(SYNTHETIC_FLOW, direction, deps, catalog=StubCatalog(surface, direction))


class TestUnapplicableTransformReason:
    """`_unapplicable_transform_reason` refuses a declared rewrite this engine cannot place."""

    def _surface(self, direction: RailDirection, target) -> RailSurface:
        return synthetic_surface(
            ActionRef(name="takes_text", target=TAKES_TEXT_TARGET), direction=direction, transform_target=target
        )

    def test_a_rail_that_only_judges_has_nothing_to_refuse(self):
        """The common surface, which declares no rewrite at all."""
        assert _unapplicable_transform_reason(self._surface(RailDirection.INPUT, None)) is None

    @pytest.mark.parametrize(
        "direction, target",
        [
            (RailDirection.INPUT, TransformTarget.USER_MESSAGE),
            (RailDirection.OUTPUT, TransformTarget.BOT_MESSAGE),
        ],
        ids=["input-user", "output-bot"],
    )
    def test_a_direction_rewriting_its_own_variable_is_servable(self, direction, target):
        """The agreement every shipped surface has, and the reason this check refuses nothing today."""
        assert _unapplicable_transform_reason(self._surface(direction, target)) is None

    @pytest.mark.parametrize(
        "direction, target",
        [
            (RailDirection.INPUT, TransformTarget.BOT_MESSAGE),
            (RailDirection.OUTPUT, TransformTarget.USER_MESSAGE),
            (RailDirection.INPUT, TransformTarget.RELEVANT_CHUNKS),
            (RailDirection.RETRIEVAL, TransformTarget.RELEVANT_CHUNKS),
        ],
        ids=["input-bot", "output-user", "input-chunks", "retrieval"],
    )
    def test_a_rewrite_with_nowhere_to_land_reports_why(self, direction, target):
        """Nothing in the manifest schema forbids the mismatch, so the refusal is what keeps it out."""
        reason = _unapplicable_transform_reason(self._surface(direction, target))

        assert reason is not None
        assert target.value in reason
        assert "cannot apply" in reason


class TestDeclaredTransformTarget:
    """A rail reports the variable its surface says it may rewrite, which is how it is scheduled."""

    def test_a_block_only_rail_declares_no_rewrite(self, deps):
        """The shipped rails all judge without rewriting, so nothing about their scheduling changes."""
        assert compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).transform_target is None

    def test_a_rewriting_rail_reports_the_variable_it_may_rewrite(self, deps):
        """Scheduling reads the manifest's declaration, not a verdict some request happened to produce."""
        surface = synthetic_surface(
            ActionRef(name="takes_text", target=TAKES_TEXT_TARGET), transform_target=TransformTarget.USER_MESSAGE
        )

        assert uncompiled_rail(surface, deps).transform_target is TransformTarget.USER_MESSAGE

    def test_the_declaration_is_independent_of_what_a_rail_answers(self, deps):
        """Most requests to a rewriting rail come back as a plain allow; it is still scheduled first."""
        surface = synthetic_surface(
            ActionRef(name="takes_text", target=TAKES_TEXT_TARGET), transform_target=TransformTarget.USER_MESSAGE
        )
        rail = uncompiled_rail(surface, deps)

        assert rail.transform_target is TransformTarget.USER_MESSAGE
        assert rail.surface.transform_target is TransformTarget.USER_MESSAGE


class TestSurfacesOutsideTheTier:
    """Which catalog surfaces this engine refuses, and that the refusal is cheap."""

    @pytest.mark.parametrize(
        "direction, flow",
        sorted(RETRIEVAL_DEPENDENT_SURFACES, key=lambda surface: surface[1]),
        ids=lambda value: value if isinstance(value, str) else value.value,
    )
    def test_retrieval_dependent_surfaces_do_not_compile(self, deps, direction, flow):
        """A surface reading retrieval evidence is refused; IORails supplies none."""
        with pytest.raises(RailCompilationError, match="retrieval"):
            compile_rail(flow, direction, deps)

    def test_the_deny_list_names_only_real_surfaces(self):
        """Every refused surface exists in the catalog, so no entry can rot into a no-op."""
        surfaces = default_rail_catalog().surfaces()
        missing = sorted(
            f"{direction.value} {name!r}"
            for direction, name in _RETRIEVAL_CONTEXT_SURFACES
            if (direction, name) not in surfaces
        )

        assert not missing, f"deny-list names surfaces the catalog does not have: {missing}"

    def test_the_input_output_tier_splits_into_servable_and_refused(self):
        """Every input/output surface is servable but for the eight refused ones.

        Measured: 59 servable -- 41 that only judge and the 18 that may rewrite -- against 8
        refused: the seven retrieval-evidence surfaces plus jailbreak heuristics, whose backend
        cannot be told apart from its manifest sibling's. Pinning the split rather than a
        predicate means a newly added manifest surface, or an action that grows a binding
        IORails cannot fill, fails here rather than in a config.
        """
        servable, refused = [], []
        for (direction, name), surface in default_rail_catalog().surfaces().items():
            if direction is RailDirection.RETRIEVAL:
                continue
            bucket = refused if unsupported_surface_reason(surface) is not None else servable
            bucket.append((direction, name))

        assert sorted(refused) == sorted(RETRIEVAL_DEPENDENT_SURFACES | UNSUPPORTED_RAIL_SURFACES)
        assert len(servable) == 59

    def test_the_rewriting_surfaces_are_all_servable(self):
        """The eighteen this work exists for: nine each way, every one of them runnable.

        Counted rather than named, because the names are the manifests' business; what this
        pins is that no direction is left half-enabled.
        """
        rewriting = [
            (direction, name)
            for (direction, name), surface in default_rail_catalog().surfaces().items()
            if direction is not RailDirection.RETRIEVAL and surface.transform_target is not None
        ]

        assert len(rewriting) == 18
        assert [key for key in rewriting if key[0] is RailDirection.INPUT] != []
        assert [key for key in rewriting if key[0] is RailDirection.OUTPUT] != []
        assert all(unsupported_surface_reason(default_rail_catalog().surfaces()[key]) is None for key in rewriting)

    def test_refusal_precedes_the_action_import(self, deps, monkeypatch):
        """An unrunnable surface is refused before its action module is imported."""

        def unreachable(ref):
            raise AssertionError(f"resolve_import_ref ran for a refused surface: {ref}")

        monkeypatch.setattr("nemoguardrails.guardrails.compiled_rail.resolve_import_ref", unreachable)

        with pytest.raises(RailCompilationError, match="retrieval"):
            compile_rail("self check facts", RailDirection.OUTPUT, deps)


class TestUnservableReason:
    """`unservable_reason` reports why a flow cannot run here so engine selection can fall back."""

    def test_an_unknown_flow_reports_the_missing_surface(self):
        """A flow with no catalog surface yields a reason, rather than raising during selection."""
        reason = unservable_reason("not a real rail", RailDirection.INPUT)

        assert reason is not None
        assert "no surface" in reason

    def test_a_shipped_surface_reports_no_reason(self):
        """The control: a runnable flow yields None, so the reason above is not unconditional."""
        assert unservable_reason(CONTENT_SAFETY_INPUT, RailDirection.INPUT) is None


class TestManifestBindingContract:
    """Every manifest binding must name a parameter its action really declares.

    ``compile_rail`` only refuses what an action cannot be *passed*, and a ``**kwargs``
    catch-all accepts anything — so a mistyped ``action_param`` lands there silently and the
    action uses its default. Closed statically here, across every surface.
    """

    def test_every_manifest_binding_names_a_declared_parameter(self):
        """No surface binds a parameter its action does not declare by name."""
        mismatches: list[str] = []
        checked: set[str] = set()

        for (direction, name), surface in default_rail_catalog().surfaces().items():
            try:
                action = resolve_import_ref(surface.action)
            except Exception:
                continue  # an optional integration that is not installed here
            declared = _declared_parameters(action)
            checked.add(name)
            mismatches += [
                f"{direction.value} {name!r} binds {b.action_param!r}, not in {sorted(declared)}"
                for b in surface.bindings
                if b.action_param not in declared
            ]

        assert not mismatches, "manifest bindings naming undeclared parameters:\n" + "\n".join(mismatches)
        # Cannot pass vacuously: these four have no optional dependencies, so they always import.
        assert EXECUTABLE_SURFACES <= checked, f"only checked {sorted(checked)}"


class TestBindingResolution:
    """Each BindingKind fills its action parameter from the right source."""

    @pytest.mark.asyncio
    async def test_surface_param_binding_supplies_the_configured_value(self, deps, content_safety_action):
        """$model=content_safety reaches the action as model_name."""
        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert content_safety_action.kwargs["model_name"] == "content_safety"

    @pytest.mark.asyncio
    async def test_context_carries_the_request_messages(self, deps, content_safety_action):
        """The per-request context dict exposes user_message for context-bound actions."""
        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert content_safety_action.kwargs["context"]["user_message"] == "hello there"

    @pytest.mark.asyncio
    async def test_literal_binding_supplies_a_constant(self, deps, content_safety_action):
        """A literal binding reaches the action as the value baked into the manifest.

        Uses a synthetic surface because every shipped surface with a literal binding belongs
        to an optional integration, and a unit test must not depend on installed extras.
        ``threshold_mode`` mirrors what activefence and gcpnlp really bake in; binding
        ``model_name`` here instead would drag in model validation, which is not the subject.
        """
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (Binding.literal("threshold_mode", "detailed"),))

        await compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface)).run(USER_MESSAGES)

        assert content_safety_action.kwargs["threshold_mode"] == "detailed"

    @pytest.mark.asyncio
    async def test_user_message_is_empty_when_the_request_has_no_user_turn(self, deps, content_safety_action):
        """A request with no user turn yields an empty user_message instead of raising.

        Matches the library actions, which read ``context.get(...)`` with a default and call
        the model with empty text. The hand-written rails raised here and failed closed.
        """
        await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run([{"role": "system", "content": "hi"}])

        assert content_safety_action.kwargs["context"]["user_message"] == ""

    @pytest.mark.asyncio
    async def test_bot_response_reaches_an_output_rail_as_bot_message(self, deps, monkeypatch):
        """An output rail's context carries the generated response under bot_message."""
        action = RecordingAction(signature_of=content_safety_check_output)
        monkeypatch.setattr("nemoguardrails.library.content_safety.actions.content_safety_check_output", action)

        rail = compile_rail("content safety check output $model=content_safety", RailDirection.OUTPUT, deps)
        await rail.run(USER_MESSAGES, bot_response="the reply")

        assert action.kwargs["context"]["bot_message"] == "the reply"


class TestContextBindingResolution:
    """A context binding maps one conversation variable onto a differently named parameter.

    This is the kind 3a refused and PR 4 implements. It is not "pass the context dict" — the
    dict is already injected under ``context`` for actions that declare it. A context binding
    names *one* variable and *one* action parameter, and 25 block-only surfaces need it.
    """

    @pytest.fixture
    def text_action(self, monkeypatch) -> RecordingAction:
        """A recording double shaped like a vendor action that takes its content as ``text``."""
        action = RecordingAction(signature_of=takes_text)
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, action)
        return action

    @pytest.mark.asyncio
    async def test_user_message_fills_a_differently_named_parameter(self, deps, text_action):
        """``user_message`` reaches an action that calls the parameter ``text``."""
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (Binding.context("text", "user_message"),))

        await compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface)).run(USER_MESSAGES)

        assert text_action.kwargs["text"] == "hello there"

    @pytest.mark.asyncio
    async def test_bot_message_fills_the_bound_parameter_on_an_output_rail(self, deps, text_action):
        """``bot_message`` reaches the bound parameter, which is how the 12 output surfaces read."""
        surface = synthetic_surface(
            CONTENT_SAFETY_ACTION_REF,
            (Binding.context("text", "bot_message"),),
            direction=RailDirection.OUTPUT,
        )
        catalog = StubCatalog(surface, RailDirection.OUTPUT)

        rail = compile_rail(SYNTHETIC_FLOW, RailDirection.OUTPUT, deps, catalog=catalog)
        await rail.run(USER_MESSAGES, bot_response="the reply")

        assert text_action.kwargs["text"] == "the reply"

    @pytest.mark.asyncio
    async def test_the_value_is_resolved_per_request(self, deps, text_action):
        """Two requests through one compiled rail bind two different values.

        A context binding cannot be frozen the way ``literal`` and ``surface_param`` are, and
        freezing it would pin every later request to the first request's message.
        """
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (Binding.context("text", "user_message"),))
        rail = compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface))

        await rail.run([{"role": "user", "content": "first"}])
        first = text_action.kwargs["text"]
        await rail.run([{"role": "user", "content": "second"}])

        assert (first, text_action.kwargs["text"]) == ("first", "second")

    def test_a_context_key_iorails_cannot_supply_is_refused_at_compile_time(self, deps):
        """A binding naming a variable outside the request context fails compilation.

        The request context holds ``user_message`` and ``bot_message`` and nothing else, so a
        binding on ``relevant_chunks`` would otherwise pass the action an empty value and the
        rail would return a verdict computed over no evidence.
        """
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (Binding.context("text", "relevant_chunks"),))

        with pytest.raises(RailCompilationError, match="relevant_chunks"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface))

    def test_a_context_binding_the_action_does_not_declare_is_refused(self, deps, monkeypatch):
        """A context binding is checked against the signature like every other kind.

        Without this the keyword lands on an action that cannot take it, raising TypeError per
        request, which the fail-closed envelope reports as a rail block.
        """
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, takes_only_text)
        surface = synthetic_surface(CONTENT_SAFETY_ACTION_REF, (Binding.context("user_prompt", "user_message"),))

        with pytest.raises(RailCompilationError, match="user_prompt"):
            compile_rail(SYNTHETIC_FLOW, RailDirection.INPUT, deps, catalog=StubCatalog(surface))


class TestSynchronousActions:
    """A synchronous library action runs; ``CompiledRail`` does not assume every action awaits."""

    @pytest.mark.asyncio
    async def test_a_synchronous_action_runs(self, deps, monkeypatch):
        """A plain ``def`` action's outcome is returned rather than awaited as a coroutine.

        ``validate_guardrails_ai_input`` and ``validate_guardrails_ai_output`` are synchronous,
        so awaiting unconditionally raises TypeError on every request and the envelope reports
        a working rail as a block. ``ActionDispatcher`` has always handled this for LLMRails.
        """
        expected = RailOutcome.block(reason="validator tripped")

        def synchronous_action(**kwargs):
            return expected

        monkeypatch.setattr(CONTENT_SAFETY_ACTION, synchronous_action)

        outcome = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert outcome == expected


class TestModelDependencyValidation:
    """A rail naming a model the config does not declare fails to compile.

    Without this ``llms[model_name]`` fails at request time and the fail-closed envelope
    reports a configuration error as a rail block.

    **The literal case is the one that matters.** ``RailsConfig`` already rejects a ``$model=``
    naming an undeclared type (``check_model_exists_for_input_rails``, ``config.py:988``), so
    that half is defence in depth and unreachable from a real config. It reads the ``$model=``
    suffix, though, which a manifest *literal* binding does not have — so a rail whose model
    is baked into the manifest passes config validation and fails per request. Measured:
    ``llama guard check input`` with only a ``main`` model is accepted by ``RailsConfig``.
    """

    def test_a_configured_surface_param_model_compiles(self, deps):
        """The control: $model= naming a configured model is accepted."""
        assert compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps) is not None

    def test_an_unconfigured_surface_param_model_is_refused(self, deps):
        """$model= naming an undeclared model fails at compile time.

        Defence in depth: ``RailsConfig`` rejects such a config first, so this fires only for
        a caller reaching ``compile_rail`` directly.
        """
        with pytest.raises(RailCompilationError, match="absent_model"):
            compile_rail("content safety check input $model=absent_model", RailDirection.INPUT, deps)

    def test_an_unconfigured_literal_model_is_refused(self, deps):
        """A manifest-baked model name is validated, and nothing else validates it.

        ``llama guard check input`` binds ``model_name="llama_guard"`` as a literal, so the
        flow string names no model and ``RailsConfig``'s check skips the flow entirely. This
        is the real gap the compile-time check closes.
        """
        with pytest.raises(RailCompilationError, match="llama_guard"):
            compile_rail("llama guard check input", RailDirection.INPUT, deps)

    def test_a_configured_literal_model_compiles(self, deps):
        """The same rail compiles once the model is declared, so the refusal is about the model.

        Without this pair, a refusal for any other reason would read as a passing test.
        """
        deps_with_llama_guard = replace(deps, llms={**deps.llms, "llama_guard": FakeLLMModel(responses=["safe"])})

        assert compile_rail("llama guard check input", RailDirection.INPUT, deps_with_llama_guard) is not None


class TestMissingOptionalDependencies:
    """A rail whose declared distribution is absent is refused at compile time.

    The alternative is worse than it sounds: library actions import their optional dependency
    lazily, so the ImportError arrives per request and the fail-closed envelope renders it as
    a guardrail block, with the real cause reaching the log only.

    Installed-ness is injected rather than read from the environment. These cases have to
    assert both halves -- refused when absent, compiled when present -- and only one half is
    observable in any given venv, so reading the real one would leave the other untested and
    make the suite's meaning depend on which extras a developer happens to have.
    """

    @pytest.fixture
    def absent(self, monkeypatch):
        """Every declared distribution is missing."""
        monkeypatch.setattr("nemoguardrails.guardrails.compiled_rail._is_installed", lambda _: False)

    @pytest.fixture
    def installed(self, monkeypatch):
        """Every declared distribution is present."""
        monkeypatch.setattr("nemoguardrails.guardrails.compiled_rail._is_installed", lambda _: True)

    # (rails_config, flow, direction, refused-when-absent). Refused-when-installed is always
    # False, asserted for every row by test_nothing_is_refused_when_the_distribution_is_present.
    CASES = [
        # No remote backend: the declaration is the whole story.
        ({}, "detect sensitive data on input", RailDirection.INPUT, True),
        ({}, "cleanlab trustworthiness", RailDirection.OUTPUT, True),
        # hf_classifier chooses its backend per classifier, through `engine`.
        (_HF_LOCAL_CONFIG, _HF_FLOW, RailDirection.INPUT, True),
        (_HF_REMOTE_CONFIG, _HF_FLOW, RailDirection.INPUT, False),
        ({"hf_classifier": {}}, _HF_FLOW, RailDirection.INPUT, True),
        (_HF_OTHER_CLASSIFIER_CONFIG, _HF_FLOW, RailDirection.INPUT, True),
        ({}, _HF_FLOW, RailDirection.INPUT, True),
        # jailbreak_detection is remote when either endpoint is set, local otherwise.
        ({"jailbreak_detection": {}}, "jailbreak detection model", RailDirection.INPUT, True),
        ({}, "jailbreak detection model", RailDirection.INPUT, True),
        (
            {"jailbreak_detection": {"nim_base_url": "http://nim:8000"}},
            "jailbreak detection model",
            RailDirection.INPUT,
            False,
        ),
        (
            {"jailbreak_detection": {"server_endpoint": "http://jb:1337"}},
            "jailbreak detection model",
            RailDirection.INPUT,
            False,
        ),
        (
            {"jailbreak_detection": {"nim_url": "nim", "nim_port": 8000}},
            "jailbreak detection model",
            RailDirection.INPUT,
            False,
        ),
        (
            {"jailbreak_detection": {"nim_server_endpoint": "classify"}},
            "jailbreak detection model",
            RailDirection.INPUT,
            True,
        ),
        # Declares no optional dependency at all.
        (_REGEX_CONFIG, "regex check input", RailDirection.INPUT, False),
        ({}, "content safety check input $model=content_safety", RailDirection.INPUT, False),
    ]
    IDS = [
        "sdd_no_remote_backend",
        "cleanlab_no_remote_backend",
        "hf_engine_local",
        "hf_engine_vllm",
        "hf_section_empty",
        "hf_classifier_not_configured",
        "hf_section_absent",
        "jailbreak_no_endpoint",
        "jailbreak_section_absent",
        "jailbreak_nim_base_url",
        "jailbreak_server_endpoint",
        "jailbreak_deprecated_nim_url",
        "jailbreak_only_classification_path",
        "regex_declares_nothing",
        "content_safety_declares_nothing",
    ]

    @pytest.mark.parametrize(("rails_config", "flow", "direction", "refused"), CASES, ids=IDS)
    def test_the_configured_backend_decides_when_the_distribution_is_absent(
        self, absent, rails_config, flow, direction, refused
    ):
        """A rail is refused only when the configuration selects a backend needing the package.

        Whether the action accepts ``http_client`` says what the function can receive, not
        which backend will run: an hf_classifier on ``engine: local`` accepts a client,
        ignores it, and still needs ``transformers``.
        """
        deps = _deps_with_rails_config(rails_config)

        if refused:
            with pytest.raises(RailCompilationError, match="which the environment does not have"):
                compile_rail(flow, direction, deps)
        else:
            assert compile_rail(flow, direction, deps) is not None

    @pytest.mark.parametrize(("rails_config", "flow", "direction", "refused"), CASES, ids=IDS)
    def test_nothing_is_refused_when_the_distribution_is_present(
        self, installed, rails_config, flow, direction, refused
    ):
        """With the package installed every configuration compiles, local backends included."""
        assert compile_rail(flow, direction, _deps_with_rails_config(rails_config)) is not None

    def test_a_present_distribution_is_reported_installed(self):
        """The check answers both ways; every other case here injects the answer instead."""
        assert _is_installed("pydantic") is True
        assert _is_installed("no-such-distribution-anywhere") is False

    def test_a_rail_with_no_config_at_all_is_treated_as_local(self):
        """With nothing to read, the backend check assumes in-process and enforces the deps."""
        assert _hf_classifier_runs_locally(None, {}) is True
        assert _jailbreak_detection_runs_locally(None, {}) is True

    def test_the_message_names_the_distributions_and_the_extra(self, absent):
        """The refusal is actionable: which distributions are missing, and what installs them."""
        with pytest.raises(RailCompilationError) as excinfo:
            compile_rail("detect sensitive data on input", RailDirection.INPUT, _deps_with_rails_config({}))

        message = str(excinfo.value)
        assert "presidio-analyzer, presidio-anonymizer, spacy" in message
        assert "install the 'sdd' extra" in message

    def test_only_the_missing_distributions_are_named(self, monkeypatch):
        """A partially installed manifest reports the gap, not everything it declares."""
        monkeypatch.setattr(
            "nemoguardrails.guardrails.compiled_rail._is_installed",
            lambda distribution: distribution != "torch",
        )

        with pytest.raises(RailCompilationError, match=r"needs torch, which") as excinfo:
            compile_rail("jailbreak detection model", RailDirection.INPUT, _deps_with_rails_config({}))

        assert "scikit-learn" not in str(excinfo.value)


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
        """topic_safety_check_input declares events, so it receives the prior turns as history."""
        action = RecordingAction(signature_of=topic_safety_check_input)
        monkeypatch.setattr(TOPIC_SAFETY_ACTION, action)
        messages = [
            {"role": "user", "content": "an earlier question"},
            {"role": "assistant", "content": "an earlier answer"},
            *USER_MESSAGES,
        ]

        await compile_rail(TOPIC_SAFETY_INPUT, RailDirection.INPUT, deps).run(messages)

        assert action.kwargs["events"] == [
            {"type": "UserMessage", "text": "an earlier question"},
            {"type": "StartUtteranceBotAction", "script": "an earlier answer"},
        ]

    @pytest.mark.asyncio
    async def test_events_stop_at_the_checked_turn_for_an_assistant_terminated_transcript(self, deps, monkeypatch):
        """A check() transcript ending in an assistant turn leaves that turn out of the history.

        The action appends the checked turn last, from context, so an assistant turn that
        followed it in the transcript would otherwise be classified ahead of it.
        """
        action = RecordingAction(signature_of=topic_safety_check_input)
        monkeypatch.setattr(TOPIC_SAFETY_ACTION, action)
        messages = [
            {"role": "user", "content": "an earlier question"},
            {"role": "assistant", "content": "an earlier answer"},
            *USER_MESSAGES,
            {"role": "assistant", "content": "the reply being checked"},
        ]

        await compile_rail(TOPIC_SAFETY_INPUT, RailDirection.INPUT, deps).run(messages)

        assert action.kwargs["events"] == [
            {"type": "UserMessage", "text": "an earlier question"},
            {"type": "StartUtteranceBotAction", "script": "an earlier answer"},
        ]
        assert action.kwargs["context"]["user_message"] == "hello there"

    @pytest.mark.asyncio
    async def test_http_client_is_supplied_to_vendor_actions(self, deps, monkeypatch):
        """jailbreak_detection_model declares http_client and receives the one compiled into the rail."""
        action = RecordingAction(signature_of=jailbreak_detection_model)
        monkeypatch.setattr("nemoguardrails.library.jailbreak_detection.actions.jailbreak_detection_model", action)
        mock_client = MagicMock()

        await compile_rail(JAILBREAK_INPUT, RailDirection.INPUT, deps, http_client=mock_client).run(USER_MESSAGES)

        assert action.kwargs["http_client"] is mock_client

    @pytest.mark.asyncio
    async def test_close_calls_http_client_close(self, deps):
        """close() forwards to the rail's HTTP client so the connection pool is released."""
        mock_client = AsyncMock()
        rail = compile_rail(JAILBREAK_INPUT, RailDirection.INPUT, deps, http_client=mock_client)

        await rail.close()

        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_a_no_op_when_no_http_client(self, deps):
        """close() on a rail with no HTTP client (LLM-backed) does not raise."""
        rail = compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps)

        await rail.close()


class TestOutcomePassthrough:
    """The action's RailOutcome is returned unmodified; CompiledRail invents nothing."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expected",
        [
            pytest.param(RailOutcome.allow(metadata={"policy_violations": []}), id="allow_with_metadata"),
            pytest.param(RailOutcome.block(metadata={"policy_violations": ["S1"]}), id="block_with_evidence"),
            pytest.param(RailOutcome.block(reason="policy 4 tripped"), id="block_with_reason"),
            pytest.param(RailOutcome.block(), id="block_without_reason"),
        ],
    )
    async def test_outcome_is_returned_unmodified(self, deps, monkeypatch, expected):
        """Whatever the action returns comes back identical — decision, reason and metadata.

        Equality covers the cases that used to be separate tests: an absent reason stays
        ``None`` rather than being invented, and metadata is neither dropped nor added to.
        """
        monkeypatch.setattr(CONTENT_SAFETY_ACTION, RecordingAction(expected, signature_of=content_safety_check_input))

        outcome = await compile_rail(CONTENT_SAFETY_INPUT, RailDirection.INPUT, deps).run(USER_MESSAGES)

        assert outcome == expected

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

    CURRENT_TURN = {"role": "user", "content": "the turn being checked"}

    @pytest.mark.parametrize(
        ("role", "event"),
        [
            ("user", {"type": "UserMessage", "text": "hi"}),
            ("assistant", {"type": "StartUtteranceBotAction", "script": "hi"}),
            ("system", {"type": "SystemMessage", "content": "hi"}),
        ],
    )
    def test_each_role_maps_to_its_event_shape(self, role, event):
        """Each role maps to the event type and payload key ``to_chat_messages`` reads."""
        assert messages_to_events([{"role": role, "content": "hi"}, self.CURRENT_TURN]) == [event]

    def test_contentless_turn_is_skipped(self):
        """An assistant tool-call turn has no content and is dropped rather than crashing."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            self.CURRENT_TURN,
        ]

        assert messages_to_events(messages) == [{"type": "UserMessage", "text": "hi"}]

    def test_order_is_preserved(self):
        """Turns keep conversation order so history reads correctly."""
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            self.CURRENT_TURN,
        ]

        assert [event["type"] for event in messages_to_events(messages)] == [
            "SystemMessage",
            "UserMessage",
            "StartUtteranceBotAction",
            "UserMessage",
            "StartUtteranceBotAction",
        ]

    def test_the_turn_being_checked_is_left_out(self):
        """A lone user turn yields no history: the action appends it from the context itself."""
        assert messages_to_events([self.CURRENT_TURN]) == []

    def test_only_the_last_user_turn_is_left_out(self):
        """Earlier user turns are history; only the turn being checked is withheld."""
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            self.CURRENT_TURN,
        ]

        texts = [event.get("text") for event in messages_to_events(messages) if event["type"] == "UserMessage"]

        assert texts == ["u1"]

    def test_turns_after_the_checked_turn_are_dropped(self):
        """A transcript ending in an assistant turn yields only the history preceding the checked turn."""
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            self.CURRENT_TURN,
            {"role": "assistant", "content": "a2"},
        ]

        assert messages_to_events(messages) == [
            {"type": "UserMessage", "text": "u1"},
            {"type": "StartUtteranceBotAction", "script": "a1"},
        ]

    def test_a_transcript_with_no_user_turn_is_all_history(self):
        """With no user turn to check, every turn is emitted as history."""
        messages = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a1"},
        ]

        assert messages_to_events(messages) == [
            {"type": "SystemMessage", "content": "s"},
            {"type": "StartUtteranceBotAction", "script": "a1"},
        ]


class TestModelCallCapture:
    """Model calls are captured from a per-rail sink, so attribution cannot leak."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("content_safety_action")
    async def test_a_rail_that_makes_no_model_call_reports_none(self, deps):
        """A vendor rail that never reaches a model produces no captured call."""
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
