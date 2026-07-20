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

from typing import Any

import pytest

from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.actions.rail_executor import RailInvocation, execute_rail, resolve_rail_invocation
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    RailActions,
    RailCatalog,
    RailDirection,
    RailManifest,
    RailManifestRecord,
    RailSpec,
    RailSurface,
    default_rail_catalog,
)

ACTION = ActionRef(name="surface_check", target="tests.test_rail_executor:_surface_check")
SURFACE = RailSurface(
    name="check input",
    direction=RailDirection.INPUT,
    action=ACTION,
    bindings=(
        Binding.surface_param("model_name", "model"),
        Binding.surface_param("variant", "variant", required=False),
        Binding.context("text", "user_message"),
        Binding.literal("mode", "input"),
    ),
)
CATALOG = RailCatalog(
    (
        RailManifestRecord(
            manifest=RailManifest(
                name="test",
                spec=RailSpec(actions=RailActions(refs=(ACTION,)), surfaces=(SURFACE,)),
            ),
            source="tests.test_rail_executor",
        ),
    )
)


def _surface_check() -> RailOutcome:
    return RailOutcome.allow()


def test_resolve_rail_invocation_applies_declared_bindings():
    invocation = resolve_rail_invocation(
        CATALOG,
        RailDirection.INPUT,
        "check input $model=guard $variant=strict",
        context={"user_message": "hello"},
    )

    assert invocation.surface is SURFACE
    assert invocation.action_params == {
        "model_name": "guard",
        "variant": "strict",
        "text": "hello",
        "mode": "input",
    }


def test_resolve_rail_invocation_omits_missing_optional_binding():
    invocation = resolve_rail_invocation(
        CATALOG,
        "input",
        "check input $model=guard",
        context={"user_message": "hello"},
    )

    assert "variant" not in invocation.action_params


@pytest.mark.parametrize(
    ("configured_surface", "context", "match"),
    [
        ("check input", {"user_message": "hello"}, "requires surface_param value 'model'"),
        ("check input $model=guard", {}, "requires context value 'user_message'"),
        ("check input $model=guard $unknown=x", {"user_message": "hello"}, "unsupported parameters"),
    ],
)
def test_resolve_rail_invocation_rejects_invalid_bindings(configured_surface, context, match):
    with pytest.raises(ValueError, match=match):
        resolve_rail_invocation(CATALOG, "input", configured_surface, context=context)


def test_resolve_rail_invocation_rejects_wrong_direction():
    with pytest.raises(ValueError, match="not declared for direction 'output'"):
        resolve_rail_invocation(CATALOG, "output", "check input")


def test_builtin_surfaces_resolve_every_declared_binding():
    catalog = default_rail_catalog()

    for surface in catalog.surfaces().values():
        parameter_names = [
            binding.key for binding in surface.bindings if binding.kind == "surface_param" and binding.key is not None
        ]
        context = {
            binding.key: f"{binding.key}-value"
            for binding in surface.bindings
            if binding.kind == "context" and binding.key is not None
        }
        configured = surface.name + "".join(f" ${name}={name}-value" for name in parameter_names)

        invocation = resolve_rail_invocation(catalog, surface.direction, configured, context=context)

        assert set(invocation.action_params) == {binding.action_param for binding in surface.bindings}


@pytest.mark.asyncio
async def test_execute_rail_injects_backend_values_and_returns_verdict():
    captured: dict[str, Any] = {}

    async def action(
        model_name,
        text,
        mode,
        context,
        events,
        config,
        llm_task_manager,
        dependency,
        variant=None,
    ) -> RailOutcome:
        captured.update(
            {
                "model_name": model_name,
                "text": text,
                "mode": mode,
                "context": context,
                "events": events,
                "config": config,
                "llm_task_manager": llm_task_manager,
                "dependency": dependency,
                "variant": variant,
            }
        )
        return RailOutcome.block(reason="blocked")

    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action(action, ACTION.name)
    invocation = resolve_rail_invocation(
        CATALOG,
        "input",
        "check input $model=guard",
        context={"user_message": "hello"},
        events=[{"type": "UserMessage", "text": "earlier"}],
        config="config",
        llm_task_manager="task-manager",
        registered_action_params={"dependency": "dependency"},
    )

    outcome = await execute_rail(invocation, dispatcher)

    assert outcome.is_blocked
    assert captured == {
        "model_name": "guard",
        "text": "hello",
        "mode": "input",
        "context": {"user_message": "hello"},
        "events": [{"type": "UserMessage", "text": "earlier"}],
        "config": "config",
        "llm_task_manager": "task-manager",
        "dependency": "dependency",
        "variant": None,
    }


@pytest.mark.asyncio
async def test_execute_rail_injects_context_named_parameters():
    async def action(__context__tenant) -> RailOutcome:
        return RailOutcome.allow(metadata={"tenant": __context__tenant})

    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action(action, ACTION.name)
    invocation = RailInvocation(surface=SURFACE, context={"tenant": "acme"})

    outcome = await execute_rail(invocation, dispatcher)

    assert outcome.metadata == {"tenant": "acme"}


@pytest.mark.asyncio
async def test_execute_rail_rejects_non_outcome_returns():
    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action(lambda: ActionResult(return_value=RailOutcome.allow()), ACTION.name)

    with pytest.raises(TypeError, match="got ActionResult"):
        await execute_rail(RailInvocation(surface=SURFACE), dispatcher)


@pytest.mark.asyncio
async def test_execute_rail_validates_declared_transform_target():
    async def action() -> RailOutcome:
        return RailOutcome.transform([(TransformTarget.USER_MESSAGE, "rewritten")])

    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action(action, ACTION.name)
    transform_surface = SURFACE.model_copy(update={"transform_target": TransformTarget.USER_MESSAGE})

    outcome = await execute_rail(RailInvocation(surface=transform_surface), dispatcher)

    assert outcome.transform_text == {"user_message": "rewritten"}


@pytest.mark.asyncio
async def test_execute_rail_rejects_undeclared_transform_target():
    async def action() -> RailOutcome:
        return RailOutcome.transform([(TransformTarget.USER_MESSAGE, "rewritten")])

    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action(action, ACTION.name)

    with pytest.raises(ValueError, match="declared transform target None"):
        await execute_rail(RailInvocation(surface=SURFACE), dispatcher)


class _FailedDispatcher(ActionDispatcher):
    def __init__(self):
        super().__init__(load_all_actions=False)
        self.register_action(_surface_check, ACTION.name)

    async def execute_action(self, action_name, params):
        return None, "failed"


@pytest.mark.asyncio
async def test_execute_rail_raises_on_dispatch_failure():
    with pytest.raises(RuntimeError, match="failed with status 'failed'"):
        await execute_rail(RailInvocation(surface=SURFACE), _FailedDispatcher())


@pytest.mark.asyncio
async def test_execute_rail_rejects_unregistered_action():
    with pytest.raises(ValueError, match="not registered"):
        await execute_rail(RailInvocation(surface=SURFACE), ActionDispatcher(load_all_actions=False))
