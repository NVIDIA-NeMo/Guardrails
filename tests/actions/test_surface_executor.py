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

import pytest

from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.actions.execution import (
    ActionExecutionScope,
    resolve_action_parameters,
)
from nemoguardrails.actions.rail_outcome import RailDecision, RailOutcome
from nemoguardrails.actions.surface_executor import (
    RailSurfaceExecutor,
    resolve_surface_parameters,
)
from nemoguardrails.llm.models.resources import LLMModelResources
from nemoguardrails.llm.taskmanager import LLMTaskManager
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
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.testing import FakeLLMModel


def _scope(
    *,
    context=None,
    main=None,
    specialized_models=None,
) -> ActionExecutionScope:
    config = RailsConfig(models=[])
    return ActionExecutionScope(
        config=config,
        context=context or {},
        events=[{"type": "UserMessage"}],
        llm_task_manager=LLMTaskManager(config),
        model_resources=LLMModelResources(
            main=main,
            specialized_models=specialized_models or {},
            caches={},
        ),
    )


def _catalog(surface: RailSurface) -> RailCatalog:
    manifest = RailManifest(
        name="test",
        spec=RailSpec(
            actions=RailActions(refs=(surface.action,)),
            surfaces=(surface,),
        ),
    )
    return RailCatalog((RailManifestRecord(manifest=manifest, source="test"),))


@pytest.mark.asyncio
async def test_executes_surface_with_manifest_bindings_and_shared_dependencies():
    calls = []

    async def check(
        text,
        mode,
        model_name,
        config,
        context,
        events,
        llm_task_manager,
        llm,
        llms,
        model_caches,
    ):
        calls.append(
            {
                "text": text,
                "mode": mode,
                "model_name": model_name,
                "config": config,
                "context": context,
                "events": events,
                "llm_task_manager": llm_task_manager,
                "llm": llm,
                "llms": llms,
                "model_caches": model_caches,
            }
        )
        return RailOutcome.block(metadata={"source": mode})

    action = ActionRef(name="check", target="tests.actions.test_surface_executor:check")
    surface = RailSurface(
        name="check input",
        direction=RailDirection.INPUT,
        action=action,
        bindings=(
            Binding.context("text", "user_message"),
            Binding.literal("mode", "input"),
            Binding.surface_param("model_name", "model"),
        ),
    )
    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action(check, name=action.name)
    executor = RailSurfaceExecutor(
        catalog=_catalog(surface),
        dispatcher=dispatcher,
    )
    main = FakeLLMModel(responses=[])
    specialized = FakeLLMModel(responses=[])
    scope = _scope(
        context={"user_message": "hello"},
        main=main,
        specialized_models={"safety": specialized},
    )

    outcome = await executor.execute(
        "check input $model=safety",
        direction=RailDirection.INPUT,
        scope=scope,
    )

    assert outcome.decision is RailDecision.BLOCK
    assert outcome.metadata == {"source": "input"}
    assert calls == [
        {
            "text": "hello",
            "mode": "input",
            "model_name": "safety",
            "config": scope.config,
            "context": {"user_message": "hello"},
            "events": [{"type": "UserMessage"}],
            "llm_task_manager": scope.llm_task_manager,
            "llm": main,
            "llms": {"safety": specialized},
            "model_caches": {},
        }
    ]


def test_missing_required_surface_binding_fails_before_dispatch():
    action = ActionRef(name="check", target="tests.actions.test_surface_executor:check")
    surface = RailSurface(
        name="check input",
        direction=RailDirection.INPUT,
        action=action,
        bindings=(Binding.context("text", "user_message"),),
    )

    with pytest.raises(ValueError, match="context value 'user_message'"):
        resolve_surface_parameters(surface, {}, {})


@pytest.mark.asyncio
async def test_unknown_surface_fails_resolution():
    dispatcher = ActionDispatcher(load_all_actions=False)
    executor = RailSurfaceExecutor(
        catalog=RailCatalog(),
        dispatcher=dispatcher,
    )

    with pytest.raises(LookupError, match="No input rail surface"):
        await executor.execute(
            "missing",
            direction=RailDirection.INPUT,
            scope=_scope(),
        )


def test_all_builtin_surfaces_resolve_action_dependencies():
    catalog = default_rail_catalog()
    dispatcher = ActionDispatcher(
        load_all_actions=True,
        rail_catalog=catalog,
    )
    model = FakeLLMModel(responses=[])
    scope = _scope(
        context={
            "user_message": "user",
            "bot_message": "bot",
            "relevant_chunks": "chunks",
        },
        main=model,
        specialized_models={
            "llama_guard": model,
            "patronus_lynx": model,
        },
    )
    resolved = set()

    for key, surface in catalog.surfaces().items():
        surface_parameters = {
            binding.key: "configured"
            for binding in surface.bindings
            if binding.kind == "surface_param" and binding.key is not None
        }
        explicit_parameters = resolve_surface_parameters(
            surface,
            surface_parameters,
            scope.context,
        )
        action = dispatcher.get_action(surface.action.name)
        assert action is not None
        try:
            resolve_action_parameters(
                action,
                explicit_parameters,
                scope,
                action_name=surface.action.name,
            )
        except Exception as error:
            pytest.fail(f"{key}: {error}")
        resolved.add(key)

    assert resolved == set(catalog.surfaces())
