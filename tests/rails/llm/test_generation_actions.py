# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from types import SimpleNamespace

from nemoguardrails.rails.llm.startup import generation_actions
from nemoguardrails.rails.llm.startup.generation_actions import (
    generation_actions_class_for_colang_version,
    register_llm_generation_actions,
)


class RuntimeWithActionRegistration:
    def __init__(self):
        self.llm_task_manager = object()
        self.registered_actions = None
        self.override = None

    def register_actions(self, actions, override=True):
        self.registered_actions = actions
        self.override = override


class FakeGenerationActions:
    def __init__(
        self,
        config,
        llm,
        llm_task_manager,
        get_embedding_search_provider_instance,
        verbose,
    ):
        self.config = config
        self.llm = llm
        self.llm_task_manager = llm_task_manager
        self.get_embedding_search_provider_instance = get_embedding_search_provider_instance
        self.verbose = verbose


class FakeGenerationActionsV2(FakeGenerationActions):
    pass


class FakeRails:
    def __init__(self, config, llm, runtime, get_embedding_search_provider_instance):
        self.config = config
        self.llm = llm
        self.runtime = runtime
        self._get_embeddings_search_provider_instance = get_embedding_search_provider_instance
        self._llm_generation_actions = None


def test_generation_actions_class_for_colang_version_uses_v1_actions(monkeypatch):
    monkeypatch.setattr(generation_actions, "LLMGenerationActions", FakeGenerationActions)

    actions_class = generation_actions_class_for_colang_version("1.0")

    assert actions_class is FakeGenerationActions


def test_generation_actions_class_for_colang_version_uses_v2_actions(monkeypatch):
    monkeypatch.setattr(generation_actions, "LLMGenerationActionsV2dotx", FakeGenerationActionsV2)

    actions_class = generation_actions_class_for_colang_version("2.x")

    assert actions_class is FakeGenerationActionsV2


def test_register_llm_generation_actions_registers_without_overriding(monkeypatch):
    monkeypatch.setattr(generation_actions, "LLMGenerationActions", FakeGenerationActions)
    config = SimpleNamespace(colang_version="1.0")
    llm = object()
    runtime = RuntimeWithActionRegistration()

    def get_embedding_search_provider_instance(esp_config=None):
        del esp_config
        return None

    rails = FakeRails(
        config=config,
        llm=llm,
        runtime=runtime,
        get_embedding_search_provider_instance=get_embedding_search_provider_instance,
    )

    register_llm_generation_actions(rails, verbose=True)

    assert isinstance(rails._llm_generation_actions, FakeGenerationActions)
    assert rails._llm_generation_actions.config is config
    assert rails._llm_generation_actions.llm is llm
    assert rails._llm_generation_actions.llm_task_manager is runtime.llm_task_manager
    assert (
        rails._llm_generation_actions.get_embedding_search_provider_instance is get_embedding_search_provider_instance
    )
    assert rails._llm_generation_actions.verbose is True
    assert runtime.registered_actions is rails._llm_generation_actions
    assert runtime.override is False
