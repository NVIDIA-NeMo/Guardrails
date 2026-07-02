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
from typing import Any, cast

import pytest

from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.rails.llm.config import Document, RailsConfig
from nemoguardrails.rails.llm.startup import knowledge_base
from nemoguardrails.rails.llm.startup.knowledge_base import (
    build_knowledge_base_for_docs,
    init_knowledge_base,
)
from nemoguardrails.rails.llm.types import KnowledgeBaseSurface


class RuntimeWithActionParams:
    def __init__(self):
        self.registered_action_params = {}

    def register_action_param(self, name, value):
        self.registered_action_params[name] = value


@pytest.mark.asyncio
async def test_build_knowledge_base_for_docs_returns_none_without_docs(monkeypatch):
    def fail_if_constructed(*args, **kwargs):
        raise AssertionError("KnowledgeBase should not be constructed without docs.")

    monkeypatch.setattr(knowledge_base, "KnowledgeBase", fail_if_constructed)

    kb = await build_knowledge_base_for_docs(
        config=RailsConfig(models=[]),
        get_embedding_search_provider_instance=lambda _: cast(EmbeddingsIndex, None),
    )

    assert kb is None


@pytest.mark.asyncio
async def test_build_knowledge_base_for_docs_builds_docs_backed_kb(monkeypatch):
    provider = cast(EmbeddingsIndex, object())
    created = {}

    class FakeKnowledgeBase:
        def __init__(self, documents, config, get_embedding_search_provider_instance):
            self.documents = documents
            self.config = config
            self.get_embedding_search_provider_instance = get_embedding_search_provider_instance
            self.init_called = False
            self.build_called = False
            created["kb"] = self

        def init(self):
            self.init_called = True

        async def build(self):
            self.build_called = True

    monkeypatch.setattr(knowledge_base, "KnowledgeBase", FakeKnowledgeBase)
    config = RailsConfig(
        models=[],
        docs=[
            Document(format="md", content="# Alpha\n\nA"),
            Document(format="md", content="# Beta\n\nB"),
        ],
    )

    kb = cast(
        Any,
        await build_knowledge_base_for_docs(
            config=config,
            get_embedding_search_provider_instance=lambda _: provider,
        ),
    )

    assert kb is created["kb"]
    assert kb.documents == ["# Alpha\n\nA", "# Beta\n\nB"]
    assert kb.config is config.knowledge_base
    assert kb.get_embedding_search_provider_instance(None) is provider
    assert kb.init_called is True
    assert kb.build_called is True


def test_init_knowledge_base_uses_thread_path_and_registers_kb(monkeypatch):
    built_kb = object()
    calls = {"thread_started": False, "thread_joined": False}

    async def fake_build_knowledge_base_for_docs(config, get_embedding_search_provider_instance):
        assert rails._kb is None
        assert config is rails.config
        assert get_embedding_search_provider_instance is get_provider_instance
        return built_kb

    class FakeThread:
        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            calls["thread_started"] = True
            self.target(*self.args)

        def join(self):
            calls["thread_joined"] = True

    def get_provider_instance(_=None):
        return None

    rails = SimpleNamespace(
        config=RailsConfig(
            models=[],
            docs=[Document(format="md", content="# Alpha\n\nA")],
        ),
        _kb="existing",
        runtime=RuntimeWithActionParams(),
        _get_embeddings_search_provider_instance=get_provider_instance,
    )
    monkeypatch.setattr(
        knowledge_base,
        "build_knowledge_base_for_docs",
        fake_build_knowledge_base_for_docs,
    )
    monkeypatch.setattr(knowledge_base.threading, "Thread", FakeThread)
    monkeypatch.setattr(knowledge_base, "get_or_create_event_loop", lambda: object())

    init_knowledge_base(cast(KnowledgeBaseSurface, rails))

    assert rails._kb is built_kb
    assert rails.runtime.registered_action_params["kb"] is built_kb
    assert calls == {"thread_started": True, "thread_joined": True}
