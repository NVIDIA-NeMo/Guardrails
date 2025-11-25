# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.embeddings.providers.fastembed import FastEmbedEmbeddingModel

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")

LIVE_TEST_MODE = os.environ.get("LIVE_TEST")


@pytest.fixture
def app():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "with_kb_openai_embeddings"))

    return LLMRails(config)


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
def test_custom_llm_registration(app):
    assert isinstance(app.llm_generation_actions.flows_index._model, FastEmbedEmbeddingModel)

    assert app.kb.index.embedding_engine == "openai"
    assert app.kb.index.embedding_model == "text-embedding-ada-002"


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
def test_live_query(app):
    result = app.generate(messages=[{"role": "user", "content": "What is NeMo Guardrails?"}])

    assert result == {
        "content": "NeMo Guardrails is an open-source toolkit for easily adding "
        "programmable guardrails to LLM-based conversational systems. "
        'Guardrails (or "rails" for short) are specific ways of '
        "controlling the output of a large language model, such as not "
        "talking about politics, responding in a particular way to "
        "specific user requests, following a predefined dialog path, using "
        "a particular language style, extracting structured data, and "
        "more.",
        "role": "assistant",
    }
