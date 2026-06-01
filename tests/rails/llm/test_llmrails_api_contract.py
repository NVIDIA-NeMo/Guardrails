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

import pickle

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.context import explain_info_var
from nemoguardrails.logging.explain import ExplainInfo
from tests.utils import FakeLLMModel


def _config() -> RailsConfig:
    return RailsConfig.from_content(config={"models": []})


def test_constructor_keeps_public_compatibility_state_visible():
    config = _config()
    llm = FakeLLMModel(responses=[])

    rails = LLMRails(config=config, llm=llm)

    assert rails.config is config
    assert rails.llm is llm
    assert rails.runtime is not None
    assert rails.llm_generation_actions is not None
    assert rails.embedding_search.providers == {}
    assert rails.events_history_cache == {}
    assert rails.explain_info is None


def test_update_llm_keeps_runtime_generation_actions_and_public_attr_in_sync():
    rails = LLMRails(config=_config(), llm=FakeLLMModel(responses=[]))
    new_llm = FakeLLMModel(responses=["updated"])

    rails.update_llm(new_llm)

    assert rails.llm is new_llm
    assert rails.llm_generation_actions.llm is new_llm
    assert rails.runtime.registered_action_params["llm"] is new_llm


@pytest.mark.asyncio
async def test_sync_wrappers_raise_when_called_from_async_loop():
    rails = LLMRails(config=_config(), llm=FakeLLMModel(responses=[]))

    with pytest.raises(RuntimeError, match="sync `generate` inside async code"):
        rails.generate(prompt="hi")

    with pytest.raises(RuntimeError, match="sync `generate_events` inside async code"):
        rails.generate_events([])

    with pytest.raises(RuntimeError, match="sync `generate_events` inside async code"):
        rails.process_events([])

    with pytest.raises(RuntimeError, match="sync `check` inside async code"):
        rails.check([{"role": "user", "content": "hi"}])


def test_getstate_serializes_config_only():
    rails = LLMRails(config=_config(), llm=FakeLLMModel(responses=[]))
    rails.events_history_cache["cached"] = [{"type": "CachedEvent"}]
    rails.explain_info = ExplainInfo()
    rails.register_action_param("custom_param", object())

    state = rails.__getstate__()

    assert state == {"config": rails.config}


def test_explain_prefers_active_request_context_over_latest_instance_info():
    rails = LLMRails(config=_config(), llm=FakeLLMModel(responses=[]))
    latest_explain_info = ExplainInfo()
    request_explain_info = ExplainInfo()
    rails.explain_info = latest_explain_info
    token = explain_info_var.set(request_explain_info)
    try:
        assert rails.explain() is request_explain_info
        assert rails.explain_info is request_explain_info
    finally:
        explain_info_var.reset(token)


def test_pickle_round_trip_reinitializes_from_config_without_runtime_state():
    rails = LLMRails(config=_config(), llm=FakeLLMModel(responses=[]))
    rails.events_history_cache["cached"] = [{"type": "CachedEvent"}]
    rails.register_action_param("custom_param", "value")

    restored = pickle.loads(pickle.dumps(rails))

    assert isinstance(restored, LLMRails)
    assert restored.config == rails.config
    assert restored.llm is None
    assert restored.events_history_cache == {}
    assert "custom_param" not in restored.runtime.registered_action_params
