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

import json
from unittest.mock import MagicMock

import pytest
from langchain.llms.base import BaseLLM
from pydantic import ValidationError

from nemoguardrails.rails.llm.config import Model, RailsConfig, TaskPrompt
from nemoguardrails.rails.llm.llmrails import LLMRails


def test_task_prompt_valid_content():
    prompt = TaskPrompt(task="example_task", content="This is a valid prompt.")
    assert prompt.task == "example_task"
    assert prompt.content == "This is a valid prompt."
    assert prompt.messages is None


def test_task_prompt_valid_messages():
    prompt = TaskPrompt(task="example_task", messages=["Hello", "How can I help you?"])
    assert prompt.task == "example_task"
    assert prompt.messages == ["Hello", "How can I help you?"]
    assert prompt.content is None


def test_task_prompt_missing_content_and_messages():
    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task")
    assert "One of `content` or `messages` must be provided." in str(excinfo.value)


def test_task_prompt_both_content_and_messages():
    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(
            task="example_task",
            content="This is a prompt.",
            messages=["Hello", "How can I help you?"],
        )
    assert "Only one of `content` or `messages` must be provided." in str(excinfo.value)


def test_task_prompt_models_validation():
    prompt = TaskPrompt(
        task="example_task",
        content="Test prompt",
        models=["openai", "openai/gpt-3.5-turbo"],
    )
    assert prompt.models == ["openai", "openai/gpt-3.5-turbo"]

    prompt = TaskPrompt(task="example_task", content="Test prompt", models=[])
    assert prompt.models == []

    prompt = TaskPrompt(task="example_task", content="Test prompt", models=None)
    assert prompt.models is None


def test_task_prompt_max_length_validation():
    prompt = TaskPrompt(task="example_task", content="Test prompt")
    assert prompt.max_length == 16000

    prompt = TaskPrompt(task="example_task", content="Test prompt", max_length=1000)
    assert prompt.max_length == 1000

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", max_length=0)
    assert "Input should be greater than or equal to 1" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", max_length=-1)
    assert "Input should be greater than or equal to 1" in str(excinfo.value)


def test_task_prompt_mode_validation():
    prompt = TaskPrompt(task="example_task", content="Test prompt")
    # default mode is "standard"
    assert prompt.mode == "standard"

    prompt = TaskPrompt(task="example_task", content="Test prompt", mode="chat")
    assert prompt.mode == "chat"

    prompt = TaskPrompt(task="example_task", content="Test prompt", mode=None)
    assert prompt.mode is None


def test_task_prompt_stop_tokens_validation():
    prompt = TaskPrompt(
        task="example_task", content="Test prompt", stop=["\n", "Human:", "Assistant:"]
    )
    assert prompt.stop == ["\n", "Human:", "Assistant:"]

    prompt = TaskPrompt(task="example_task", content="Test prompt", stop=[])
    assert prompt.stop == []

    prompt = TaskPrompt(task="example_task", content="Test prompt", stop=None)
    assert prompt.stop is None

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", stop=[1, 2, 3])
    assert "Input should be a valid string" in str(excinfo.value)


def test_task_prompt_max_tokens_validation():
    prompt = TaskPrompt(task="example_task", content="Test prompt")
    assert prompt.max_tokens is None

    prompt = TaskPrompt(task="example_task", content="Test prompt", max_tokens=1000)
    assert prompt.max_tokens == 1000

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", max_tokens=0)
    assert "Input should be greater than or equal to 1" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", max_tokens=-1)
    assert "Input should be greater than or equal to 1" in str(excinfo.value)


def test_rails_config_addition():
    """Tests that adding two RailsConfig objects merges both into a single RailsConfig."""
    config1 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        config_path="test_config.yml",
    )
    config2 = RailsConfig(
        models=[Model(type="secondary", engine="anthropic", model="claude-3")],
        config_path="test_config2.yml",
    )

    result = config1 + config2

    assert isinstance(result, RailsConfig)
    assert len(result.models) == 2
    assert result.config_path == "test_config.yml,test_config2.yml"


def test_rails_config_model_conflicts():
    """Tests that adding two RailsConfig objects with conflicting models raises an error."""
    config1 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        config_path="config1.yml",
    )

    # Different engine for same model type
    config2 = RailsConfig(
        models=[Model(type="main", engine="nim", model="gpt-3.5-turbo")],
        config_path="config2.yml",
    )
    with pytest.raises(
        ValueError,
        match="Both config files should have the same engine for the same model type",
    ):
        config1 + config2

    # Different model for same model type
    config3 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-4")],
        config_path="config3.yml",
    )
    with pytest.raises(
        ValueError,
        match="Both config files should have the same model for the same model type",
    ):
        config1 + config3


def test_rails_config_actions_server_url_conflicts():
    """Tests that adding two RailsConfig objects with different values for `actions_server_url` raises an error."""
    config1 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        actions_server_url="http://localhost:8000",
    )

    config2 = RailsConfig(
        models=[Model(type="secondary", engine="anthropic", model="claude-3")],
        actions_server_url="http://localhost:9000",
    )

    with pytest.raises(
        ValueError, match="Both config files should have the same actions_server_url"
    ):
        config1 + config2


def test_rails_config_simple_field_overwriting():
    """Tests that fields from the second config overwrite fields from the first config."""
    config1 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        streaming=False,
        lowest_temperature=0.1,
        colang_version="1.0",
    )

    config2 = RailsConfig(
        models=[Model(type="secondary", engine="anthropic", model="claude-3")],
        streaming=True,
        lowest_temperature=0.5,
        colang_version="2.x",
    )

    result = config1 + config2

    assert result.streaming is True
    assert result.lowest_temperature == 0.5
    assert result.colang_version == "2.x"


def test_rails_config_nested_dictionary_merging():
    """Tests nested dictionaries are merged correctly."""
    config1 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        rails={
            "input": {"flows": ["flow1"], "parallel": False},
            "output": {"flows": ["flow2"]},
        },
        knowledge_base={
            "folder": "kb1",
            "embedding_search_provider": {"name": "provider1"},
        },
        custom_data={"setting1": "value1", "nested": {"key1": "val1"}},
    )

    config2 = RailsConfig(
        models=[Model(type="secondary", engine="anthropic", model="claude-3")],
        rails={
            "input": {"flows": ["flow3"], "parallel": True},
            "retrieval": {"flows": ["flow4"]},
        },
        knowledge_base={
            "folder": "kb2",
            "embedding_search_provider": {"name": "provider2"},
        },
        custom_data={"setting2": "value2", "nested": {"key2": "val2"}},
    )

    result = config1 + config2

    assert result.rails.input.flows == ["flow3", "flow1"]
    assert result.rails.input.parallel is True
    assert result.rails.output.flows == ["flow2"]
    assert result.rails.retrieval.flows == ["flow4"]

    assert result.knowledge_base.folder == "kb2"
    assert result.knowledge_base.embedding_search_provider.name == "provider2"

    assert result.custom_data["setting1"] == "value1"
    assert result.custom_data["setting2"] == "value2"
    assert result.custom_data["nested"]["key1"] == "val1"
    assert result.custom_data["nested"]["key2"] == "val2"


def test_rails_config_none_prompts():
    """Test that configs with None prompts can be added without errors."""
    config1 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        prompts=None,
        rails={"input": {"flows": ["self_check_input"]}},
    )
    config2 = RailsConfig(
        models=[Model(type="secondary", engine="anthropic", model="claude-3")],
        prompts=[],
    )

    result = config1 + config2
    assert result is not None
    assert result.prompts is not None


def test_rails_config_none_config_path():
    """Test that configs with None config_path can be added."""
    config1 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        config_path=None,
    )
    config2 = RailsConfig(
        models=[Model(type="secondary", engine="anthropic", model="claude-3")],
        config_path="config2.yml",
    )

    result = config1 + config2
    # should not have leading comma after fix
    assert result.config_path == "config2.yml"

    config3 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        config_path=None,
    )
    config4 = RailsConfig(
        models=[Model(type="secondary", engine="anthropic", model="claude-3")],
        config_path=None,
    )

    result2 = config3 + config4
    assert result2.config_path == ""


def test_llm_rails_configure_streaming_with_attr():
    """Check LLM has the streaming attribute set if RailsConfig has it"""

    mock_llm = MagicMock(spec=BaseLLM)
    config = RailsConfig(
        models=[],
        streaming=True,
    )

    rails = LLMRails(config, llm=mock_llm)
    setattr(mock_llm, "streaming", None)
    rails._configure_main_llm_streaming(llm=mock_llm)

    assert mock_llm.streaming


def test_llm_rails_configure_streaming_without_attr(caplog):
    """Check LLM has the streaming attribute set if RailsConfig has it"""

    mock_llm = MagicMock(spec=BaseLLM)
    config = RailsConfig(
        models=[],
        streaming=True,
    )

    rails = LLMRails(config, llm=mock_llm)
    rails._configure_main_llm_streaming(mock_llm)

    assert caplog.messages[-1] == "Provided main LLM does not support streaming."


def test_rails_config_streaming_supported_no_output_flows():
    """Check `streaming_supported` property doesn't depend on RailsConfig.streaming with no output flows"""

    config = RailsConfig(
        models=[],
        streaming=False,
    )
    assert config.streaming_supported


def test_rails_config_flows_streaming_supported_true():
    """Create RailsConfig and check the `streaming_supported Check LLM has the streaming attribute set if RailsConfig has it"""

    rails = {
        "output": {
            "flows": ["content_safety_check_output"],
            "streaming": {"enabled": True},
        }
    }
    prompts = [{"task": "content safety check output", "content": "..."}]
    rails_config = RailsConfig.model_validate(
        {"models": [], "rails": rails, "prompts": prompts}
    )
    assert rails_config.streaming_supported


def test_rails_config_flows_streaming_supported_false():
    """Create RailsConfig and check the `streaming_supported Check LLM has the streaming attribute set if RailsConfig has it"""

    rails = {
        "output": {
            "flows": ["content_safety_check_output"],
            "streaming": {"enabled": False},
        }
    }
    prompts = [{"task": "content safety check output", "content": "..."}]
    rails_config = RailsConfig.model_validate(
        {"models": [], "rails": rails, "prompts": prompts}
    )
    assert not rails_config.streaming_supported
