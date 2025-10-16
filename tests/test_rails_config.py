# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.llm.prompts import TaskPrompt
from nemoguardrails.rails.llm.config import (
    Model,
    RailsConfig,
    _get_flow_model,
    _validate_rail_prompts,
)

TEST_API_KEY_NAME = "DUMMY_OPENAI_API_KEY"
TEST_API_KEY_VALUE = "sk-svcacct-abcdefGHIJKlmnoPQRSTuvXYZ1234567890"


@pytest.fixture(
    params=[
        [
            TaskPrompt(task="self_check_input", output_parser=None, content="..."),
            TaskPrompt(task="self_check_facts", output_parser="parser1", content="..."),
            TaskPrompt(
                task="self_check_output", output_parser="parser2", content="..."
            ),
        ],
        [
            {"task": "self_check_input", "output_parser": None},
            {"task": "self_check_facts", "output_parser": "parser1"},
            {"task": "self_check_output", "output_parser": "parser2"},
        ],
    ]
)
def prompts(request):
    return request.param


def test_check_output_parser_exists(caplog, prompts):
    caplog.set_level(logging.INFO)
    values = {"prompts": prompts}

    result = RailsConfig.check_output_parser_exists(values)

    assert result == values
    assert (
        "Deprecation Warning: Output parser is not registered for the task."
        in caplog.text
    )
    assert "self_check_input" in caplog.text


def test_check_prompt_exist_for_self_check_rails():
    """Test that prompts are correctly validated for self-check rails."""

    values = {
        "rails": {
            "input": {"flows": ["self check input"]},
            "output": {"flows": ["self check facts", "self check output"]},
        },
        "prompts": [
            {"task": "self_check_input", "content": "..."},
            {"task": "self_check_facts", "content": "..."},
            {"task": "self_check_output", "content": "..."},
        ],
    }
    result = RailsConfig.check_prompt_exist_for_self_check_rails(values)
    assert result == values

    # missing prompt is an invalid case
    values = {
        "rails": {
            "input": {"flows": ["self check input"]},
            "output": {"flows": ["self check facts", "self check output"]},
        },
        "prompts": [
            {"task": "self_check_input", "content": "..."},
            {"task": "self_check_facts", "content": "..."},
            # missings self_check_output prompt
        ],
    }
    with pytest.raises(
        ValueError, match="You must provide a `self_check_output` prompt template"
    ):
        RailsConfig.check_prompt_exist_for_self_check_rails(values)


def test_fill_in_default_values_for_v2_x():
    """Test that default values are correctly filled in for v2.x."""

    values = {"instructions": [], "sample_conversation": None, "colang_version": "2.x"}
    result = RailsConfig.fill_in_default_values_for_v2_x(values)
    assert "instructions" in result
    assert len(result["instructions"]) > 0
    assert "sample_conversation" in result
    assert result["sample_conversation"] is not None


def test_rails_config_from_path():
    """Test loading RailsConfig from path."""

    config_path = os.path.join(os.path.dirname(__file__), "test_configs", "general")
    config = RailsConfig.from_path(config_path)
    assert config is not None
    assert len(config.instructions) > 0
    assert config.sample_conversation is not None


def test_rails_config_from_path_yml_extension():
    """Test loading RailsConfig when the config directory ends with a .yml suffix.

    Ensures a directory mistakenly named with a YAML extension is treated as a directory,
    not a file, and its internal YAML config is loaded properly.
    """

    with tempfile.TemporaryDirectory(suffix=".yml") as temp_dir:
        temp_path = Path(temp_dir)

        minimal_yaml = (
            "models: []\n"
            "instructions:\n"
            "  - type: general\n"
            "    content: Test instruction\n"
            "sample_conversation: Test conversation\n"
        )

        # place a config file inside the directory-with-.yml suffix
        (temp_path / "config.yml").write_text(minimal_yaml)

        config = RailsConfig.from_path(str(temp_path))
        assert config is not None
        assert len(config.instructions) > 0
        assert config.sample_conversation is not None


def test_rails_config_parse_obj():
    """Test parsing RailsConfig from object."""

    config_obj = {
        "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"}],
        "instructions": [{"type": "general", "content": "Test instruction"}],
        "sample_conversation": "Test conversation",
        "flows": [
            {
                "id": "test_flow",
                "elements": [
                    {"type": "user_say", "content": "Hello"},
                    {"type": "bot_say", "content": "Hi there!"},
                ],
            }
        ],
    }
    config = RailsConfig.model_validate(config_obj)
    assert config is not None
    assert len(config.instructions) == 1
    assert config.sample_conversation == "Test conversation"
    assert len(config.flows) == 1
    assert config.flows[0]["id"] == "test_flow"


def test_model_api_key_optional():
    """Check if we don't set an `api_key_env_var` the Model can still be created"""
    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="openai",
                model="gpt-3.5-turbo-instruct",
                api_key_env_var=None,
            )
        ]
    )
    assert config.models[0].api_key_env_var is None


def test_model_api_key_var_not_set():
    """Check if we reference an invalid env key we throw an error"""
    with pytest.raises(
        ValueError,
        match=f"Model API Key environment variable '{TEST_API_KEY_NAME}' not set.",
    ):
        _ = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="openai",
                    model="gpt-3.5-turbo-instruct",
                    api_key_env_var=TEST_API_KEY_NAME,
                )
            ]
        )


@mock.patch.dict(os.environ, {TEST_API_KEY_NAME: ""})
def test_model_api_key_var_empty_string():
    """Check if we reference a valid env var with empty string as value we throw an error"""
    with pytest.raises(
        ValueError,
        match=f"Model API Key environment variable '{TEST_API_KEY_NAME}' not set.",
    ):
        _ = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="openai",
                    model="gpt-3.5-turbo-instruct",
                    api_key_env_var=TEST_API_KEY_NAME,
                )
            ]
        )


@mock.patch.dict(os.environ, {TEST_API_KEY_NAME: TEST_API_KEY_VALUE})
def test_model_api_key_value_valid_string():
    """Check if we reference a valid api_key_env_var we can create the Model"""

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="openai",
                model="gpt-3.5-turbo-instruct",
                api_key_env_var=TEST_API_KEY_NAME,
            )
        ]
    )
    assert config.models[0].api_key_env_var == TEST_API_KEY_NAME


@mock.patch.dict(
    os.environ,
    {
        TEST_API_KEY_NAME: TEST_API_KEY_VALUE,
        "DUMMY_NVIDIA_API_KEY": "nvapi-abcdef12345",
    },
)
def test_model_api_key_value_multiple_strings():
    """Check if we reference a valid api_key_env_var we can create the Model"""

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="openai",
                model="gpt-3.5-turbo-instruct",
                api_key_env_var=TEST_API_KEY_NAME,
            ),
            Model(
                type="content_safety",
                engine="nim",
                model="nvidia/llama-3.1-nemoguard-8b-content-safety",
                api_key_env_var="DUMMY_NVIDIA_API_KEY",
            ),
        ]
    )
    assert config.models[0].api_key_env_var == TEST_API_KEY_NAME
    assert config.models[1].api_key_env_var == "DUMMY_NVIDIA_API_KEY"


@mock.patch.dict(os.environ, {TEST_API_KEY_NAME: TEST_API_KEY_VALUE})
def test_model_api_key_value_multiple_strings_one_missing():
    """Check if we have multiple models and one references an invalid api_key_env_var we throw error"""
    with pytest.raises(
        ValueError,
        match=f"Model API Key environment variable 'DUMMY_NVIDIA_API_KEY' not set.",
    ):
        _ = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="openai",
                    model="gpt-3.5-turbo-instruct",
                    api_key_env_var=TEST_API_KEY_NAME,
                ),
                Model(
                    type="content_safety",
                    engine="nim",
                    model="nvidia/llama-3.1-nemoguard-8b-content-safety",
                    api_key_env_var="DUMMY_NVIDIA_API_KEY",
                ),
            ]
        )


@mock.patch.dict(
    os.environ, {TEST_API_KEY_NAME: TEST_API_KEY_VALUE, "DUMMY_NVIDIA_API_KEY": ""}
)
def test_model_api_key_value_multiple_strings_one_empty():
    """Check if we have multiple models and one references an invalid api_key_env_var we throw error"""
    with pytest.raises(
        ValueError,
        match=f"Model API Key environment variable 'DUMMY_NVIDIA_API_KEY' not set.",
    ):
        _ = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="openai",
                    model="gpt-3.5-turbo-instruct",
                    api_key_env_var=TEST_API_KEY_NAME,
                ),
                Model(
                    type="content_safety",
                    engine="nim",
                    model="nvidia/llama-3.1-nemoguard-8b-content-safety",
                    api_key_env_var="DUMMY_NVIDIA_API_KEY",
                ),
            ]
        )


class TestConfigHelpers:
    def test_get_flow_model_flow_only(self):
        """Check we return None if the flow doesn't have a model definition"""
        assert _get_flow_model("self check output") is None

    def test_get_flow_model_flow_and_model(self):
        """Check we return None if the flow doesn't have a model definition"""
        assert (
            _get_flow_model("content safety check input $model=content_safety")
            == "content_safety"
        )

    def test_validate_rail_prompts(self):
        """Check we don't raise ValueError if there's a matching prompt for a rail"""

        _validate_rail_prompts(
            ["content safety check input $model=content_safety"],
            ["content_safety_check_input $model=content_safety"],
            "content safety check input",
        )

    def test_validate_rail_prompts_wrong_flow_id_raises(self):
        """Check we raise a ValueError if we have wrong flow_id but correct model"""

        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_input \$model=content_safety` prompt template.",
        ):
            _validate_rail_prompts(
                ["content safety check input $model=content_safety"],
                ["topic_safety_check_input $model=content_safety"],
                "content safety check input",
            )

    def test_validate_rail_prompts_wrong_model_raises(self):
        """Check we don't raise ValueError if there's a matching prompt for a rail"""

        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_input \$model=content_safety` prompt template.",
        ):
            _validate_rail_prompts(
                ["content safety check input $model=content_safety"],
                ["content_safety_check_input $model=local_content_safety"],
                "content safety check input",
            )

    def test_validate_rail_prompts_no_prompt_raises(self):
        """Check we don't raise ValueError if there's a matching prompt for a rail"""

        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_input \$model=content_safety` prompt template.",
        ):
            _validate_rail_prompts(
                ["content safety check input $model=content_safety"],
                [],
                "content safety check input",
            )


class TestContentSafetyConfig:
    """Tests for content-safety config validation"""

    def test_content_safety_input_missing_prompt_raises(self):
        """Check Content Safety output rail raises ValueError if we don't have a prompt"""
        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_input \$model=content_safety` prompt template.",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety
                    """
            )

    def test_content_safety_output_missing_prompt_raises(self):
        """Check Content Safety output rail raises ValueError if we don't have a prompt"""
        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_output \$model=content_safety` prompt template.",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                rails:
                  output:
                    flows:
                      - content safety check output $model=content_safety

                """,
            )

    def test_input_content_safety_has_model(self):
        """Check we create RailsConfig with input content-safety model specified"""

        config = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: content_safety
                engine: nim
                model: nvidia/llama-3.1-nemoguard-8b-content-safety

            rails:
              input:
                flows:
                  - content safety check input $model=content_safety

            prompts:
              - task: content_safety_check_input $model=content_safety
                content: Check content safety
            """,
        )

        # Check a few fields to make sure we created the config correctly
        assert config.models[0].type == "content_safety"
        assert (
            config.rails.input.flows[0]
            == "content safety check input $model=content_safety"
        )

    def test_output_content_safety_has_model(self):
        """Check we create RailsConfig with output content-safety model specified"""

        config = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: content_safety
                engine: nim
                model: nvidia/llama-3.1-nemoguard-8b-content-safety

            rails:
              output:
                flows:
                  - content safety check output $model=content_safety

            prompts:
              - task: content_safety_check_output $model=content_safety
                content: Check content safety
            """,
        )

        # Check a few fields to make sure we created config correctly
        assert config.models[0].type == "content_safety"
        assert (
            config.rails.output.flows[0]
            == "content safety check output $model=content_safety"
        )

    def test_input_output_content_safety_has_model(self):
        """Check we create RailsConfig with output content-safety model specified"""

        config = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: content_safety
                engine: nim
                model: nvidia/llama-3.1-nemoguard-8b-content-safety

            rails:
              input:
                flows:
                  - content safety check input $model=content_safety

              output:
                flows:
                  - content safety check output $model=content_safety

            prompts:
              - task: content_safety_check_output $model=content_safety
                content: Check content safety
              - task: content_safety_check_input $model=content_safety
                content: Check content safety
            """,
        )

        # Check a few fields to make sure we created config correctly
        assert config.models[0].type == "content_safety"
        assert (
            config.rails.input.flows[0]
            == "content safety check input $model=content_safety"
        )
        assert (
            config.rails.output.flows[0]
            == "content safety check output $model=content_safety"
        )

    def test_input_content_safety_no_model_raises(self):
        """Check we raise ValueError when creating an input content safety rail with no model"""

        with pytest.raises(
            ValueError,
            match="No `content_safety` model provided for input flow `content safety check input`",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: main
                    engine: openai
                    model: gpt-4o

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety

                prompts:
                  - task: content_safety_check_input $model=content_safety
                    content: Check content safety
                    """,
            )

    def test_input_content_safety_wrong_model_raises(self):
        """Check we raise ValueError when creating an input content safety rail with no model"""

        with pytest.raises(
            ValueError,
            match="No `content_safety` model provided for input flow `content safety check input",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: local_content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety

                prompts:
                  - task: content_safety_check_input $model=content_safety
                    content: Check content safety
                    """,
            )

    def test_output_content_safety_no_model_raises(self):
        """Check we raise ValueError when creating an output content safety rail with no model"""

        with pytest.raises(
            ValueError,
            match="No `content_safety` model provided for output flow `content safety check output`",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: main
                    engine: openai
                    model: gpt-4o

                rails:
                  output:
                    flows:
                      - content safety check output $model=content_safety

                prompts:
                  - task: content_safety_check_output $model=content_safety
                    content: Check content safety
                """,
            )

    def test_output_content_safety_wrong_model_raises(self):
        """Check we raise ValueError when creating an output content safety rail with wrong model"""

        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_output \$model=content_safety` prompt template",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: local_content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                rails:
                  output:
                    flows:
                      - content safety check output $model=content_safety

                prompts:
                  - task: content_safety_check_input $model=content_safety
                    content: Check content safety
                    """,
            )


class TestTopicSafetyConfig:
    """Tests for topic-safety config validation"""

    def test_topic_safety_has_model_and_prompt(self):
        """Check we create config correctly when both model and prompt is provided"""
        config = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: topic_control
                engine: nim
                model: nvidia/llama-3.1-nemoguard-8b-topic-control

            rails:
              input:
                flows:
                    - topic safety check input $model=topic_control

            prompts:
              - task: topic_safety_check_input $model=topic_control
                content: Verify the user input is on-topic
            """,
        )

        # Check a few fields to make sure we created the config correctly
        assert config.models[0].type == "topic_control"
        assert config.models[0].model == "nvidia/llama-3.1-nemoguard-8b-topic-control"
        assert (
            config.rails.input.flows[0]
            == "topic safety check input $model=topic_control"
        )
        assert config.prompts[0].task == "topic_safety_check_input $model=topic_control"

    def test_topic_safety_no_prompt_raises(self):
        """Check if we don't provide a topic-safety prompt we raise a ValueError"""

        with pytest.raises(
            ValueError,
            match="You must provide a `topic_safety_check_input \$model=topic_control` prompt template",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: topic_control
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-topic-control

                rails:
                  input:
                    flows:
                        - topic safety check input $model=topic_control

                prompts:
                  - task: content_safety_check_input $model=content_safety
                    content: Check the content is safe
                """,
            )

    def test_topic_safety_no_model_raises(self):
        """Check if we don't provide a topic-safety model we raise a ValueError"""
        with pytest.raises(
            ValueError,
            match="No `topic_control` model provided for input flow `topic safety check input`",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                rails:
                  input:
                    flows:
                        - topic safety check input $model=topic_control

                prompts:
                  - task: topic_safety_check_input $model=topic_control
                    content: Verify the user input is on-topic
                    """,
            )

    def test_topic_safety_no_model_no_prompt_raises(self):
        """Check a missing model and prompt raises ValueError"""
        with pytest.raises(
            ValueError,
            match="You must provide a `topic_safety_check_input \$model=topic_control` prompt template",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                rails:
                  input:
                    flows:
                        - topic safety check input $model=topic_control

                prompts:
                  - task: content_safety_check_input $model=content_safety
                    content: Check the content is safe
                    """,
            )


class TestCombinedConfig:
    """Test combinations of content-safety and topic-safety rails with non-standard model names"""

    def test_hero_separate_models_no_prompts_raises(self):
        """Check if we use separate models for input and output content-safety this passes checks"""

        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_input \$model=my_content_safety` prompt template",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: my_content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                  - type: your_topic_control
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-topic-control

                  - type: our_content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety
                rails:
                  input:
                    flows:
                      - content safety check input $model=my_content_safety
                      - topic safety check input $model=your_topic_control

                  output:
                    flows:
                      - content safety check output $model=our_content_safety
                      """,
            )

    def test_hero_separate_models_with_prompts(self):
        """Check if we use separate models with non-standard names with prompts it all works"""

        config = RailsConfig.from_content(
            yaml_content="""
                models:
                  - type: my_content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                  - type: your_topic_control
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-topic-control

                  - type: our_content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety
                rails:
                  input:
                    flows:
                      - content safety check input $model=my_content_safety
                      - topic safety check input $model=your_topic_control

                  output:
                    flows:
                      - content safety check output $model=our_content_safety

                prompts:
                  - task: content_safety_check_input $model=my_content_safety
                    content: Check the input content is safe
                  - task: content_safety_check_output $model=our_content_safety
                    content: Check the output content is safe
                  - task: topic_safety_check_input $model=your_topic_control
                    content: Verify the user input is on-topic

                      """,
        )

        # Check a few fields to make sure we created config correctly
        assert config.models[0].type == "my_content_safety"
        assert config.models[1].type == "your_topic_control"
        assert config.models[2].type == "our_content_safety"

        assert (
            config.rails.input.flows[0]
            == "content safety check input $model=my_content_safety"
        )
        assert (
            config.rails.input.flows[1]
            == "topic safety check input $model=your_topic_control"
        )

        assert (
            config.rails.output.flows[0]
            == "content safety check output $model=our_content_safety"
        )

    def test_hero_with_prompts(self):
        """Create hero workflow with no prompts. Expect Content Safety input prompt check to fail"""
        config = RailsConfig.from_content(
            yaml_content="""
                models:
                  - type: main
                    engine: nim
                    model: meta/llama-3.3-70b-instruct

                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                  - type: your_topic_control
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-topic-control

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety
                      - topic safety check input $model=your_topic_control
                      - jailbreak detection model

                  output:
                    flows:
                      - content safety check output $model=content_safety

                  config:
                    jailbreak_detection:
                      nim_base_url: "https://ai.api.nvidia.com"
                      nim_server_endpoint: "/v1/security/nvidia/nemoguard-jailbreak-detect"
                      api_key_env_var: NVIDIA_API_KEY

                prompts:
                  - task: content_safety_check_input $model=content_safety
                    content: Check the input content is safe
                  - task: content_safety_check_output $model=content_safety
                    content: Check the output content is safe
                  - task: topic_safety_check_input $model=your_topic_control
                    content: Verify the user input is on-topic
                """
        )

        for model in config.models:
            assert model.engine == "nim"
        assert config.models[0].type == "main"
        assert config.models[0].model == "meta/llama-3.3-70b-instruct"
        assert config.models[1].type == "content_safety"
        assert config.models[1].model == "nvidia/llama-3.1-nemoguard-8b-content-safety"
        assert config.models[2].type == "your_topic_control"
        assert config.models[2].model == "nvidia/llama-3.1-nemoguard-8b-topic-control"

    def test_hero_no_prompts_raises(self):
        """Create hero workflow with no prompts. Expect Content Safety input prompt check to fail"""
        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_input \$model=content_safety` prompt template",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: main
                    engine: nim
                    model: meta/llama-3.3-70b-instruct

                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                  - type: topic_control
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-topic-control

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety
                      - topic safety check input $model=your_topic_control
                      - jailbreak detection model

                  output:
                    flows:
                      - content safety check output $model=content_safety

                  config:
                    jailbreak_detection:
                      nim_base_url: "https://ai.api.nvidia.com"
                      nim_server_endpoint: "/v1/security/nvidia/nemoguard-jailbreak-detect"
                      api_key_env_var: NVIDIA_API_KEY
                """
            )

    def test_hero_no_output_content_safety_prompt_raises(self):
        """Create hero workflow with no prompts. Expect Content Safety input prompt check to fail"""
        with pytest.raises(
            ValueError,
            match="You must provide a `topic_safety_check_input \$model=your_topic_control` prompt template",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: main
                    engine: nim
                    model: meta/llama-3.3-70b-instruct

                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                  - type: your_topic_control
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-topic-control

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety
                      - topic safety check input $model=your_topic_control
                      - jailbreak detection model

                  output:
                    flows:
                      - content safety check output $model=content_safety

                  config:
                    jailbreak_detection:
                      nim_base_url: "https://ai.api.nvidia.com"
                      nim_server_endpoint: "/v1/security/nvidia/nemoguard-jailbreak-detect"
                      api_key_env_var: NVIDIA_API_KEY

                prompts:
                  - task: content_safety_check_input $model=content_safety
                    content: Check the input content is safe
                """
            )

    def test_hero_no_topic_safety_prompt_raises(self):
        """Create hero workflow with no prompts. Expect Content Safety input prompt check to fail"""
        with pytest.raises(
            ValueError,
            match="You must provide a `topic_safety_check_input \$model=your_topic_control` prompt template",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: main
                    engine: nim
                    model: meta/llama-3.3-70b-instruct

                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                  - type: your_topic_control
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-topic-control

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety
                      - topic safety check input $model=your_topic_control
                      - jailbreak detection model

                  output:
                    flows:
                      - content safety check output $model=content_safety

                  config:
                    jailbreak_detection:
                      nim_base_url: "https://ai.api.nvidia.com"
                      nim_server_endpoint: "/v1/security/nvidia/nemoguard-jailbreak-detect"
                      api_key_env_var: NVIDIA_API_KEY

                prompts:
                  - task: content_safety_check_input $model=content_safety
                    content: Check the input content is safe
                  - task: content_safety_check_output $model=content_safety
                    content: Check the output content is safe
                """
            )

    def test_hero_topic_safety_prompt_raises(self):
        """Create hero workflow with no prompts. Expect Content Safety input prompt check to fail"""
        with pytest.raises(
            ValueError,
            match="You must provide a `content_safety_check_input \$model=content_safety` prompt template",
        ):
            _ = RailsConfig.from_content(
                yaml_content="""
                models:
                  - type: main
                    engine: nim
                    model: meta/llama-3.3-70b-instruct

                  - type: content_safety
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-content-safety

                  - type: your_topic_control
                    engine: nim
                    model: nvidia/llama-3.1-nemoguard-8b-topic-control

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety
                      - topic safety check input $model=your_topic_control
                      - jailbreak detection model

                  output:
                    flows:
                      - content safety check output $model=content_safety

                  config:
                    jailbreak_detection:
                      nim_base_url: "https://ai.api.nvidia.com"
                      nim_server_endpoint: "/v1/security/nvidia/nemoguard-jailbreak-detect"
                      api_key_env_var: NVIDIA_API_KEY

                prompts:
                  - task: topic_safety_check_input $model=topic_control
                    content: Verify the user input is on-topic
                """
            )
