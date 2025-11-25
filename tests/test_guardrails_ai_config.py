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

"""Tests for guardrails_ai configuration parsing."""

from nemoguardrails.rails.llm.config import RailsConfig


def test_guardrails_ai_config_parsing():
    """Test that guardrails_ai configuration is properly parsed."""

    config_content = """
    models:
      - type: main
        engine: openai
        model: gpt-4

    rails:
      config:
        guardrails_ai:
          validators:
            - name: toxic_language
              parameters:
                threshold: 0.7
                validation_method: "full"
              metadata:
                context: "customer_service"

            - name: pii
              parameters:
                entities: ["email", "phone"]
              metadata: {}

            - name: competitor_check
              parameters:
                competitors: ["Apple", "Google"]
              metadata:
                strict: true
    """

    config = RailsConfig.from_content(yaml_content=config_content)

    assert config.rails.config.guardrails_ai is not None

    validators = config.rails.config.guardrails_ai.validators
    assert len(validators) == 3

    toxic_validator = validators[0]
    assert toxic_validator.name == "toxic_language"
    assert toxic_validator.parameters["threshold"] == 0.7
    assert toxic_validator.parameters["validation_method"] == "full"
    assert toxic_validator.metadata["context"] == "customer_service"

    pii_validator = validators[1]
    assert pii_validator.name == "pii"
    assert pii_validator.parameters["entities"] == ["email", "phone"]
    assert pii_validator.metadata == {}

    competitor_validator = validators[2]
    assert competitor_validator.name == "competitor_check"
    assert competitor_validator.parameters["competitors"] == ["Apple", "Google"]
    assert competitor_validator.metadata["strict"] is True


def test_guardrails_ai_get_validator_config():
    """Test that guardrails_ai configuration is properly parsed."""

    config_content = """
    models:
      - type: main
        engine: openai
        model: gpt-4

    rails:
      config:
        guardrails_ai:
          validators:
            - name: toxic_language
              parameters:
                threshold: 0.7
                validation_method: "full"
              metadata:
                context: "customer_service"

            - name: pii
              parameters:
                entities: ["email", "phone"]
              metadata: {}

            - name: competitor_check
              parameters:
                competitors: ["Apple", "Google"]
              metadata:
                strict: true
    """

    config = RailsConfig.from_content(yaml_content=config_content)

    assert config.rails.config.guardrails_ai is not None

    guardrails_ai = config.rails.config.guardrails_ai
    validators = guardrails_ai.validators
    assert len(validators) == 3

    toxic_validator = guardrails_ai.get_validator_config("toxic_language")
    assert toxic_validator.name == "toxic_language"

    pii_validator = guardrails_ai.get_validator_config("pii")
    assert pii_validator.name == "pii"
    assert pii_validator.parameters["entities"] == ["email", "phone"]
    assert pii_validator.metadata == {}

    competitor_validator = validators[2]
    assert competitor_validator.name == "competitor_check"
    assert competitor_validator.parameters["competitors"] == ["Apple", "Google"]
    assert competitor_validator.metadata["strict"] is True


def test_guardrails_ai_config_defaults():
    """Test default values for guardrails_ai configuration."""

    config_content = """
    models:
      - type: main
        engine: openai
        model: gpt-4

    rails:
      config:
        guardrails_ai:
          validators:
            - name: simple_validator
    """

    config = RailsConfig.from_content(yaml_content=config_content)

    validator = config.rails.config.guardrails_ai.validators[0]
    assert validator.name == "simple_validator"
    assert validator.parameters == {}
    assert validator.metadata == {}


def test_guardrails_ai_config_empty():
    """Test empty guardrails_ai configuration."""

    config_content = """
    models:
      - type: main
        engine: openai
        model: gpt-4
    """

    config = RailsConfig.from_content(yaml_content=config_content)

    assert config.rails.config.guardrails_ai is not None
    assert config.rails.config.guardrails_ai.validators == []
