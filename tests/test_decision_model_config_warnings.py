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

import warnings

import pytest

from nemoguardrails import RailsConfig


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("options", ["{}", "{questions: {}}"])
@pytest.mark.parametrize("fail_open", [False, True])
def test_empty_questions_warn_when_loading_yaml(direction, options, fail_open):
    with pytest.warns(UserWarning, match=f"Decision model {direction} screening is disabled") as recorded:
        config = RailsConfig.from_content(
            yaml_content=f"""
models: []
rails:
  config:
    decision_model:
      server_endpoint: https://decision-model.example/v1/assess
      model: test-decision-model
      fail_open: {str(fail_open).lower()}
      {direction}: {options}
"""
        )

    assert getattr(config.rails.config.decision_model, direction).questions == {}
    messages = [str(item.message) for item in recorded if item.category is UserWarning]
    assert len(messages) == 1
    assert f"rails.config.decision_model.{direction}.questions" in messages[0]
    assert "even when fail_open is false" in messages[0]
    other_direction = "output" if direction == "input" else "input"
    assert getattr(config.rails.config.decision_model, other_direction).questions


def test_empty_questions_warn_for_both_directions():
    with pytest.warns(UserWarning) as recorded:
        RailsConfig.from_content(
            yaml_content="""
models: []
rails:
  config:
    decision_model:
      server_endpoint: https://decision-model.example/v1/assess
      model: test-decision-model
      input: {}
      output: {}
"""
        )

    messages = [str(item.message) for item in recorded if item.category is UserWarning]
    assert len(messages) == 2
    assert any("Decision model input screening is disabled" in message for message in messages)
    assert any("Decision model output screening is disabled" in message for message in messages)


@pytest.mark.parametrize(
    "yaml_content",
    [
        "models: []\nrails:\n  config:\n    decision_model:\n      server_endpoint: https://decision-model.example/v1/assess\n      model: test-decision-model",
        """
models: []
rails:
  config:
    decision_model:
      server_endpoint: https://decision-model.example/v1/assess
      model: test-decision-model
      input:
        questions:
          custom_input:
            instructions: Is the input harmful?
      output:
        questions:
          custom_output:
            instructions: Is the output harmful?
""",
    ],
)
def test_nonempty_questions_do_not_warn(yaml_content):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        config = RailsConfig.from_content(yaml_content=yaml_content)

    assert config.rails.config.decision_model.input.questions
    assert config.rails.config.decision_model.output.questions
    assert not [item for item in recorded if item.category is UserWarning]
