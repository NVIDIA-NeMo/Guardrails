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

import os

import pytest

from nemoguardrails.eval.models import (
    ComplianceCheckLog,
    EvalConfig,
    EvalOutput,
    ExpectedOutput,
    InteractionSet,
    Policy,
)

ROOT = os.path.dirname(__file__)


def test_interaction_set_expected_output_instantiation():
    """Test that ExpectedOutput is correctly instantiated based on type."""

    # string type
    data = {"type": "string", "policy": "test_policy"}
    interaction_set = InteractionSet.model_validate(
        {"id": "test_id", "inputs": ["test input"], "expected_output": [data]}
    )
    assert len(interaction_set.expected_output) == 1
    assert interaction_set.expected_output[0].type == "string"
    assert interaction_set.expected_output[0].policy == "test_policy"

    # dict type
    data = {"type": "dict", "policy": "test_policy"}
    interaction_set = InteractionSet.model_validate(
        {"id": "test_id", "inputs": ["test input"], "expected_output": [data]}
    )
    assert len(interaction_set.expected_output) == 1
    assert interaction_set.expected_output[0].type == "dict"
    assert interaction_set.expected_output[0].policy == "test_policy"


def test_eval_config_policy_validation_valid():
    """Test that policy validation works correctly."""

    config = EvalConfig.model_validate(
        {
            "policies": [{"id": "policy1", "description": "Test policy"}],
            "interactions": [
                {
                    "id": "test_id",
                    "inputs": ["test input"],
                    "expected_output": [{"type": "string", "policy": "policy1"}],
                }
            ],
        }
    )
    assert len(config.policies) == 1
    assert len(config.interactions) == 1


def test_eval_config_policy_validation_invalid_policy_not_found():
    # invalid case, policy not found
    with pytest.raises(
        ValueError, match="Invalid policy id policy2 used in interaction set"
    ):
        EvalConfig.model_validate(
            {
                "policies": [{"id": "policy1", "description": "Test policy"}],
                "interactions": [
                    {
                        "id": "test_id",
                        "inputs": ["test input"],
                        "expected_output": [
                            {
                                "type": "string",
                                "policy": "policy2",
                            }
                        ],
                    }
                ],
            }
        )


def test_eval_config_from_path():
    """Test loading config from path."""

    config = EvalConfig.from_path(os.path.join(ROOT, "config_yml"))
    assert config is not None
    assert len(config.policies) > 0


def test_compliance_check_log():
    """Test ComplianceCheckLog model."""
    log = ComplianceCheckLog.model_validate({"id": "test_id", "llm_calls": []})
    assert log.id == "test_id"
    assert log.llm_calls == []


def test_eval_output():
    """Test EvalOutput model."""
    output = EvalOutput.model_validate(
        {
            "results": [
                {
                    "id": "test_id",
                    "input": "test input",
                    "output": "test_output",
                    "compliance_checks": [
                        {
                            "id": "check_id",
                            "created_at": "2024-01-01T00:00:00",
                            "method": "test_method",
                            "compliance": {"policy1": True},
                            "details": "test details",
                            "interaction_id": "test_id",
                        }
                    ],
                }
            ],
            "logs": [],
        }
    )
    assert len(output.results) == 1
    assert output.results[0].id == "test_id"
    assert output.results[0].input == "test input"
    assert output.results[0].output == "test_output"
    assert len(output.results[0].compliance_checks) == 1
    assert output.results[0].compliance_checks[0].id == "check_id"
    assert output.results[0].compliance_checks[0].interaction_id == "test_id"
