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

import logging

import pytest
import yaml
from pydantic import ValidationError

from nemoguardrails.rails.llm.config import RailsConfig


@pytest.fixture(params=["dict", "yaml", "file", "directory", "validate"])
def load_config(request, tmp_path):
    def load(data):
        data = {"models": [], **data}
        if request.param == "dict":
            return RailsConfig.from_content(config=data)
        if request.param == "yaml":
            return RailsConfig.from_content(yaml_content=yaml.safe_dump(data))
        if request.param == "validate":
            return RailsConfig.model_validate(data)
        (tmp_path / "config.yml").write_text(yaml.safe_dump(data))
        path = tmp_path if request.param == "directory" else tmp_path / "config.yml"
        return RailsConfig.from_path(str(path))

    return load


@pytest.mark.parametrize(
    ("data", "path", "suggestion"),
    [
        ({"railz": {"input": {"flows": ["some rail"]}}}, "railz", "rails"),
        ({"flowz": []}, "flowz", "flows"),
        ({"rails": {"input": {"flowz": ["some rail"]}}}, "rails.input.flowz", "flows"),
        ({"rails": {"execution": {"flows": ["some rail"]}}}, "rails.execution", None),
        ({"rails": {"config": {"glienr": {}}}}, "rails.config.glienr", "gliner"),
        (
            {"instructions": [{"type": "general", "content": "Example instruction", "typo": True}]},
            "instructions[0].typo",
            None,
        ),
        (
            {"rails": {"config": {"sensitive_data_detection": {"input": {"entites": ["PERSON"]}}}}},
            "rails.config.sensitive_data_detection.input.entites",
            "entities",
        ),
        (
            {
                "rails": {
                    "config": {
                        "hf_classifier": {
                            "named": {
                                "engine": "local",
                                "model": "example",
                                "blocked_labels": ["unsafe"],
                                "threshhold": 0.8,
                            }
                        }
                    }
                }
            },
            "rails.config.hf_classifier.named.threshhold",
            "threshold",
        ),
        (
            {
                "rails": {
                    "config": {
                        "hf_classifier": {
                            "named": {
                                "engine": "vllm",
                                "model": "example",
                                "base_url": "https://example.com",
                                "blocked_labels": ["unsafe"],
                                "task": "text-classification",
                            }
                        }
                    }
                }
            },
            "rails.config.hf_classifier.named.task",
            None,
        ),
    ],
)
def test_unknown_keys_warn_through_real_loading(load_config, caplog, data, path, suggestion):
    with caplog.at_level(logging.WARNING, logger="nemoguardrails.rails.llm.config"):
        config = load_config(data)
    messages = [r.message for r in caplog.records if r.name == "nemoguardrails.rails.llm.config"]
    assert len(messages) == 1
    assert f"Unknown configuration key '{path}'" in messages[0]
    if suggestion:
        assert f"Did you mean '{suggestion}'?" in messages[0]
    if path == "rails.execution":
        assert "will not enable execution rails" in messages[0]
    assert config.rails.input.flows == []


def test_valid_extension_fields_are_preserved_without_warnings(load_config, caplog):
    config = load_config(
        {
            "custom_data": {"my_custom_key": {"arbitrary": True}},
            "models": [{"type": "main", "engine": "openai", "model": "example", "parameters": {"vendor_option": True}}],
            "rails": {
                "output": {"streaming": {"vendor_option": True}},
                "config": {
                    "sensitive_data_detection": {"input": {"entities": ["PERSON"]}},
                    "hf_classifier": {
                        "named": {
                            "engine": "local",
                            "model": "example",
                            "blocked_labels": ["unsafe"],
                            "parameters": {"vendor_option": True},
                        }
                    },
                },
            },
        }
    )
    assert config.custom_data["my_custom_key"] == {"arbitrary": True}
    assert config.models[0].parameters["vendor_option"] is True
    assert config.rails.output.streaming.model_extra["vendor_option"] is True
    assert config.rails.config.sensitive_data_detection.input.entities == ["PERSON"]
    assert not [r for r in caplog.records if r.name == "nemoguardrails.rails.llm.config"]


def test_strict_library_schema_still_rejects_unknown_keys(load_config):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_config({"rails": {"config": {"gliner": {"input": {"entites": []}}}}})


def test_imported_config_warns_without_logging_source_path(tmp_path, caplog):
    imported = tmp_path / "secret-config-value"
    imported.mkdir()
    (imported / "config.yml").write_text("railz: {}\nrails:\n  input:\n    flowz: []\n")
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.yml").write_text(yaml.safe_dump({"models": [], "import_paths": [str(imported)]}))
    RailsConfig.from_path(str(root))
    assert "Unknown configuration key 'railz'" in caplog.text
    assert "secret-config-value" not in caplog.text
    assert "Unknown configuration key 'rails.input.flowz'" in caplog.text


def test_warning_does_not_include_config_values(load_config, caplog):
    load_config({"railz": {"private": "secret-config-value"}})
    assert "Unknown configuration key 'railz'" in caplog.text
    assert "secret-config-value" not in caplog.text
