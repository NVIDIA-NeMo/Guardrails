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


import pytest
from pydantic import ValidationError

from nemoguardrails.rails.llm.config import Model, Rails, RailsConfig, RailsConfigData, TaskPrompt


def test_tool_rail_fields_use_canonical_names():
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "tool_call": {"flows": ["tool call validation"]},
                "tool_result": {"flows": ["tool result validation"]},
            },
        }
    )
    rails = config.rails

    assert rails.tool_call.flows == ["tool call validation"]
    assert rails.tool_result.flows == ["tool result validation"]
    assert set(rails.model_dump()) >= {"tool_call", "tool_result"}
    assert "tool_output" not in rails.model_dump()
    assert "tool_input" not in rails.model_dump()


def test_deprecated_tool_rail_fields_migrate():
    with pytest.warns(DeprecationWarning) as warnings:
        config = RailsConfig.from_content(
            config={
                "models": [],
                "rails": {
                    "tool_output": {"flows": ["tool call validation"]},
                    "tool_input": {"flows": ["tool result validation"]},
                },
            }
        )
    rails = config.rails

    tool_warnings = {str(item.message) for item in warnings if str(item.message).startswith("rails.tool_")}
    assert tool_warnings == {
        "rails.tool_output is deprecated; use rails.tool_call instead.",
        "rails.tool_input is deprecated; use rails.tool_result instead.",
    }
    assert rails.tool_call.flows == ["tool call validation"]
    assert rails.tool_result.flows == ["tool result validation"]

    with pytest.warns(DeprecationWarning, match="Use 'tool_call' instead"):
        assert rails.tool_output is rails.tool_call
    with pytest.warns(DeprecationWarning, match="Use 'tool_result' instead"):
        assert rails.tool_input is rails.tool_result


@pytest.mark.parametrize(
    ("canonical_name", "deprecated_name"),
    [("tool_call", "tool_output"), ("tool_result", "tool_input")],
)
def test_tool_rail_fields_reject_canonical_and_deprecated_names(canonical_name, deprecated_name):
    with pytest.raises(
        ValidationError, match=f"Cannot configure both rails.{deprecated_name} and rails.{canonical_name}"
    ):
        RailsConfig.from_content(config={"models": [], "rails": {canonical_name: {}, deprecated_name: {}}})


def test_deprecated_tool_rail_fields_are_marked_in_schema():
    properties = Rails.model_json_schema()["properties"]

    assert properties["tool_output"]["deprecated"] is True
    assert properties["tool_input"]["deprecated"] is True


def test_task_prompt_valid_content():
    prompt = TaskPrompt(task="example_task", content="This is a valid prompt.")
    assert prompt.task == "example_task"
    assert prompt.content == "This is a valid prompt."
    assert prompt.messages is None


def test_clavata_config_field_preserves_legacy_exports():
    import nemoguardrails.rails.llm.config as config_module
    from nemoguardrails.library.clavata import rail_config as clavata_config
    from nemoguardrails.rails.llm.config import ClavataRailConfig, ClavataRailOptions

    assert ClavataRailConfig is clavata_config.ClavataRailConfig
    assert ClavataRailOptions is clavata_config.ClavataRailOptions
    assert "ClavataRailConfig" not in config_module.__dict__
    assert "ClavataRailConfig" in config_module.__all__
    assert "ClavataRailConfig" in dir(config_module)


def test_clavata_config_field_preserves_defaults_and_schema():
    from nemoguardrails.rails.llm.config import ClavataRailConfig

    field = RailsConfigData.model_fields["clavata"]

    assert field.default_factory is ClavataRailConfig
    assert field.description == "Configuration for Clavata."
    assert isinstance(RailsConfigData().clavata, ClavataRailConfig)
    assert RailsConfigData.__module__ == "nemoguardrails.rails.llm.config"
    assert "clavata" in RailsConfigData.model_json_schema()["properties"]


def test_clavata_config_field_yaml_preserves_default_and_null():
    from nemoguardrails.rails.llm.config import ClavataRailConfig

    default_config = RailsConfig.from_content(
        yaml_content="""
            models: []
        """,
    )
    null_config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata: null
        """,
    )

    assert isinstance(default_config.rails.config.clavata, ClavataRailConfig)
    assert null_config.rails.config.clavata is None


def test_clavata_config_field_yaml_parses_configured_value():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    AnimalSounds: 00000000-0000-0000-0000-000000000000
                  input:
                    policy: AnimalSounds
                    labels:
                      - DogBarking
        """,
    )

    assert config.rails.config.clavata.policies["AnimalSounds"] == "00000000-0000-0000-0000-000000000000"
    assert config.rails.config.clavata.input.policy == "AnimalSounds"
    assert config.rails.config.clavata.input.labels == ["DogBarking"]


def test_builtin_rails_config_fields_canonical_set_and_legacy_exports():
    import nemoguardrails.rails.llm.config as config_module
    from nemoguardrails.manifests import all_rail_manifests
    from nemoguardrails.manifests.config_schema import RailConfigSpec
    from nemoguardrails.rails.llm.rails_config_fields import (
        all_config_fields,
        config_key_to_rail_name,
    )

    expected_exports = {
        "ai_defense": ("AIDefenseRailConfig",),
        "autoalign": ("AutoAlignOptions", "AutoAlignRailConfig"),
        "clavata": ("ClavataRailConfig", "ClavataRailOptions"),
        "content_safety": ("ContentSafetyConfig", "MultilingualConfig", "ReasoningConfig"),
        "context_bloat_detection": ("ContextBloatDetectionConfig",),
        "crowdstrike_aidr": ("CrowdStrikeAIDRRailConfig",),
        "factchecking": ("FactCheckingRailConfig",),
        "f5": ("F5GuardrailsRailConfig",),
        "fiddler": ("FiddlerGuardrails",),
        "gliner": ("GLiNERDetection", "GLiNERDetectionOptions"),
        "guardrails_ai": ("GuardrailsAIRailConfig", "GuardrailsAIValidatorConfig"),
        "hf_classifier": (
            "HFClassifierConfig",
            "LocalHFClassifierConfig",
            "RemoteHFClassifierConfig",
        ),
        "injection_detection": ("InjectionDetection",),
        "jailbreak_detection": ("JailbreakDetectionConfig",),
        "pangea": ("PangeaRailConfig", "PangeaRailOptions"),
        "patronusai": (
            "PatronusEvaluationSuccessStrategy",
            "PatronusEvaluateApiParams",
            "PatronusEvaluateConfig",
            "PatronusRailConfig",
        ),
        "polygraf": ("PolygrafDetection", "PolygrafDetectionOptions"),
        "privateai": ("PrivateAIDetection", "PrivateAIDetectionOptions"),
        "regex": ("RegexDetection", "RegexDetectionOptions"),
        "sensitive_data_detection": (
            "SensitiveDataDetection",
            "SensitiveDataDetectionOptions",
        ),
        "trend_micro": ("TrendMicroRailConfig",),
    }
    expected_rail_names = set(expected_exports) | {
        "activefence",
        "cleanlab",
        "factchecking.align_score",
        "gcp_moderate_text",
        "hallucination",
        "llama_guard",
        "policyai",
        "prompt_security",
        "self_check.facts",
        "self_check.input_check",
        "self_check.output_check",
        "topic_safety",
    }
    expected_config_keys = {
        "ai_defense",
        "autoalign",
        "clavata",
        "content_safety",
        "context_bloat_detection",
        "crowdstrike_aidr",
        "fact_checking",
        "f5",
        "fiddler",
        "gliner",
        "guardrails_ai",
        "hf_classifier",
        "injection_detection",
        "jailbreak_detection",
        "pangea",
        "patronus",
        "polygraf",
        "privateai",
        "regex_detection",
        "sensitive_data_detection",
        "trend_micro",
    }

    manifests = all_rail_manifests()
    config_fields = all_config_fields()
    config_specs = {field.config_key: field.config for field in config_fields.values()}

    assert set(manifests) == expected_rail_names
    assert set(config_specs) == expected_config_keys
    assert set(RailsConfigData.model_fields) == expected_config_keys
    assert config_key_to_rail_name() == {
        "fact_checking": "factchecking",
        "patronus": "patronusai",
        "regex_detection": "regex",
    }

    for rail_name, export_names in expected_exports.items():
        field = config_fields[rail_name]
        assert isinstance(field.config, RailConfigSpec)
        assert set(field.config.exports) == set(export_names)
        for export_name in export_names:
            assert getattr(config_module, export_name) is field.config.exports[export_name]

    assert "topic_safety" not in config_fields


def test_rail_config_specs_are_not_hashable():
    from nemoguardrails.manifests.config_schema import RailConfigSpec, rail_field

    assert RailConfigSpec.__hash__ is None

    with pytest.raises(TypeError, match="unhashable"):
        hash(RailConfigSpec(annotation=str, field_info=rail_field(default=None)))


def test_legacy_config_spec_key_is_accepted():
    from nemoguardrails.manifests.config_schema import RailConfigSpec, rail_field

    spec = RailConfigSpec(key="legacy", annotation=str, field_info=rail_field(default=None))

    assert spec.key == "legacy"


def test_discover_rails_config_fields_fails_on_reentrant_discovery():
    import nemoguardrails.rails.llm.rails_config_fields as rails_config_fields

    rails_config_fields._reset_rails_config_fields_cache()
    rails_config_fields._discovering = True

    try:
        with pytest.raises(RuntimeError, match="re-entered"):
            rails_config_fields.discover_rails_config_fields()
    finally:
        rails_config_fields._reset_rails_config_fields_cache()


def test_builtin_rail_config_modules_use_supported_imports():
    import ast
    from pathlib import Path

    allowed_imports = {
        "__future__",
        "enum",
        "logging",
        "os",
        "pydantic",
        "re",
        "typing",
        "nemoguardrails.manifests.config_schema",
    }
    library_path = Path("nemoguardrails/library")
    disallowed_imports = []

    for path in sorted(library_path.rglob("rail_config.py")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            if isinstance(node, ast.Import):
                disallowed_imports.extend(
                    f"{path}: import {alias.name}" for alias in node.names if alias.name not in allowed_imports
                )
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name not in allowed_imports:
                    disallowed_imports.append(f"{path}: from {module_name} import ...")

    assert disallowed_imports == []


def test_builtin_rail_config_models_use_supported_base_model():
    import importlib

    from pydantic import BaseModel

    from nemoguardrails.manifests.config_schema import RailConfigBaseModel
    from nemoguardrails.rails.llm.rails_config_fields import all_config_fields

    unsupported_models = []
    for origin in sorted({field.origin for field in all_config_fields().values()}):
        module = importlib.import_module(origin)
        for name, value in vars(module).items():
            if not isinstance(value, type):
                continue
            if value.__module__ != origin:
                continue
            if not issubclass(value, BaseModel) or value is RailConfigBaseModel:
                continue
            if not issubclass(value, RailConfigBaseModel):
                unsupported_models.append(f"{origin}.{name}")
                continue
    assert unsupported_models == []

    from nemoguardrails.library.gliner.rail_config import GLiNERDetection, GLiNERDetectionOptions

    assert RailConfigBaseModel.model_config.get("extra") == "ignore"
    assert GLiNERDetection.model_config.get("extra") == "forbid"
    assert GLiNERDetectionOptions.model_config.get("extra") == "forbid"


def test_rail_config_schema_exposes_supported_surface():
    import nemoguardrails.manifests.config_schema as rail_config_schema

    assert "BaseModel" not in rail_config_schema.__all__
    assert not hasattr(rail_config_schema, "BaseModel")
    assert rail_config_schema.RailConfigBaseModel.model_config.get("extra") == "ignore"


def test_rail_config_modules_are_referenced_by_manifests():
    from pathlib import Path

    from nemoguardrails.manifests import all_rail_manifests, import_ref_target

    library_path = Path("nemoguardrails/library")
    config_files = sorted(library_path.rglob("config.py"))
    rail_config_modules = {
        ".".join(("nemoguardrails", "library", *path.relative_to(library_path).with_suffix("").parts))
        for path in sorted(library_path.rglob("rail_config.py"))
    }
    manifest_config_modules = [
        import_ref_target(manifest.config_schema.spec).partition(":")[0]
        for manifest in all_rail_manifests().values()
        if manifest.config_schema is not None
    ]

    assert config_files == []
    assert rail_config_modules == set(manifest_config_modules)
    assert len(manifest_config_modules) == len(set(manifest_config_modules))


def test_builtin_rail_exports_do_not_shadow_core_config_names():
    import ast
    from pathlib import Path

    import nemoguardrails.rails.llm.config as config_module
    from nemoguardrails.rails.llm.rails_config_fields import config_exported_names

    def collect_targets(target):
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {name for element in target.elts for name in collect_targets(element)}
        return set()

    config_source = Path(config_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(config_source)
    core_names = set()

    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            core_names.add(node.name)
        elif isinstance(node, ast.Import):
            core_names.update(alias.asname or alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            core_names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                core_names.update(collect_targets(target))
        elif isinstance(node, ast.AnnAssign):
            core_names.update(collect_targets(node.target))
        elif isinstance(node, ast.For):
            core_names.update(collect_targets(node.target))

    assert config_module.Model is Model
    assert config_module.Model.__module__ == "nemoguardrails.rails.llm.config"
    assert set(config_exported_names()).isdisjoint(core_names)


def test_rail_config_export_shadowing_validation_reports_origin():
    from nemoguardrails.rails.llm.rails_config_fields import validate_no_config_export_shadowing

    with pytest.raises(ValueError, match="ClavataRailConfig.*clavata.*rail_config"):
        validate_no_config_export_shadowing({"ClavataRailConfig"})

    validate_no_config_export_shadowing({"MissingRailExport"})


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
    prompt = TaskPrompt(task="example_task", content="Test prompt", stop=["\n", "Human:", "Assistant:"])
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

    with pytest.raises(ValueError, match="Both config files should have the same actions_server_url"):
        config1 + config2


def test_rails_config_simple_field_overwriting():
    """Tests that fields from the second config overwrite fields from the first config."""
    config1 = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        lowest_temperature=0.1,
        colang_version="1.0",
    )

    config2 = RailsConfig(
        models=[Model(type="secondary", engine="anthropic", model="claude-3")],
        lowest_temperature=0.5,
        colang_version="2.x",
    )

    result = config1 + config2

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
