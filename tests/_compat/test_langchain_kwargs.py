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

from types import SimpleNamespace

import pytest

from nemoguardrails._compat.langchain_kwargs import check_langchain_kwargs


def _model(model_type="main", engine="openai", parameters=None):
    return SimpleNamespace(type=model_type, engine=engine, parameters=parameters or {})


class TestNoOpOnNonDefaultFramework:
    def test_langchain_framework_skips_check(self):
        models = [_model(parameters={"streaming": True, "verbose": True, "model_kwargs": {"x": 1}})]
        check_langchain_kwargs(models, active_framework="langchain")

    def test_other_framework_skips_check(self):
        models = [_model(parameters={"streaming": True})]
        check_langchain_kwargs(models, active_framework="some_custom_framework")


class TestBaseFlagsRaise:
    def test_streaming_raises_on_default(self):
        models = [_model(parameters={"streaming": True})]
        with pytest.raises(ValueError, match=r"streaming"):
            check_langchain_kwargs(models, active_framework="default")

    def test_each_base_flag_raises(self):
        for flag in ("streaming", "disable_streaming", "verbose", "cache", "callbacks", "tags", "metadata", "name"):
            with pytest.raises(ValueError, match=rf"{flag}"):
                check_langchain_kwargs([_model(parameters={flag: True})], active_framework="default")

    def test_model_kwargs_uses_unpack_action(self):
        with pytest.raises(ValueError, match=r"unpack `model_kwargs`"):
            check_langchain_kwargs([_model(parameters={"model_kwargs": {"top_k": 50}})], active_framework="default")


class TestProviderAliasPatternDetection:
    def test_nim_alias_still_detected_via_pattern(self):
        with pytest.raises(ValueError, match=r"nvidia_api_key.*to.*api_key"):
            check_langchain_kwargs(
                [_model(engine="nim", parameters={"nvidia_api_key": "nvapi-..."})],
                active_framework="default",
            )

    def test_nvidia_ai_endpoints_alias_still_detected_via_pattern(self):
        with pytest.raises(ValueError, match=r"nvidia_base_url.*to.*base_url"):
            check_langchain_kwargs(
                [_model(engine="nvidia_ai_endpoints", parameters={"nvidia_base_url": "https://..."})],
                active_framework="default",
            )

    def test_openai_api_key_detected(self):
        with pytest.raises(ValueError, match=r"openai_api_key.*to.*api_key"):
            check_langchain_kwargs(
                [_model(engine="openai", parameters={"openai_api_key": "sk-..."})],
                active_framework="default",
            )

    def test_cohere_api_key_detected(self):
        with pytest.raises(ValueError, match=r"cohere_api_key.*to.*api_key"):
            check_langchain_kwargs(
                [_model(engine="cohere", parameters={"cohere_api_key": "co-..."})],
                active_framework="default",
            )

    @pytest.mark.parametrize("engine", ["azure", "azure_openai"])
    def test_azure_endpoint_allowed_for_azure_engines(self, engine):
        check_langchain_kwargs(
            [_model(engine=engine, parameters={"azure_endpoint": "https://..."})],
            active_framework="default",
        )

    def test_azure_endpoint_detected_and_collapsed_to_base_url_for_non_azure_engine(self):
        with pytest.raises(ValueError, match=r"azure_endpoint.*to.*base_url"):
            check_langchain_kwargs(
                [_model(engine="openai", parameters={"azure_endpoint": "https://..."})],
                active_framework="default",
            )

    def test_xyz_base_url_detected(self):
        with pytest.raises(ValueError, match=r"xyz_base_url.*to.*base_url"):
            check_langchain_kwargs(
                [_model(engine="xyz", parameters={"xyz_base_url": "https://..."})],
                active_framework="default",
            )

    def test_huggingfacehub_api_base_collapsed_to_base_url(self):
        with pytest.raises(ValueError, match=r"huggingfacehub_api_base.*to.*base_url"):
            check_langchain_kwargs(
                [_model(engine="huggingface", parameters={"huggingfacehub_api_base": "https://..."})],
                active_framework="default",
            )


class TestCanonicalNamesNotFalseFlagged:
    def test_api_key_does_not_trigger(self):
        check_langchain_kwargs(
            [_model(parameters={"api_key": "sk-test"})],
            active_framework="default",
        )

    def test_base_url_does_not_trigger(self):
        check_langchain_kwargs(
            [_model(parameters={"base_url": "https://api.openai.com/v1"})],
            active_framework="default",
        )


class TestValidParameters:
    def test_legitimate_wire_fields_pass(self):
        params = {
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "tools": [{"type": "function"}],
        }
        check_langchain_kwargs([_model(parameters=params)], active_framework="default")

    def test_client_config_fields_pass(self):
        params = {
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "timeout": 30,
            "max_retries": 3,
        }
        check_langchain_kwargs([_model(parameters=params)], active_framework="default")

    def test_provider_extensions_pass(self):
        params = {
            "nvext": {"foo": "bar"},
            "min_p": 0.05,
            "top_k": 40,
            "guided_json": {"type": "object"},
            "keep_alive": "10m",
            "temperature": 0.7,
        }
        check_langchain_kwargs([_model(engine="nim", parameters=params)], active_framework="default")


class TestEmptyConfig:
    def test_no_models(self):
        check_langchain_kwargs([], active_framework="default")

    def test_model_without_parameters(self):
        check_langchain_kwargs([_model(parameters=None)], active_framework="default")

    def test_model_with_empty_parameters(self):
        check_langchain_kwargs([_model(parameters={})], active_framework="default")


class TestMultipleViolations:
    def test_aggregates_across_models(self):
        models = [
            _model(model_type="main", parameters={"streaming": True}),
            _model(model_type="self_check_input", parameters={"verbose": True}),
        ]
        with pytest.raises(ValueError) as excinfo:
            check_langchain_kwargs(models, active_framework="default")
        assert "main" in str(excinfo.value)
        assert "self_check_input" in str(excinfo.value)

    def test_message_includes_sunset_note(self):
        with pytest.raises(ValueError, match=r"removed in 0\.23\.0"):
            check_langchain_kwargs([_model(parameters={"streaming": True})], active_framework="default")


class TestErrorMessageBothPaths:
    def test_message_contains_both_remediation_paths_and_sunset(self):
        with pytest.raises(ValueError) as excinfo:
            check_langchain_kwargs(
                [_model(engine="nim", parameters={"streaming": True, "nvidia_api_key": "nvapi-..."})],
                active_framework="default",
            )
        message = str(excinfo.value)
        assert "Adapt to the default framework" in message
        assert "Keep 0.21 LangChain behavior" in message
        assert "NEMOGUARDRAILS_LLM_FRAMEWORK=langchain" in message
        assert "removed in 0.23.0" in message
        assert "remove `streaming`" in message
        assert "rename `nvidia_api_key` to `api_key`" in message
