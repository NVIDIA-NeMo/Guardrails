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

from unittest.mock import MagicMock

import pytest

from nemoguardrails.llm.frameworks import (
    _reset_frameworks,
    get_default_framework,
    get_framework,
    register_framework,
    set_default_framework,
)
from nemoguardrails.types import LLMModel


@pytest.fixture(autouse=True)
def clean_registry():
    _reset_frameworks()
    yield
    _reset_frameworks()


class FakeFramework:
    def create_model(self, model_name, provider_name, model_kwargs=None):
        return MagicMock(spec=LLMModel)


class TestRegistry:
    def test_register_and_get_framework(self):
        fw = FakeFramework()
        register_framework("fake", fw)
        assert get_framework("fake") is fw

    def test_register_duplicate_raises_valueerror(self):
        register_framework("dup", FakeFramework())
        with pytest.raises(ValueError, match="already registered"):
            register_framework("dup", FakeFramework())

    def test_get_unregistered_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown framework"):
            get_framework("nonexistent")

    def test_langchain_lazy_auto_registration(self):
        fw = get_framework("langchain")
        from nemoguardrails.integrations.langchain.llm_adapter import LangChainFramework

        assert isinstance(fw, LangChainFramework)

    def test_set_and_get_default_framework(self):
        register_framework("custom", FakeFramework())
        set_default_framework("custom")
        assert get_default_framework() == "custom"

    def test_set_default_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown framework"):
            set_default_framework("nonexistent")

    def test_default_is_langchain(self):
        assert get_default_framework() == "langchain"

    def test_default_from_env_var(self, monkeypatch):
        monkeypatch.setenv("NEMOGUARDRAILS_LLM_FRAMEWORK", "litellm")
        _reset_frameworks()
        assert get_default_framework() == "litellm"

    def test_reset_clears_registry(self):
        register_framework("temp", FakeFramework())
        _reset_frameworks()
        with pytest.raises(KeyError):
            get_framework("temp")
