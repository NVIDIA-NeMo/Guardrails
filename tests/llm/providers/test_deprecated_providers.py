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

import warnings
from unittest.mock import patch

import pytest

from nemoguardrails.llm.providers.providers import (
    discover_langchain_providers,
)


class MockBaseLLM:
    def _call(self, *args, **kwargs):
        return "Mock response"


@pytest.fixture
def mock_discover_function():
    with patch("nemoguardrails.llm.providers.providers._discover_langchain_community_llm_providers") as mock_func:
        mock_providers = {"mock_provider": MockBaseLLM}
        mock_func.return_value = mock_providers
        with patch("nemoguardrails.llm.providers.providers._patch_acall_method_to") as mock_patch:
            with patch("nemoguardrails.llm.providers.providers._llm_providers") as mock_llm_providers:
                mock_llm_providers.update(mock_providers)
                yield mock_func


def test_discover_langchain_providers_deprecation(mock_discover_function):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        discover_langchain_providers()
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
        assert "v0.15.0" in str(w[0].message)


def test_discover_langchain_providers_functionality(mock_discover_function):
    # ensure the function still works as expected
    discover_langchain_providers()
    # as the function is deprecated, we verify that it calls the underlying function
    mock_discover_function.assert_called_once()
