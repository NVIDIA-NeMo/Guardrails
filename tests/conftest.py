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

from unittest.mock import patch

import pytest

try:
    import langchain_core  # noqa: F401
except ImportError:
    collect_ignore_glob = ["integrations/langchain/*.py", "integrations/langchain/**/*.py"]

REASONING_TRACE_MOCK_PATH = "nemoguardrails.actions.llm.generation.get_and_clear_reasoning_trace_contextvar"


@pytest.fixture(autouse=True)
def reset_reasoning_trace_var():
    """Reset reasoning_trace_var before each test to prevent state leakage."""
    from nemoguardrails.context import reasoning_trace_var

    reasoning_trace_var.set(None)
    yield
    reasoning_trace_var.set(None)


@pytest.fixture
def langchain_framework():
    from nemoguardrails.llm.frameworks import _reset_frameworks, set_default_framework

    _reset_frameworks()
    set_default_framework("langchain")
    yield
    _reset_frameworks()


def pytest_configure(config):
    patch("prompt_toolkit.PromptSession", autospec=True).start()
