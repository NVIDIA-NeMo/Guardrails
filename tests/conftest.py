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

from unittest.mock import patch

import pytest

from nemoguardrails.context import reasoning_trace_var


def pytest_configure(config):
    patch("prompt_toolkit.PromptSession", autospec=True).start()


@pytest.fixture(autouse=True)
def reset_reasoning_trace():
    """Reset the reasoning_trace_var before each test.

    This fixture runs automatically for every test (autouse=True) to ensure
    a clean state for the reasoning trace context variable.

    current Issues with ContextVar approach, not only specific to this case:
        global State: ContextVar creates global state that's hard to track and manage
        implicit Flow: The reasoning trace flows through the system in a non-obvious way
        testing Complexity:  It causes test isolation problems that we are trying to avoid using this fixture
    """
    # reset the variable before the test
    reasoning_trace_var.set(None)
    yield
    # reset the variable after the test as well (in case the test fails)
    reasoning_trace_var.set(None)
