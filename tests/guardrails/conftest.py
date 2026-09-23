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

"""Fixtures shared across the IORails guardrails tests."""

import pytest

from nemoguardrails.context import llm_call_info_var, llm_response_metadata_var, llm_stats_var


@pytest.fixture
def reset_llm_call_context():
    """Reset the context variables ``llm_call`` writes, before and after a test.

    ``reasoning_trace_var``, ``tool_calls_var``, and ``explain_info_var`` already
    have autouse resets in ``tests/conftest.py``; these three do not.
    """
    call_info_token = llm_call_info_var.set(None)
    stats_token = llm_stats_var.set(None)
    metadata_token = llm_response_metadata_var.set(None)
    yield
    llm_call_info_var.reset(call_info_token)
    llm_stats_var.reset(stats_token)
    llm_response_metadata_var.reset(metadata_token)
