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

"""Unit tests for the ToolRailAction base (span wrapper + fail-closed error handling)."""

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.tool_rail_action import ToolRailAction


class _ProbeRail(ToolRailAction):
    action_name = "probe tool rail"


class TestToolRailActionGuarded:
    def test_requires_model_is_false(self):
        assert _ProbeRail().requires_model is False

    def test_returns_safe_check_result(self):
        assert _ProbeRail()._guarded(lambda: RailResult(is_safe=True)) == RailResult(is_safe=True)

    def test_returns_unsafe_check_result(self):
        result = _ProbeRail()._guarded(lambda: RailResult(is_safe=False, reason="bad call"))
        assert result.is_safe is False
        assert result.reason == "bad call"

    def test_exception_fails_closed(self):
        def boom() -> RailResult:
            raise ValueError("kaboom")

        result = _ProbeRail()._guarded(boom)
        assert result.is_safe is False
        assert result.reason is not None
        assert "probe tool rail" in result.reason
        assert "kaboom" in result.reason
