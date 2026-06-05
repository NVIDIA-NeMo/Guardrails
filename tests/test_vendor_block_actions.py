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

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.cleanlab.actions import _cleanlab_outcome
from nemoguardrails.library.trend_micro.actions import GuardResult, _trend_micro_outcome


@pytest.mark.parametrize(
    ("guard_result", "expected"),
    [
        (
            GuardResult(action="Allow", reason="No threats detected"),
            RailOutcome.allow(reason="No threats detected", action="Allow"),
        ),
        (
            GuardResult(action="Block", reason="Prompt Attack Detected"),
            RailOutcome.block(reason="Prompt Attack Detected", action="Block"),
        ),
    ],
)
def test_trend_micro_outcome_preserves_action_and_reason(guard_result, expected):
    assert _trend_micro_outcome(guard_result) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.59, RailOutcome.block(trustworthiness_score=0.59)),
        (0.6, RailOutcome.allow(trustworthiness_score=0.6)),
        (0.61, RailOutcome.allow(trustworthiness_score=0.61)),
    ],
)
def test_cleanlab_outcome_pins_threshold(score, expected):
    assert _cleanlab_outcome(score) == expected
