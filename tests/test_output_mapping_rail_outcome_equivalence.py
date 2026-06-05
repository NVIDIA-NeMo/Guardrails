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

from nemoguardrails.actions import action
from nemoguardrails.actions.output_mapping import is_output_blocked
from nemoguardrails.library.content_safety.actions import (
    content_safety_check_output_mapping,
)
from nemoguardrails.library.self_check.output_check.actions import self_check_output


@action()
def detect_only_action(result):
    return result


def _candidate_is_blocked(result):
    if isinstance(result, bool):
        return not result
    if isinstance(result, (int, float)):
        return result < 0.5
    return False


REPRESENTATIVE_RAW_RETURNS = [
    True,
    False,
    1.0,
    0.0,
    0.49,
    0.5,
    0.51,
    42,
    "hi",
    None,
    {"allowed": False},
    [],
]


@pytest.mark.parametrize("raw_return", REPRESENTATIVE_RAW_RETURNS)
def test_candidate_interpretation_matches_is_output_blocked(raw_return):
    outcome_blocked = _candidate_is_blocked(raw_return)
    mapping_blocked = is_output_blocked(raw_return, detect_only_action)

    assert outcome_blocked == mapping_blocked


def test_numeric_threshold_boundary_is_pinned_on_both_sides():
    assert _candidate_is_blocked(0.49) is True
    assert is_output_blocked(0.49, detect_only_action) is True

    assert _candidate_is_blocked(0.5) is False
    assert is_output_blocked(0.5, detect_only_action) is False


def test_boolean_semantics_are_pinned_on_both_sides():
    assert _candidate_is_blocked(True) is False
    assert is_output_blocked(True, detect_only_action) is False

    assert _candidate_is_blocked(False) is True
    assert is_output_blocked(False, detect_only_action) is True


@action(output_mapping=content_safety_check_output_mapping)
def content_safety_action(result):
    return result


def test_content_safety_mapping_characterization_for_p3_golden():
    assert is_output_blocked({"allowed": False}, content_safety_action) is True
    assert is_output_blocked({"allowed": True}, content_safety_action) is False


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (True, False),
        (False, True),
    ],
)
def test_self_check_output_mapping_matches_default_interpret(raw_return, expected_blocked):
    outcome_blocked = _candidate_is_blocked(raw_return)
    mapping_blocked = is_output_blocked(raw_return, self_check_output)

    assert outcome_blocked is expected_blocked
    assert mapping_blocked is expected_blocked
