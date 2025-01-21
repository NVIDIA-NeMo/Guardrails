# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails.utils import new_event_dict, safe_eval


def test_event_generation():
    event_type = "UserIntent"
    user_intent = "user greets bot"
    e = new_event_dict(event_type, intent=user_intent)

    assert "event_created_at" in e
    assert "source_uid" in e
    assert e["type"] == event_type
    assert e["intent"] == user_intent


def test_action_event_generation():
    event_type = "StartUtteranceBotAction"
    script = "Hello. Nice to see you!"
    intensity = 0.5
    e = new_event_dict(event_type, script=script, intensity=intensity)

    assert "event_created_at" in e
    assert "source_uid" in e
    assert e["type"] == event_type
    assert e["script"] == script
    assert e["intensity"] == intensity
    assert e["action_info_modality"] == "bot_speech"
    assert e["action_info_modality_policy"] == "replace"


def test_override_default_parameter():
    event_type = "StartUtteranceBotAction"
    script = "Hello. Nice to see you!"
    intensity = 0.5
    e = new_event_dict(
        event_type, script=script, intensity=intensity, source_uid="my_uid"
    )

    assert "event_created_at" in e
    assert "source_uid" in e
    assert e["source_uid"] == "my_uid"
    assert e["type"] == event_type
    assert e["script"] == script
    assert e["intensity"] == intensity
    assert e["action_info_modality"] == "bot_speech"
    assert e["action_info_modality_policy"] == "replace"


def test_action_finished_event():
    event_type = "UtteranceBotActionFinished"
    final_script = "Hello. Nice to see you!"
    e = new_event_dict(
        event_type,
        final_script=final_script,
        is_success=True,
        failure_reason="Nothing all worked.",
        action_uid="1234",
    )

    assert "action_finished_at" in e

    # Check that failure reason has been removed for a successful action
    assert "failure_reason" not in e

    # Check basic properties
    assert "event_created_at" in e
    assert "source_uid" in e
    assert e["type"] == event_type
    assert e["final_script"] == final_script
    assert e["action_info_modality"] == "bot_speech"
    assert e["action_info_modality_policy"] == "replace"


def test_start_action_event():
    event_type = "StartUtteranceBotAction"
    script = "Hello. Nice to see you!"
    e = new_event_dict(
        event_type,
        script=script,
    )

    assert "action_uid" in e
    assert e["script"] == script
    assert e["action_info_modality"] == "bot_speech"
    assert e["action_info_modality_policy"] == "replace"


def test_action_events_require_action_id():
    with pytest.raises(AssertionError, match=r".*action_uid.*"):
        event_type = "StopUtteranceBotAction"
        script = "Hello. Nice to see you!"
        e = new_event_dict(
            event_type,
            script=script,
        )


def test_wrong_property_type():
    with pytest.raises(AssertionError, match=r".*script.*"):
        event_type = "StartUtteranceBotAction"
        script = 1
        e = new_event_dict(
            event_type,
            script=script,
        )


@pytest.mark.parametrize(
    "input_value, expected_result",
    [
        ('"It\'s a sunny day"', "It's a sunny day"),  # double quotes with single quote
        (
            "\"He said, 'Hello'\"",
            "He said, 'Hello'",
        ),  # double quotes with nested single quote
        (
            "It's a sunny day",
            "It's a sunny day",
        ),  # unquoted string containing single quote
        (
            "It is a sunny day",
            "It is a sunny day",
        ),  # plain string not wrapped in quotes
        ("", ""),  # empty string
    ],
)
def test_safe_eval(input_value, expected_result):
    """Test safe_eval with various input values."""
    result = safe_eval(input_value)
    assert result == expected_result
