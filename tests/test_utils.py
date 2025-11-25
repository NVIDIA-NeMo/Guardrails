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

from nemoguardrails.utils import compute_hash, new_event_dict, safe_eval


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
    e = new_event_dict(event_type, script=script, intensity=intensity, source_uid="my_uid")

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


@pytest.fixture(params=[AttributeError, ValueError])
def md5_not_available(request):
    with patch("hashlib.md5", side_effect=request.param):
        yield


def test_hash_without_md5(md5_not_available):
    hash_value = compute_hash("test")
    assert isinstance(hash_value, str)
    assert len(hash_value) == 64  # SHA256 hash is 64 characters long


def test_hash_with_md5():
    hash_value = compute_hash("test")
    assert isinstance(hash_value, str)
    assert len(hash_value) == 32  # MD5 hash is 32 characters long


@pytest.mark.asyncio
async def test_extract_error_json():
    """Test the extract_error_json function with different formats of error messages."""
    from nemoguardrails.utils import extract_error_json

    # Test standard format with dict
    error_message = "Error code: 401 - {'error': {'message': 'Incorrect API key provided', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"
    result = extract_error_json(error_message)
    assert result["error"]["message"] == "Incorrect API key provided"
    assert result["error"]["type"] == "invalid_request_error"
    assert result["error"]["code"] == "invalid_api_key"

    # Test with json format
    error_message = 'Error code: 401 - {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}'
    result = extract_error_json(error_message)
    assert result["error"]["message"] == "Incorrect API key provided"
    assert result["error"]["type"] == "invalid_request_error"
    assert result["error"]["code"] == "invalid_api_key"

    error_message = "Some generic error without JSON"
    result = extract_error_json(error_message)
    assert result["error"]["message"] == "Some generic error without JSON"

    # malformed json
    error_message = "Error code: 500 - {malformed json}"
    result = extract_error_json(error_message)
    assert "Invalid error " in result["error"]["message"]

    # dangerous AST expressions?
    error_message = "Error code: 500 - {'__complex__': '1j', 'message': 'This is potentially a malicious message? 1'}"
    result = extract_error_json(error_message)
    assert isinstance(result, dict)
    assert "error" in result
    assert "Invalid error format: Potentially unsafe" in result["error"]["message"]

    # None in error dict
    error_message = "Error code: 500 - {'error': {'message': 'Test message', 'param': None}}"
    result = extract_error_json(error_message)
    assert isinstance(result, dict)
    assert "error" in result
    assert result["error"]["message"] == "Test message"
    if "param" in result["error"]:
        assert result["error"]["param"] is None

    # very nested structure
    error_message = "Error code: 500 - {'error': {'nested': {'deeper': {'message': 'Too deep'}}}}"
    result = extract_error_json(error_message)
    assert "Invalid error format: Object too deeply" in result["error"]["message"]

    # large message
    large_message = "x" * 15000
    error_message = f"Error code: 500 - {{'error': {{'message': '{large_message}'}}}}"
    result = extract_error_json(error_message)
    assert len(result["error"]["message"]) <= 400 + len("... (truncated)")
    assert "... (truncated)" in result["error"]["message"]

    # list in errors
    error_message = "Error code: 500 - {'error': {'items': [1, 2, 3], 'message': 'List test'}}"
    result = extract_error_json(error_message)
    assert "deeply nested" in result["error"]["message"]

    # escaped characters
    error_message = 'Error code: 500 - {"error": {"message": "Line\\nbreak\\ttab"}}'
    result = extract_error_json(error_message)
    assert result["error"]["message"] == "Line\nbreak\ttab"

    # empty error object
    error_message = "Error code: 500 - {}"
    result = extract_error_json(error_message)
    assert result == {}

    # jnon string error message
    error_message = "Error code: 500 - {'error': {'message': 123}}"
    result = extract_error_json(error_message)
    assert result["error"]["message"] == 123

    # multiple error codes
    # we cannot parse it
    error_message = "Error code: 500 - Error code: 401 - {'error': {'message': 'Multiple codes'}}"
    result = extract_error_json(error_message)
    assert result["error"]["message"] == error_message
    with pytest.raises(KeyError):
        result["error"]["code"]
