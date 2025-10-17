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

from typing import Any, Dict, List, Union

import pytest

from nemoguardrails.rails.llm.utils import (
    get_action_details_from_flow_id,
    get_history_cache_key,
)


def test_basic():
    assert get_history_cache_key([]) == ""

    assert get_history_cache_key([{"role": "user", "content": "hi"}]) == "hi"

    assert (
        get_history_cache_key(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        )
        == "hi:Hello!:How are you?"
    )


def test_with_context():
    assert (
        get_history_cache_key(
            [
                {"role": "context", "content": {"user_name": "John"}},
                {"role": "user", "content": "hi"},
            ],
        )
        == '{"user_name": "John"}:hi'
    )

    assert (
        get_history_cache_key(
            [
                {"role": "context", "content": {"user_name": "John"}},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        )
        == '{"user_name": "John"}:hi:Hello!:How are you?'
    )


def test_multimodal_content():
    """Test get_history_cache_key with multimodal content (list-based content)."""
    multimodal_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."},
                },
            ],
        }
    ]
    assert get_history_cache_key(multimodal_messages) == "What's in this image?"

    multi_text_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "text", "text": "Second part"},
            ],
        }
    ]
    assert get_history_cache_key(multi_text_messages) == "First part Second part"

    mixed_content_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."},
                },
                {"type": "text", "text": "World"},
            ],
        }
    ]
    assert get_history_cache_key(mixed_content_messages) == "Hello World"

    empty_text_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."},
                },
            ],
        }
    ]
    assert get_history_cache_key(empty_text_messages) == ""


def test_get_action_details_from_flow_id_exact_match():
    """Test get_action_details_from_flow_id with exact flow ID match."""
    flows = [
        {
            "id": "test_flow",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "test.co",
                        "line_text": "execute action_name",
                    },
                    "action_name": "test_action",
                    "action_params": {"param1": "value1"},
                }
            ],
        }
    ]

    action_name, action_params = get_action_details_from_flow_id("test_flow", flows)
    assert action_name == "test_action"
    assert action_params == {"param1": "value1"}


def test_get_action_details_from_flow_id_content_safety():
    """Test get_action_details_from_flow_id  ."""
    flows = [
        {
            "id": "content safety check output",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "safety.co",
                        "line_text": "execute content_safety_check",
                    },
                    "action_name": "content_safety_check",
                    "action_params": {"model": "gpt-4"},
                }
            ],
        }
    ]

    action_name, action_params = get_action_details_from_flow_id(
        "content safety check output $model=anothe_model_config", flows
    )
    assert action_name == "content_safety_check"
    assert action_params == {"model": "gpt-4"}


def test_get_action_details_from_flow_id_topic_safety():
    """Test get_action_details_from_flow_id with topic safety."""
    flows = [
        {
            "id": "topic safety check output",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "topic.co",
                        "line_text": "execute topic_safety_check",
                    },
                    "action_name": "topic_safety_check",
                    "action_params": {"model": "claude"},
                }
            ],
        }
    ]

    action_name, action_params = get_action_details_from_flow_id(
        "topic safety check output $model=claude_model", flows
    )
    assert action_name == "topic_safety_check"
    assert action_params == {"model": "claude"}


def test_get_action_details_from_flow_id_no_match():
    """Test get_action_details_from_flow_id when no flow matches."""
    flows = [
        {
            "id": "different_flow",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "test.co",
                        "line_text": "execute test_action",
                    },
                    "action_name": "test_action",
                    "action_params": {},
                }
            ],
        }
    ]

    with pytest.raises(
        ValueError, match="No action found for flow_id: nonexistent_flow"
    ):
        get_action_details_from_flow_id("nonexistent_flow", flows)


def test_get_action_details_from_flow_id_no_run_action():
    """Test get_action_details_from_flow_id when flow has no run_action element."""
    flows = [
        {
            "id": "test_flow",
            "elements": [{"_type": "other_element", "some_data": "value"}],
        }
    ]

    with pytest.raises(
        ValueError, match="No run_action element found for flow_id: test_flow"
    ):
        get_action_details_from_flow_id("test_flow", flows)


def test_get_action_details_from_flow_id_invalid_run_action():
    """Test get_action_details_from_flow_id when run_action element is invalid."""
    flows = [
        {
            "id": "test_flow",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "test.py",
                        "line_text": "execute test_action",
                    },
                    "action_name": "test_action",
                    "action_params": {},
                }
            ],
        }
    ]

    with pytest.raises(
        ValueError, match="No run_action element found for flow_id: test_flow"
    ):
        get_action_details_from_flow_id("test_flow", flows)


def test_get_action_details_from_flow_id_multiple_run_actions():
    """Test get_action_details_from_flow_id with multiple run_action elements."""
    flows = [
        {
            "id": "multi_action_flow",
            "elements": [
                {"_type": "other_element", "data": "ignore"},
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "multi.co",
                        "line_text": "execute first_action",
                    },
                    "action_name": "first_action",
                    "action_params": {"order": "first"},
                },
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "multi.co",
                        "line_text": "execute second_action",
                    },
                    "action_name": "second_action",
                    "action_params": {"order": "second"},
                },
            ],
        }
    ]

    # Should return the first valid run_action element
    action_name, action_params = get_action_details_from_flow_id(
        "multi_action_flow", flows
    )
    assert action_name == "first_action"
    assert action_params == {"order": "first"}


@pytest.fixture
def dummy_flows() -> List[Union[Dict, Any]]:
    return [
        {
            "id": "test_flow",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "flows.v1.co",
                        "line_text": "execute something",
                    },
                    "action_name": "test_action",
                    "action_params": {"param1": "value1"},
                }
            ],
        },
        {
            "id": "other_flow is prefix",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "flows.v1.co",
                        "line_text": "execute something else",
                    },
                    "action_name": "other_action",
                    "action_params": {"param2": "value2"},
                }
            ],
        },
        {
            "id": "test_rails_co",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "rails.co",
                        "line_text": "execute something",
                    },
                    "action_name": "test_action_supported",
                    "action_params": {"param1": "value1"},
                }
            ],
        },
        {
            "id": "test_rails_co_v2",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "rails.co",
                        "line_text": "await something",
                    },
                    "action_name": "test_action_not_supported",
                    "action_params": {"param1": "value1"},
                }
            ],
        },
    ]


def test_get_action_details_exact_match(dummy_flows):
    action_name, action_params = get_action_details_from_flow_id(
        "test_flow", dummy_flows
    )
    assert action_name == "test_action"
    assert action_params == {"param1": "value1"}


def test_get_action_details_exact_match_any_co_file(dummy_flows):
    action_name, action_params = get_action_details_from_flow_id(
        "test_rails_co", dummy_flows
    )
    assert action_name == "test_action_supported"
    assert action_params == {"param1": "value1"}


def test_get_action_details_exact_match_not_colang_2(dummy_flows):
    with pytest.raises(ValueError) as exc_info:
        get_action_details_from_flow_id("test_rails_co_v2", dummy_flows)

    assert "No run_action element found for flow_id" in str(exc_info.value)


def test_get_action_details_no_match(dummy_flows):
    # Tests that a non matching flow_id raises a ValueError
    with pytest.raises(ValueError) as exc_info:
        get_action_details_from_flow_id("non_existing_flow", dummy_flows)
    assert "No action found for flow_id" in str(exc_info.value)
