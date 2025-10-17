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
import json
from typing import Any, Dict, List, Tuple, Union

from nemoguardrails.colang.v1_0.runtime.flows import _normalize_flow_id


def get_history_cache_key(messages: List[dict]) -> str:
    """Compute the cache key for a sequence of messages.

    Args:
        messages: The list of messages.

    Returns:
        A unique string that can be used as a key for the provided sequence of messages.
    """
    if len(messages) == 0:
        return ""

    key_items = []

    for msg in messages:
        if msg["role"] == "user":
            # Check if content is a string or a list (multimodal content)
            if isinstance(msg["content"], list):
                # For multimodal content, join all text parts
                text_parts = []
                for item in msg["content"]:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                key_items.append(" ".join(text_parts))
            else:
                # Use the content directly without json.dumps
                key_items.append(msg["content"])
        elif msg["role"] == "assistant":
            key_items.append(msg["content"])
        elif msg["role"] == "context":
            key_items.append(json.dumps(msg["content"]))
        elif msg["role"] == "event":
            key_items.append(json.dumps(msg["event"]))

    # Ensure all items in key_items are strings
    key_items = [str(item) if not isinstance(item, str) else item for item in key_items]

    history_cache_key = ":".join(key_items)

    return history_cache_key


def get_action_details_from_flow_id(
    flow_id: str,
    flows: List[Union[Dict, Any]],
) -> Tuple[str, Any]:
    """Get the action name and parameters from the flow id.

    First, try to find an exact match.
    If not found, then if the provided flow_id starts with one of the special prefixes,
    return the first flow whose id starts with that same prefix.
    """

    candidate_flow = None

    normalized_flow_id = _normalize_flow_id(flow_id)

    for flow in flows:
        # If exact match, use it
        if flow["id"] == normalized_flow_id:
            candidate_flow = flow

        if candidate_flow is not None:
            break

    if candidate_flow is None:
        raise ValueError(f"No action found for flow_id: {flow_id}")

    # we have identified a candidate, look for the run_action element.
    for element in candidate_flow["elements"]:
        if (
            element["_type"] == "run_action"
            and element["_source_mapping"]["filename"].endswith(".co")
            and "execute" in element["_source_mapping"]["line_text"]
            and "action_name" in element
        ):
            return element["action_name"], element["action_params"]

    raise ValueError(f"No run_action element found for flow_id: {flow_id}")
