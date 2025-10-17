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

"""Test the flows engine."""
from nemoguardrails.colang.v1_0.runtime.flows import (
    FlowConfig,
    State,
    compute_next_state,
)

# Flow configurations for these tests
FLOW_CONFIGS = {
    "greeting": FlowConfig(
        id="greeting",
        elements=[
            {"_type": "UserIntent", "intent_name": "express greeting"},
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "express greeting"},
            },
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "offer to help"},
            },
        ],
    ),
    "greeting follow up": FlowConfig(
        id="greeting follow up",
        is_extension=True,
        priority=2,
        elements=[
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "express greeting"},
            },
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "comment random fact about today"},
            },
        ],
    ),
}


def test_extension_flows_1():
    """Test a simple sequence of two turns in a flow."""
    state = State(context={}, flow_states=[], flow_configs=FLOW_CONFIGS)

    state = compute_next_state(
        state,
        {
            "type": "UserIntent",
            "intent": "express greeting",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "express greeting"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "express greeting",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "comment random fact about today"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "comment random fact about today",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "offer to help"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "offer to help",
        },
    )
    assert state.next_step is None
