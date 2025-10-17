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
            {"_type": "UserIntent", "intent_name": "ask capabilities"},
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "inform capabilities"},
            },
        ],
    ),
    "benefits": FlowConfig(
        id="benefits",
        elements=[
            {"_type": "UserIntent", "intent_name": "ask about benefits"},
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "respond about benefits"},
            },
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "ask if user happy"},
            },
        ],
    ),
    "math": FlowConfig(
        id="math",
        elements=[
            {"_type": "UserIntent", "intent_name": "ask math question"},
            {"_type": "run_action", "action_name": "wolfram alpha request"},
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "respond to math question"},
            },
            {
                "_type": "run_action",
                "action_name": "utter",
                "action_params": {"value": "ask if user happy"},
            },
        ],
    ),
}


def test_simple_sequence():
    """Test a simple sequence of two turns in a flow."""
    state = State(
        context={},
        flow_states=[],
        flow_configs=FLOW_CONFIGS,
    )

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
    assert state.next_step is None

    state = compute_next_state(
        state,
        {
            "type": "UserIntent",
            "intent": "ask capabilities",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "inform capabilities"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "inform capabilities",
        },
    )
    assert state.next_step is None


def test_not_able_to_start_a_flow():
    """No flow should be able to start."""
    state = State(context={}, flow_states=[], flow_configs=FLOW_CONFIGS)

    state = compute_next_state(
        state,
        {
            "type": "UserIntent",
            "intent": "ask capabilities",
        },
    )
    assert state.next_step is None


def test_two_consecutive_bot_messages():
    """Test a sequence of two bot messages."""
    state = State(context={}, flow_states=[], flow_configs=FLOW_CONFIGS)

    state = compute_next_state(
        state,
        {
            "type": "UserIntent",
            "intent": "ask about benefits",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "respond about benefits"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "respond about benefits",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "ask if user happy"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "ask if user happy",
        },
    )
    assert state.next_step is None


def test_action_execution():
    """Test a sequence of with an action execution."""
    state = State(context={}, flow_states=[], flow_configs=FLOW_CONFIGS)

    state = compute_next_state(
        state,
        {
            "type": "UserIntent",
            "intent": "ask math question",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "wolfram alpha request",
    }

    state = compute_next_state(
        state,
        {
            "type": "InternalSystemActionFinished",
            "action_name": "wolfram alpha request",
            "status": "success",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "respond to math question"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "respond to math question",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "ask if user happy"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "ask if user happy",
        },
    )
    assert state.next_step is None


def test_flow_interruption():
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
    assert state.next_step is None

    state = compute_next_state(
        state,
        {
            "type": "UserIntent",
            "intent": "ask about benefits",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "respond about benefits"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "respond about benefits",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "ask if user happy"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "ask if user happy",
        },
    )
    assert state.next_step is None

    state = compute_next_state(
        state,
        {
            "type": "UserIntent",
            "intent": "ask capabilities",
        },
    )
    assert state.next_step == {
        "_type": "run_action",
        "action_name": "utter",
        "action_params": {"value": "inform capabilities"},
    }

    state = compute_next_state(
        state,
        {
            "type": "BotIntent",
            "intent": "inform capabilities",
        },
    )
    assert state.next_step is None
