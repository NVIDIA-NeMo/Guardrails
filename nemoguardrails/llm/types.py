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

from enum import Enum


class Task(Enum):
    """The various tasks that can be performed by the LLM."""

    # Core LLM tasks
    GENERAL = "general"
    GENERATE_USER_INTENT = "generate_user_intent"
    GENERATE_NEXT_STEPS = "generate_next_steps"
    GENERATE_BOT_MESSAGE = "generate_bot_message"
    GENERATE_INTENT_STEPS_MESSAGE = "generate_intent_steps_message"

    GENERATE_VALUE = "generate_value"
    GENERATE_VALUE_FROM_INSTRUCTION = "generate_value_from_instruction"
    GENERATE_USER_INTENT_FROM_USER_ACTION = "generate_user_intent_from_user_action"
    GENERATE_USER_INTENT_AND_BOT_ACTION_FROM_USER_ACTION = "generate_user_intent_and_bot_action_from_user_action"
    GENERATE_FLOW_FROM_INSTRUCTIONS = "generate_flow_from_instructions"
    GENERATE_FLOW_FROM_NAME = "generate_flow_from_name"
    GENERATE_FLOW_CONTINUATION = "generate_flow_continuation"
    GENERATE_FLOW_CONTINUATION_FROM_NLD = "generate_flow_continuation_from_flow_nld"

    # Tasks for various rails
    SELF_CHECK_INPUT = "self_check_input"
    SELF_CHECK_OUTPUT = "self_check_output"
    LLAMA_GUARD_CHECK_INPUT = "llama_guard_check_input"
    LLAMA_GUARD_CHECK_OUTPUT = "llama_guard_check_output"
    PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION = "patronus_lynx_check_output_hallucination"

    SELF_CHECK_FACTS = "self_check_facts"
    SELF_CHECK_HALLUCINATION = "self_check_hallucination"
