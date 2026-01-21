# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails.actions import action


@action(is_system_action=True)
async def check_safety_action(context: dict):
    user_message = context.get("user_message", "")

    unsafe_terms = ["unsafe", "dangerous", "harmful", "kill", "violence"]
    is_safe = not any(term in user_message.lower() for term in unsafe_terms)

    return is_safe


@action(is_system_action=True)
async def check_topic_action(context: dict):
    user_message = context.get("user_message", "")

    off_topic_terms = ["offtopic", "irrelevant", "unrelated", "stupid", "idiot"]
    is_on_topic = not any(term in user_message.lower() for term in off_topic_terms)

    return is_on_topic


@action(is_system_action=True)
async def check_with_context_update(context: dict):
    user_message = context.get("user_message", "")

    violation_count = context.get("violation_count", 0)
    context["violation_count"] = violation_count + 1

    blocked_terms = ["blocked", "forbidden"]
    is_allowed = not any(term in user_message.lower() for term in blocked_terms)

    return is_allowed


@action(is_system_action=True)
async def check_output_safety_action(context: dict):
    bot_message = context.get("bot_message", "")

    unsafe_terms = ["harmful", "dangerous", "unsafe", "violence"]
    is_safe = not any(term in bot_message.lower() for term in unsafe_terms)

    return is_safe


@action(is_system_action=True)
async def check_output_length_action(context: dict):
    bot_message = context.get("bot_message", "")

    max_length = 500
    is_valid = len(bot_message) <= max_length

    return is_valid
