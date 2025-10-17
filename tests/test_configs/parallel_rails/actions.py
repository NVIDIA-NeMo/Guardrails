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

import asyncio
from typing import Optional

from nemoguardrails.actions import action


@action(is_system_action=True)
async def check_blocked_input_terms(
    duration: float = 0.0, context: Optional[dict] = None
):
    user_message = context.get("user_message")

    # A quick hard-coded list of proprietary terms. You can also read this from a file.
    proprietary_terms = ["blocked term"]

    # Wait to simulate a delay in processing
    if isinstance(duration, str):
        duration = float(duration)
    await asyncio.sleep(duration)

    for term in proprietary_terms:
        if term.lower() in user_message.lower():
            return True

    return False


@action(is_system_action=True)
async def check_blocked_output_terms(
    duration: float = 0.0, context: Optional[dict] = None
):
    bot_response = context.get("bot_message")

    # A quick hard-coded list of proprietary terms. You can also read this from a file.
    proprietary_terms = ["blocked term"]

    # Wait to simulate a delay in processing
    if isinstance(duration, str):
        duration = float(duration)
    await asyncio.sleep(duration)

    for term in proprietary_terms:
        if term.lower() in bot_response.lower():
            return True

    return False
