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

import logging
from typing import Any, Dict, Optional

from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.utils import new_event_dict

log = logging.getLogger(__name__)


@action(is_system_action=True)
async def create_event(
    event: dict,
    context: Optional[dict] = None,
):
    """Creates an event for the bot based on the provided data.

    Args:
        event (dict): The input event data.
        context (Optional[dict]): The context for the action. Defaults to None.

    Returns:
        ActionResult: An action result containing the created event.
    """

    event_dict: Dict[str, Any] = new_event_dict(
        event["_type"], **{k: v for k, v in event.items() if k != "_type"}
    )

    # We add basic support for referring variables as values
    for k, v in event_dict.items():
        if isinstance(v, str) and v[0] == "$":
            event_dict[k] = context.get(v[1:], None) if context else None

    return ActionResult(events=[event_dict])
