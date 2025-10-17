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

import logging
from typing import Optional

from nemoguardrails.colang.v1_0.runtime.eval import eval_expression

log = logging.getLogger(__name__)


def slide(state: "State", flow_config: "FlowConfig", head: int) -> Optional[int]:
    """
    Try to slide a flow with the provided head.

    Sliding is the operation of moving through "non-matching" elements e.g., check,
    if, jump, expect, etc. It also includes calling sub-flows.

    Args:
        state (State): The current state of the dialog.
        flow_config (FlowConfig): The config of the flow that should be advanced.
        head (int): The current head.

    Returns:
        Optional[int]: The updated head after sliding. If the flow finishes, returns a negative value.

    Note:
        The convention is to return the last head multiplied by -1 when the flow finishes.
    """
    context = state.context

    if head is None:
        return None

    # The active label is the label that can be reached going backwards
    # only through sliding elements.
    active_label = None
    active_label_data = None

    # This might get called directly at the end of a flow, in which case
    # we put the prev_head on the last element.
    prev_head = head if head < len(flow_config.elements) else head - 1

    while True:
        # if we reached the end, we stop
        if head == len(flow_config.elements) or head < 0:
            # We make a convention to return the last head, multiplied by -1 when the flow finished
            return -1 * (prev_head + 1)

        prev_head = head
        pattern_item = flow_config.elements[head]

        # Updated the active label if needed
        if "_label" in pattern_item:
            active_label = pattern_item["_label"]
            active_label_data = pattern_item.get("_label_value", None)

        # We make sure the active label is propagated to all the other elements
        if active_label:
            pattern_item["_active_label"] = active_label
            pattern_item["_active_label_data"] = active_label_data

        p_type = pattern_item["_type"]

        # CHECK, IF, JUMP
        if p_type in ["check", "if", "jump"]:
            # for check and if, we need to evaluate the expression
            if p_type in ["check", "if"]:
                expr = pattern_item["expression"]
                check = eval_expression(expr, context)

                if p_type == "check":
                    if not check:
                        return None
                    else:
                        head += int(pattern_item.get("_next", 1))
                elif p_type == "if":
                    if check:
                        head += 1
                    else:
                        head += int(pattern_item["_next_else"])

            elif p_type == "jump":
                if not pattern_item.get("_absolute"):
                    head += int(pattern_item["_next"])
                else:
                    head = int(pattern_item["_next"])

        elif p_type in ["while"]:
            expr = pattern_item["expression"]
            check = eval_expression(expr, context)
            if check:
                head += int(pattern_item.get("_next", 1))
            else:
                head += int(pattern_item["_next_on_break"])

        # CONTINUE
        elif p_type == "continue":
            head += int(pattern_item.get("_next_on_continue", 1))

        # STOP
        elif p_type == "stop":
            return None

        # BREAK
        elif p_type == "break":
            head += int(pattern_item.get("_next_on_break", 1))

        # SET
        elif p_type == "set":
            value = eval_expression(pattern_item["expression"], context)

            # We transform tuples into arrays
            if isinstance(value, tuple):
                value = list(value)

            key_name = pattern_item["key"]

            # Update the context with the result of the expression and also record
            # the explicit update.
            context.update({key_name: value})
            state.context_updates.update({key_name: value})

            head += int(pattern_item.get("_next", 1))
        else:
            break

    # If we got this far, it means we had a match and the flow advanced
    return head
