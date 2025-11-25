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
import functools
from dataclasses import is_dataclass
from time import time
from typing import Any

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.colang.v2_x.runtime.flows import Action, State
from nemoguardrails.colang.v2_x.runtime.serialization import (
    json_to_state,
    state_to_json,
)
from nemoguardrails.utils import console, new_event_dict

config = RailsConfig.from_content(
    """
    import core

    flow main
      user said "hi"
      bot say "Hello!"
      user said "hi"
      bot say "Hello again!"
    """,
    """
    colang_version: "2.x"
    """,
)


def check_equal_objects(o1: Any, o2: Any, path: str):
    """Helper to compare two objects.

    For primitive types, it will check the values are equal.
    For non-primitive types, it will check the type and for specific types, it will
    go deeper based on the type.
    """
    o1_type = type(o1)
    o2_type = type(o2)

    if o1_type != o2_type:
        console.print(
            f"Found different types ([bold][red]{o1_type.__name__}[/][/] vs [bold][red]{o2_type.__name__}[/][/]) for: {path}"
        )
        raise ValueError(f"Found different types at path: {path}")

    if is_dataclass(o1):
        for field in o1.__dataclass_fields__.keys():
            o1_value = getattr(o1, field)
            o2_value = getattr(o2, field)

            check_equal_objects(o1_value, o2_value, f"{path}.{field}")
    elif isinstance(o1, list):
        for i in range(len(o1)):
            check_equal_objects(o1[i], o2[i], f"{path}[{i}]")
    elif isinstance(o1, dict):
        for k, v in o1.items():
            check_equal_objects(o1[k], o2[k], f"{path}['{k}']")
    elif isinstance(o1, functools.partial) and isinstance(o2, functools.partial):
        # we don't compare the functions
        return
    elif isinstance(o1, Action):
        # we don't compare actions
        return
    elif isinstance(o1, RailsConfig):
        # we don't compare the rails config
        return
    else:
        if o1 != o2:
            print(f"Found different values ({str(o1)[0:10]} vs {str(o2)[0:10]}) for: {path}")
            raise ValueError(f"Found different values in path: {path}")


@pytest.mark.asyncio
async def test_serialization():
    rails = LLMRails(config=config, verbose=True)

    input_events = [
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "hi",
        }
    ]

    output_events, state = await rails.runtime.process_events(events=input_events, state={}, blocking=True)

    assert isinstance(state, State)
    assert output_events[0]["script"] == "Hello!"

    avg_time = 0
    number_of_runs = 10
    for i in range(0, number_of_runs + 1):
        t0 = time()
        s = state_to_json(state)
        took = time() - t0
        if i == 0:
            # Skip warm-up run
            continue
        avg_time += took
    avg_time /= number_of_runs

    assert avg_time < 0.2

    assert isinstance(s, str)

    state_2 = json_to_state(s)

    assert isinstance(state_2, State)

    check_equal_objects(state, state_2, "state")

    input_events = []
    for event in output_events:
        if event["type"] == "StartUtteranceBotAction":
            input_events.append(
                new_event_dict(
                    "UtteranceBotActionFinished",
                    action_uid=event["action_uid"],
                    is_success=True,
                    final_script=event["script"],
                )
            )
    input_events.append(
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "hi",
        }
    )

    output_events, state_3 = await rails.runtime.process_events(events=input_events, state=state_2, blocking=True)

    assert output_events[0]["script"] == "Hello again!"


if __name__ == "__main__":
    asyncio.run(test_serialization())
