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

import contextvars
from typing import List

from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    ExecutedAction,
    GenerationLog,
)

# The processing log for the current async stack
processing_log_var = contextvars.ContextVar("processing_log", default=None)


def compute_generation_log(processing_log: List[dict]) -> GenerationLog:
    """Computes the GenerationLog based on the processing log.

    The processing log is a raw sequence of all the relevant events.
    The generation log is a more structured, curated, version of it.
    """
    generation_log = GenerationLog()

    # The list of actions to ignore during the processing.
    ignored_actions = [
        "create_event",
        "run_input_rails_in_parallel",
        "run_output_rails_in_parallel",
        "run_flows_in_parallel",
    ]
    ignored_flows = [
        "process user input",
        "run input rails",
        "run dialog rails",
        "process bot message",
        "run output rails",
    ]
    generation_flows = [
        "generate bot message",
    ]

    # Timestamps used for computing general timing information.
    input_rails_started_at = None
    input_rails_finished_at = None
    output_rails_started_at = None
    output_rails_finished_at = None

    activated_rail = None
    executed_action = None
    last_timestamp = None

    for event in processing_log:
        last_timestamp = event["timestamp"]

        if event["type"] == "step":
            # If we don't have a type for the current rail, it means we're dealing with
            # a dialog rail.
            if activated_rail is None:
                # We ignore certain system flows
                if event["flow_id"] in ignored_flows:
                    continue

                activated_rail = ActivatedRail(
                    type=(
                        "dialog"
                        if event["flow_id"] not in generation_flows
                        else "generation"
                    ),
                    name=event["flow_id"],
                    started_at=event["timestamp"],
                )
                generation_log.activated_rails.append(activated_rail)

            # If we're dealing with a dialog rail, we check that the name still corresponds
            # otherwise we create a new rail.
            if (
                activated_rail.type == "dialog"
                and activated_rail.name != event["flow_id"]
            ):
                # We ignore certain system flows
                if event["flow_id"] in ignored_flows:
                    continue

                activated_rail = ActivatedRail(
                    type=(
                        "dialog"
                        if event["flow_id"] not in generation_flows
                        else "generation"
                    ),
                    name=event["flow_id"],
                    started_at=event["timestamp"],
                )
                generation_log.activated_rails.append(activated_rail)

            for step in event["next_steps"]:
                if step["type"] == "StartInternalSystemAction":
                    action_name = step["action_name"]
                    if action_name not in ignored_actions:
                        activated_rail.decisions.append(
                            f"execute {step['action_name']}"
                        )

                elif step["type"] == "BotIntent":
                    activated_rail.decisions.append(step["intent"])

        elif event["type"] == "event":
            event_data = event["data"]
            event_type = event_data["type"]

            if event_type == "StartInputRails":
                input_rails_started_at = event["timestamp"]

            elif event_type == "StartOutputRails":
                output_rails_started_at = event["timestamp"]

            elif event_type == "StartInputRail":
                activated_rail = ActivatedRail(
                    type="input",
                    name=event_data["flow_id"],
                    started_at=event["timestamp"],
                )
                generation_log.activated_rails.append(activated_rail)

            elif event_type == "StartOutputRail":
                activated_rail = ActivatedRail(
                    type="output",
                    name=event_data["flow_id"],
                    started_at=event["timestamp"],
                )
                generation_log.activated_rails.append(activated_rail)

            elif event_type == "StartInternalSystemAction":
                action_name = event_data["action_name"]
                if action_name in ignored_actions:
                    continue

                executed_action = ExecutedAction(
                    action_name=action_name,
                    action_params=event_data["action_params"],
                    started_at=event["timestamp"],
                )
                if activated_rail is not None:
                    activated_rail.executed_actions.append(executed_action)

            elif event_type == "InternalSystemActionFinished":
                action_name = event_data["action_name"]
                if action_name in ignored_actions:
                    continue

                if executed_action is not None:
                    executed_action.finished_at = event["timestamp"]
                    if (
                        executed_action.finished_at is not None
                        and executed_action.started_at is not None
                    ):
                        executed_action.duration = (
                            executed_action.finished_at - executed_action.started_at
                        )
                    executed_action.return_value = event_data["return_value"]
                executed_action = None

            elif event_type in ["InputRailFinished", "OutputRailFinished"]:
                if activated_rail is not None:
                    activated_rail.finished_at = event["timestamp"]
                    if (
                        activated_rail.finished_at is not None
                        and activated_rail.started_at is not None
                    ):
                        activated_rail.duration = (
                            activated_rail.finished_at - activated_rail.started_at
                        )
                activated_rail = None

            elif event_type == "InputRailsFinished":
                input_rails_finished_at = event["timestamp"]

            elif event_type == "OutputRailsFinished":
                output_rails_finished_at = event["timestamp"]

        elif event["type"] == "llm_call_info":
            if executed_action is not None:
                executed_action.llm_calls.append(event["data"])

    # If at the end of the processing we still have an active rail, it is because
    # we have hit a stop. In this case, we take the last timestamp as the timestamp for
    # finishing the rail.
    if activated_rail is not None:
        activated_rail.finished_at = last_timestamp
        if (
            activated_rail.finished_at is not None
            and activated_rail.started_at is not None
        ):
            activated_rail.duration = (
                activated_rail.finished_at - activated_rail.started_at
            )

        if activated_rail.type in ["input", "output"]:
            activated_rail.stop = True
            activated_rail.decisions.append("stop")

    # If we have input rails, we also record the general stats
    if input_rails_started_at:
        # If we don't have a timestamp for when the input rails have finished,
        # we record the last timestamp.
        if input_rails_finished_at is None:
            input_rails_finished_at = last_timestamp

        generation_log.stats.input_rails_duration = (
            input_rails_finished_at - input_rails_started_at
        )

    # For all the dialog/generation rails, we set the finished time and the duration based on
    # the rail right after.
    for i in range(len(generation_log.activated_rails) - 1):
        activated_rail = generation_log.activated_rails[i]

        if activated_rail.type in ["dialog", "generation"]:
            next_rail = generation_log.activated_rails[i + 1]
            activated_rail.finished_at = next_rail.started_at
            if (
                activated_rail.finished_at is not None
                and activated_rail.started_at is not None
            ):
                activated_rail.duration = (
                    activated_rail.finished_at - activated_rail.started_at
                )

    # If we have output rails, we also record the general stats
    if output_rails_started_at:
        # If we don't have a timestamp for when the output rails have finished,
        # we record the last timestamp.
        if output_rails_finished_at is None:
            output_rails_finished_at = last_timestamp

        generation_log.stats.output_rails_duration = (
            output_rails_finished_at - output_rails_started_at
        )

    # We also need to compute the stats for dialog rails and generation.
    # And the stats for the LLM calls.
    for activated_rail in generation_log.activated_rails:
        # TODO: figure out a cleaner way to do this.
        #  the generation should not be inside the `generate_user_intent`
        # If we have a dialog rail for `generate user intent` and it has an
        # LLM call with the task `general`, then we consider this as a generation rail.
        if activated_rail.name == "generate user intent":
            if len(activated_rail.executed_actions) == 1:
                executed_action = activated_rail.executed_actions[0]

                if (
                    len(executed_action.llm_calls) == 1
                    and executed_action.llm_calls[0].task == "general"
                ):
                    activated_rail.type = "generation"

        if activated_rail.type == "dialog" and activated_rail.duration:
            generation_log.stats.dialog_rails_duration = (
                generation_log.stats.dialog_rails_duration or 0
            ) + activated_rail.duration

        if activated_rail.type == "generation" and activated_rail.duration:
            generation_log.stats.generation_rails_duration = (
                generation_log.stats.generation_rails_duration or 0
            ) + activated_rail.duration

        for executed_action in activated_rail.executed_actions:
            for llm_call in executed_action.llm_calls:
                generation_log.stats.llm_calls_count = (
                    generation_log.stats.llm_calls_count or 0
                ) + 1
                generation_log.stats.llm_calls_duration = (
                    generation_log.stats.llm_calls_duration or 0
                ) + (llm_call.duration or 0)
                generation_log.stats.llm_calls_total_prompt_tokens = (
                    generation_log.stats.llm_calls_total_prompt_tokens or 0
                ) + (llm_call.prompt_tokens or 0)
                generation_log.stats.llm_calls_total_completion_tokens = (
                    generation_log.stats.llm_calls_total_completion_tokens or 0
                ) + (llm_call.completion_tokens or 0)
                generation_log.stats.llm_calls_total_tokens = (
                    generation_log.stats.llm_calls_total_tokens or 0
                ) + (llm_call.total_tokens or 0)

    generation_log.stats.total_duration = (
        processing_log[-1]["timestamp"] - processing_log[0]["timestamp"]
    )

    return generation_log
