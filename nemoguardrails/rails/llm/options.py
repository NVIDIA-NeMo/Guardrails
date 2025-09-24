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

"""Generation options give more control over the generation and the result.

For example, to run only the input rails::

    # Since everything is enabled by default, we disable explicitly the others
    options = {
        "rails": {
            "output": False,
            "dialog": False,
            "retrieval": False
        }
    }
    messages = [{
        "role": "user",
        "content": "Am I allowed to say this?"
    }]

    rails.generate(messages=messages, options=options)

To invoke only some specific input/output rails:

    rails.generate(messages=messages, options={
        "rails": {
            "input": ["check jailbreak"],
            "output": ["output moderation v2"]
        }
    })

To provide additional parameters to the main LLM call:

    rails.generate(messages=messages, options={
        "llm_params": {
            "temperature": 0.5
        }
    })

To return additional information from the generation (i.e., context variables):

    # This will include the relevant chunks in the returned response, as part
    # of the `output_data` field.
    rails.generate(messages=messages, options={
        "output_vars": ["relevant_chunks"]
    })

To skip enforcing the rails, and only inform the user if they were triggered:

    rails.generate(messages=messages, options={
        "enforce": False
    })

    # {..., log: {"triggered_rails": {"type": "input", "name": "check jailbreak"}}}

To get more details on the LLM calls that were executed, including the raw responses:

    rails.generate(messages=messages, options={
        "log": {
            "llm_calls": True
        }
    })

    # {..., log: {"llm_calls": [...]}}

"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator

from nemoguardrails.logging.explain import LLMCallInfo, LLMCallSummary


class GenerationLogOptions(BaseModel):
    """Options for what should be included in the generation log."""

    activated_rails: bool = Field(
        default=False,
        description="Include detailed information about the rails that were activated during generation.",
    )
    llm_calls: bool = Field(
        default=False,
        description="Include information about all the LLM calls that were made. "
        "This includes: prompt, completion, token usage, raw response, etc.",
    )
    internal_events: bool = Field(
        default=False,
        description="Include the array of internal generated events.",
    )
    colang_history: bool = Field(
        default=False,
        description="Include the history of the conversation in Colang format.",
    )


class GenerationRailsOptions(BaseModel):
    """Options for what rails should be used during the generation."""

    input: Union[bool, List[str]] = Field(
        default=True,
        description="Whether the input rails are enabled or not. "
        "If a list of names is specified, then only the specified input rails will be applied.",
    )
    output: Union[bool, List[str]] = Field(
        default=True,
        description="Whether the output rails are enabled or not. "
        "If a list of names is specified, then only the specified output rails will be applied.",
    )
    retrieval: Union[bool, List[str]] = Field(
        default=True,
        description="Whether the retrieval rails are enabled or not. "
        "If a list of names is specified, then only the specified retrieval rails will be applied.",
    )
    dialog: bool = Field(
        default=True,
        description="Whether the dialog rails are enabled or not.",
    )
    tool_output: Union[bool, List[str]] = Field(
        default=True,
        description="Whether the tool output rails are enabled or not. "
        "If a list of names is specified, then only the specified tool output rails will be applied.",
    )
    tool_input: Union[bool, List[str]] = Field(
        default=True,
        description="Whether the tool input rails are enabled or not. "
        "If a list of names is specified, then only the specified tool input rails will be applied.",
    )


class GenerationOptions(BaseModel):
    """A set of options that should be applied during a generation.

    The GenerationOptions control various things such as what rails are enabled,
    additional parameters for the main LLM, whether the rails should be enforced or
    ran in parallel, what to be included in the generation log, etc.
    """

    rails: GenerationRailsOptions = Field(
        default_factory=GenerationRailsOptions,
        description="Options for which rails should be applied for the generation. "
        "By default, all rails are enabled.",
    )
    llm_params: Optional[dict] = Field(
        default=None,
        description="Additional parameters that should be used for the LLM call",
    )
    llm_output: bool = Field(
        default=False,
        description="Whether the response should also include any custom LLM output.",
    )
    output_vars: Optional[Union[bool, List[str]]] = Field(
        default=None,
        description="Whether additional context information should be returned. "
        "When True is specified, the whole context is returned. "
        "Otherwise, a list of key names can be specified.",
    )
    # TODO: add support for this
    # enforce: Optional[bool] = Field(
    #     default=True,
    #     description="Whether the rails configuration should be enforced. "
    #     "When set to False, the raw LLM call is made in parallel with running the input rails "
    #     "on the user input. Also, the output rails are applied to the output of the raw",
    # )
    log: GenerationLogOptions = Field(
        default_factory=GenerationLogOptions,
        description="Options about what to include in the log. By default, nothing is included. ",
    )

    @root_validator(pre=True, allow_reuse=True)
    def check_fields(cls, values):
        # Translate the `rails` generation option from List[str] to dict.
        if "rails" in values and isinstance(values["rails"], list):
            _rails = {
                "input": False,
                "dialog": False,
                "retrieval": False,
                "output": False,
                "tool_output": False,
                "tool_input": False,
            }
            for rail_type in values["rails"]:
                _rails[rail_type] = True
            values["rails"] = _rails

        return values


class ExecutedAction(BaseModel):
    """Information about an action that was executed."""

    action_name: str = Field(description="The name of the action that was executed.")
    action_params: Dict[str, Any] = Field(
        default_factory=dict, description="The parameters for the action."
    )
    return_value: Any = Field(
        default=None, description="The value returned by the action."
    )
    llm_calls: List[LLMCallInfo] = Field(
        default_factory=list,
        description="Information about the LLM calls made by the action.",
    )
    started_at: Optional[float] = Field(
        default=None, description="Timestamp for when the action started."
    )
    finished_at: Optional[float] = Field(
        default=None, description="Timestamp for when the action finished."
    )
    duration: Optional[float] = Field(
        default=None, description="How long the action took to execute, in seconds."
    )


class ActivatedRail(BaseModel):
    """A rail that was activated during the generation."""

    type: str = Field(
        description="The type of the rail that was activated, e.g., input, output, dialog."
    )
    name: str = Field(
        description="The name of the rail, i.e., the name of the flow implementing the rail."
    )
    decisions: List[str] = Field(
        default_factory=list,
        description="A sequence of decisions made by the rail, e.g., 'bot refuse to respond', 'stop', 'continue'.",
    )
    executed_actions: List[ExecutedAction] = Field(
        default_factory=list, description="The list of actions executed by the rail."
    )
    stop: bool = Field(
        default=False,
        description="Whether the rail decided to stop any further processing.",
    )
    additional_info: Optional[dict] = Field(
        default=None, description="Additional information coming from rail."
    )
    started_at: Optional[float] = Field(
        default=None, description="Timestamp for when the rail started."
    )
    finished_at: Optional[float] = Field(
        default=None, description="Timestamp for when the rail finished."
    )
    duration: Optional[float] = Field(
        default=None,
        description="The duration in seconds for applying the rail. "
        "Some rails are applied instantly, e.g., dialog rails, so they don't have a duration.",
    )


class GenerationStats(BaseModel):
    """General stats about the generation."""

    input_rails_duration: Optional[float] = Field(
        default=None,
        description="The time in seconds spent in processing the input rails.",
    )
    dialog_rails_duration: Optional[float] = Field(
        default=None,
        description="The time in seconds spent in processing the dialog rails.",
    )
    generation_rails_duration: Optional[float] = Field(
        default=None,
        description="The time in seconds spent in generation rails.",
    )
    output_rails_duration: Optional[float] = Field(
        default=None,
        description="The time in seconds spent in processing the output rails.",
    )
    total_duration: Optional[float] = Field(
        default=None, description="The total time in seconds."
    )
    llm_calls_duration: Optional[float] = Field(
        default=0, description="The time in seconds spent in LLM calls."
    )
    llm_calls_count: Optional[int] = Field(
        default=0, description="The number of LLM calls in total."
    )
    llm_calls_total_prompt_tokens: Optional[int] = Field(
        default=0, description="The total number of prompt tokens."
    )
    llm_calls_total_completion_tokens: Optional[int] = Field(
        default=0, description="The total number of completion tokens."
    )
    llm_calls_total_tokens: Optional[int] = Field(
        default=0, description="The total number of tokens."
    )


class GenerationLog(BaseModel):
    """Contains additional logging information associated with a generation call."""

    activated_rails: List[ActivatedRail] = Field(
        default_factory=list,
        description="The list of rails that were activated during generation.",
    )
    stats: GenerationStats = Field(
        default_factory=GenerationStats,
        description="General stats about the generation process.",
    )
    llm_calls: Optional[List[LLMCallInfo]] = Field(
        default=None,
        description="The list of LLM calls that have been made to fulfill the generation request. ",
    )
    internal_events: Optional[List[dict]] = Field(
        default=None, description="The complete sequence of internal events generated."
    )
    colang_history: Optional[str] = Field(
        default=None, description="The Colang history associated with the generation."
    )

    def print_summary(self):
        print("\n# General stats\n")

        # Percent accounted so far
        pc = 0
        duration = 0

        print(f"- Total time: {self.stats.total_duration:.2f}s")
        if self.stats.input_rails_duration and self.stats.total_duration:
            _pc = round(
                100 * self.stats.input_rails_duration / self.stats.total_duration, 2
            )
            pc += _pc
            duration += self.stats.input_rails_duration

            print(f"  - [{self.stats.input_rails_duration:.2f}s][{_pc}%]: INPUT Rails")
        if self.stats.dialog_rails_duration and self.stats.total_duration:
            _pc = round(
                100 * self.stats.dialog_rails_duration / self.stats.total_duration, 2
            )
            pc += _pc
            duration += self.stats.dialog_rails_duration

            print(
                f"  - [{self.stats.dialog_rails_duration:.2f}s][{_pc}%]: DIALOG Rails"
            )
        if self.stats.generation_rails_duration and self.stats.total_duration:
            _pc = round(
                100 * self.stats.generation_rails_duration / self.stats.total_duration,
                2,
            )
            pc += _pc
            duration += self.stats.generation_rails_duration

            print(
                f"  - [{self.stats.generation_rails_duration:.2f}s][{_pc}%]: GENERATION Rails"
            )
        if self.stats.output_rails_duration and self.stats.total_duration:
            _pc = round(
                100 * self.stats.output_rails_duration / self.stats.total_duration, 2
            )
            pc += _pc
            duration += self.stats.output_rails_duration

            print(
                f"  - [{self.stats.output_rails_duration:.2f}s][{_pc}%]: OUTPUT Rails"
            )

        processing_overhead = (self.stats.total_duration or 0) - duration
        if processing_overhead >= 0.01:
            _pc = round(100 - pc, 2)
            print(f"  - [{processing_overhead:.2f}s][{_pc}%]: Processing overhead ")

        if self.stats.llm_calls_count:
            print(
                f"- {self.stats.llm_calls_count} LLM calls, "
                f"{self.stats.llm_calls_duration:.2f}s total duration, "
                f"{self.stats.llm_calls_total_prompt_tokens} total prompt tokens, "
                f"{self.stats.llm_calls_total_completion_tokens} total completion tokens, "
                f"{self.stats.llm_calls_total_tokens} total tokens."
            )

        print("\n# Detailed stats\n")
        for activated_rail in self.activated_rails:
            action_names = ", ".join(
                action.action_name for action in activated_rail.executed_actions
            )
            llm_calls_count = 0
            llm_calls_durations = []
            for action in activated_rail.executed_actions:
                llm_calls_count += len(action.llm_calls)
                llm_calls_durations.extend(
                    [
                        f"{round(llm_call.duration or 0, 2)}s"
                        for llm_call in action.llm_calls
                    ]
                )
            print(
                f"- [{activated_rail.duration:.2f}s] {activated_rail.type.upper()} ({activated_rail.name}): "
                f"{len(activated_rail.executed_actions)} actions ({action_names}), "
                f"{llm_calls_count} llm calls [{', '.join(llm_calls_durations)}]"
            )
        print("\n")


class GenerationResponse(BaseModel):
    # TODO: add typing for the list of messages
    response: Union[str, List[dict]] = Field(
        description="The list of the generated messages."
    )
    llm_output: Optional[dict] = Field(
        default=None, description="Contains any additional output coming from the LLM."
    )
    output_data: Optional[dict] = Field(
        default=None,
        description="The output data, i.e. a dict with the values corresponding to the `output_vars`.",
    )
    log: Optional[GenerationLog] = Field(
        default=None, description="Additional logging information."
    )
    state: Optional[dict] = Field(
        default=None,
        description="A state object which can be used in subsequent calls to continue the interaction.",
    )
    tool_calls: Optional[list] = Field(
        default=None,
        description="Tool calls extracted from the LLM response, if any.",
    )
    llm_metadata: Optional[dict] = Field(
        default=None,
        description="Metadata from the LLM response (additional_kwargs, response_metadata, usage_metadata, etc.)",
    )


if __name__ == "__main__":
    print(
        GenerationOptions(rails=GenerationRailsOptions(input=False))
    )  # pragma: no cover (Can't run as script for test coverage)
