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

"""A set of actions for generating various types of completions using an LLMs."""

import logging
import re
import textwrap
from ast import literal_eval
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from rich.text import Text

from nemoguardrails.actions.actions import action
from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.actions.llm.utils import (
    escape_flow_name,
    get_first_bot_action,
    get_first_bot_intent,
    get_first_nonempty_line,
    get_first_user_intent,
    get_last_user_utterance_event_v2_x,
    llm_call,
    remove_action_intent_identifiers,
)
from nemoguardrails.colang.v2_x.lang.colang_ast import Flow, Spec, SpecOp
from nemoguardrails.colang.v2_x.runtime.errors import LlmResponseError
from nemoguardrails.colang.v2_x.runtime.flows import ActionEvent, InternalEvent
from nemoguardrails.colang.v2_x.runtime.statemachine import (
    Event,
    InternalEvents,
    State,
    find_all_active_event_matchers,
    get_element_from_head,
    get_event_from_element,
)
from nemoguardrails.context import (
    generation_options_var,
    llm_call_info_var,
    raw_llm_request,
    streaming_handler_var,
)
from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem
from nemoguardrails.llm.filters import colang
from nemoguardrails.llm.types import Task
from nemoguardrails.logging import verbose
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.utils import console, new_uuid

log = logging.getLogger(__name__)


def _remove_leading_empty_lines(s: str) -> str:
    """Remove the leading empty lines if they exist.

    A line is considered empty if it has only white spaces.
    """
    lines = s.split("\n")
    while lines and lines[0].strip() == "":
        lines = lines[1:]
    return "\n".join(lines)


class LLMGenerationActionsV2dotx(LLMGenerationActions):
    """Adapted version of LLMGenerationActions for Colang 2.x.

    It overrides some methods.
    """

    async def _init_colang_flows_index(
        self, flows: List[str]
    ) -> Optional[EmbeddingsIndex]:
        """Initialize an index with colang flows.

        The flows are expected to have full definition.

        Args
            flows: The list of flows, i.e. the flow definition from the source code.

        Returns
            An initialized index.
        """
        items = []
        for source_code in flows:
            items.append(IndexItem(text=source_code, meta={"flow": source_code}))

        # If we have no patterns, we stop.
        if len(items) == 0:
            return None

        flows_index = self.get_embedding_search_provider_instance(
            self.config.core.embedding_search_provider
        )
        await flows_index.add_items(items)
        await flows_index.build()

        return flows_index

    async def _init_flows_index(self) -> None:
        """Initializes the index of flows."""

        if not self.config.flows:
            return

        # The list of all flows that will be added to the index
        all_flows = []

        # The list of flows that have instructions, i.e. docstring at the beginning.
        instruction_flows = []
        for flow in self.config.flows:
            # RailsConfig flow can be either Dict or Flow. Convert dicts to Flow for rest of the function
            typed_flow: Flow = (
                Flow(**cast(Dict, flow)) if isinstance(flow, Dict) else flow
            )
            colang_flow = typed_flow.source_code
            if colang_flow:
                # Check if we need to exclude this flow.

                has_llm_exclude_parameter: bool = any(
                    [
                        "llm_exclude" in decorator.parameters
                        for decorator in typed_flow.decorators
                    ]
                )
                if typed_flow.file_info.get("exclude_from_llm") or (
                    "meta" in typed_flow.decorators and has_llm_exclude_parameter
                ):
                    continue

                all_flows.append(colang_flow)

                # If the first line is a comment, we consider it to be an instruction
                lines = colang_flow.split("\n")
                if len(lines) > 1:
                    first_line = lines[1].strip()
                    if first_line.startswith("#") or first_line.startswith('"""'):
                        instruction_flows.append(colang_flow)

        self.flows_index = await self._init_colang_flows_index(all_flows)
        self.instruction_flows_index = await self._init_colang_flows_index(
            instruction_flows
        )

        # If we don't have an instruction_flows_index, we fall back to using the main one
        if self.instruction_flows_index is None:
            self.instruction_flows_index = self.flows_index

    async def _collect_user_intent_and_examples(
        self, state: State, user_action: str, max_example_flows: int
    ) -> Tuple[List[str], str, bool]:
        # We search for the most relevant similar user intents
        examples = ""
        potential_user_intents = []
        is_embedding_only = False

        if self.user_message_index:
            threshold = None

            if self.config.rails.dialog.user_messages:
                threshold = (
                    self.config.rails.dialog.user_messages.embeddings_only_similarity_threshold
                )

            results = await self.user_message_index.search(
                text=user_action, max_results=max_example_flows, threshold=threshold
            )

            if self.config.rails.dialog.user_messages.embeddings_only:
                if results:
                    intent = results[0].meta["intent"]
                    potential_user_intents.append(intent)
                    is_embedding_only = True

                elif (
                    self.config.rails.dialog.user_messages.embeddings_only_fallback_intent
                ):
                    intent = (
                        self.config.rails.dialog.user_messages.embeddings_only_fallback_intent
                    )
                    potential_user_intents.append(intent)
                    is_embedding_only = True
                else:
                    # If we don't have a fallback intent, we run the search again
                    results = await self.user_message_index.search(
                        text=user_action, max_results=max_example_flows, threshold=None
                    )

                    is_embedding_only = False

            # We add these in reverse order so the most relevant is towards the end.
            for result in reversed(results):
                examples += f'user action: user said "{result.text}"\nuser intent: {result.meta["intent"]}\n\n'
                if result.meta["intent"] not in potential_user_intents:
                    potential_user_intents.append(result.meta["intent"])

        # We add all currently active user intents (heads on match statements)
        heads = find_all_active_event_matchers(state)
        for head in heads:
            el = get_element_from_head(state, head)
            element = el if isinstance(el, SpecOp) else SpecOp(**cast(Dict, el))
            flow_state = state.flow_states[head.flow_state_uid]
            event = get_event_from_element(state, flow_state, element)
            if (
                event.name == InternalEvents.FLOW_FINISHED
                and "flow_id" in event.arguments
            ):
                flow_id = event.arguments["flow_id"]
                if not isinstance(flow_id, str):
                    continue

                flow_config = state.flow_configs.get(flow_id, None)
                if flow_config and flow_id in state.flow_id_states:
                    element_flow_state_instance = state.flow_id_states[flow_id]
                    if flow_config.has_meta_tag("user_intent") or (
                        element_flow_state_instance
                        and "_user_intent" in element_flow_state_instance[0].context
                    ):
                        if flow_config.elements[1]["_type"] == "doc_string_stmt":
                            # TODO! Need to make this type-safe but no idea what's going on
                            examples += "user action: <" + (
                                flow_config.elements[1]["elements"][  # pyright: ignore
                                    0
                                ]["elements"][0]["elements"][0][3:-3]
                                + ">\n"
                            )
                            examples += f"user intent: {flow_id}\n\n"
                        elif flow_id not in potential_user_intents:
                            examples += f"user intent: {flow_id}\n\n"
                            potential_user_intents.append(flow_id)
                else:
                    # User intents that have no actual instance but only are expected through a match statement
                    examples += f"user intent: {flow_id}\n\n"
                    potential_user_intents.append(flow_id)

        examples = examples.strip("\n")

        return potential_user_intents, examples, is_embedding_only

    @action(name="GetLastUserMessageAction", is_system_action=True)
    async def get_last_user_message(
        self, events: List[dict], llm: Optional[BaseLLM] = None
    ) -> str:
        event = get_last_user_utterance_event_v2_x(events)
        assert event and event["type"] == "UtteranceUserActionFinished"
        return event["final_transcript"]

    @action(name="GenerateUserIntentAction", is_system_action=True, execute_async=True)
    async def generate_user_intent(  # pyright: ignore (TODO - Signature completely different to base class)
        self,
        state: State,
        events: List[dict],
        user_action: str,
        max_example_flows: int = 5,
        llm: Optional[BaseLLM] = None,
    ) -> str:
        """Generate the canonical form for what the user said i.e. user intent."""

        # Use action specific llm if registered else fallback to main llm
        generation_llm: Optional[Union[BaseLLM, BaseChatModel]] = (
            llm if llm else self.llm
        )

        log.info("Phase 1 :: Generating user intent")
        (
            potential_user_intents,
            examples,
            is_embedding_only,
        ) = await self._collect_user_intent_and_examples(
            state, user_action, max_example_flows
        )
        if is_embedding_only:
            return f"{potential_user_intents[0]}"

        llm_call_info_var.set(
            LLMCallInfo(task=Task.GENERATE_USER_INTENT_FROM_USER_ACTION.value)
        )

        prompt = self.llm_task_manager.render_task_prompt(
            task=Task.GENERATE_USER_INTENT_FROM_USER_ACTION,
            events=events,
            context={
                "examples": examples,
                "potential_user_intents": ", ".join(potential_user_intents),
                "user_action": user_action,
                "context": state.context,
            },
        )
        stop = self.llm_task_manager.get_stop_tokens(
            Task.GENERATE_USER_INTENT_FROM_USER_ACTION
        )

        # We make this call with lowest temperature to have it as deterministic as possible.
        result = await llm_call(
            generation_llm,
            prompt,
            stop=stop,
            llm_params={"temperature": self.config.lowest_temperature},
        )

        # Parse the output using the associated parser
        result = self.llm_task_manager.parse_task_output(
            Task.GENERATE_USER_INTENT_FROM_USER_ACTION, output=result
        )

        user_intent = get_first_nonempty_line(result)
        # GTP-4o often adds 'user intent: ' in front
        if user_intent and ":" in user_intent:
            temp_user_intent = get_first_user_intent([user_intent])
            if temp_user_intent:
                user_intent = temp_user_intent
            else:
                user_intent = None
        if user_intent is None:
            user_intent = "user was unclear"

        user_intent = escape_flow_name(user_intent.strip(" "))

        log.info(
            "Canonical form for user intent: %s", user_intent if user_intent else "None"
        )

        return f"{user_intent}" or "user unknown intent"

    @action(
        name="GenerateUserIntentAndBotAction",
        is_system_action=True,
        execute_async=True,
    )
    async def generate_user_intent_and_bot_action(
        self,
        state: State,
        events: List[dict],
        user_action: str,
        max_example_flows: int = 5,
        llm: Optional[BaseLLM] = None,
    ) -> dict:
        """Generate the canonical form for what the user said i.e. user intent and a suitable bot action."""

        # Use action specific llm if registered else fallback to main llm
        generation_llm: Optional[Union[BaseLLM, BaseChatModel]] = (
            llm if llm else self.llm
        )
        log.info("Phase 1 :: Generating user intent and bot action")

        (
            potential_user_intents,
            examples,
            is_embedding_only,
        ) = await self._collect_user_intent_and_examples(
            state, user_action, max_example_flows
        )

        llm_call_info_var.set(
            LLMCallInfo(
                task=Task.GENERATE_USER_INTENT_AND_BOT_ACTION_FROM_USER_ACTION.value
            )
        )

        prompt = self.llm_task_manager.render_task_prompt(
            task=Task.GENERATE_USER_INTENT_AND_BOT_ACTION_FROM_USER_ACTION,
            events=events,
            context={
                "examples": examples,
                "potential_user_intents": ", ".join(potential_user_intents),
                "user_action": user_action,
                "context": state.context,
            },
        )
        stop = self.llm_task_manager.get_stop_tokens(
            Task.GENERATE_USER_INTENT_AND_BOT_ACTION_FROM_USER_ACTION
        )

        # We make this call with lowest temperature to have it as deterministic as possible.
        result = await llm_call(
            generation_llm,
            prompt,
            stop=stop,
            llm_params={"temperature": self.config.lowest_temperature},
        )

        # Parse the output using the associated parser
        result = self.llm_task_manager.parse_task_output(
            Task.GENERATE_USER_INTENT_AND_BOT_ACTION_FROM_USER_ACTION, output=result
        )

        user_intent = get_first_nonempty_line(result)

        if user_intent and ":" in user_intent:
            temp_user_intent = get_first_user_intent([user_intent])
            if temp_user_intent:
                user_intent = temp_user_intent
            else:
                user_intent = None
        if user_intent is None:
            user_intent = "user was unclear"

        bot_intent = get_first_bot_intent(result.splitlines())
        bot_action = get_first_bot_action(result.splitlines())

        if bot_action is None:
            raise LlmResponseError(f"Issue with LLM response: {result}")

        user_intent = escape_flow_name(user_intent.strip(" "))

        if bot_intent:
            bot_intent = escape_flow_name(bot_intent.strip(" "))

        log.info(
            "Canonical form for user intent: %s", user_intent if user_intent else "None"
        )

        return {
            "user_intent": user_intent,
            "bot_intent": bot_intent,
            "bot_action": bot_action,
        }

    @action(name="PassthroughLLMAction", is_system_action=True, execute_async=True)
    async def passthrough_llm_action(
        self,
        user_message: str,
        state: State,
        events: List[dict],
        llm: Optional[BaseLLM] = None,
    ):
        if not llm:
            raise RuntimeError("No LLM provided to passthrough LLM Action")

        event = get_last_user_utterance_event_v2_x(events)
        if not event:
            raise RuntimeError(
                "Passthrough LLM Action couldn't find last user utterance"
            )

        # We check if we have a raw request. If the guardrails API is using
        # the `generate_events` API, this will not be set.
        raw_prompt = raw_llm_request.get()

        if raw_prompt is None:
            prompt = event["final_transcript"]
        else:
            if isinstance(raw_prompt, str):
                # If we're in completion mode, we use directly the last $user_message
                # as it may have been altered by the input rails.
                prompt = event["final_transcript"]
            elif isinstance(raw_prompt, list):
                prompt = raw_prompt.copy()

                # In this case, if the last message is from the user, we replace the text
                # just in case the input rails may have altered it.
                if prompt[-1]["role"] == "user":
                    raw_prompt[-1]["content"] = event["final_transcript"]
            else:
                raise ValueError(f"Unsupported type for raw prompt: {type(raw_prompt)}")

        # Initialize the LLMCallInfo object
        llm_call_info_var.set(LLMCallInfo(task=Task.GENERAL.value))

        generation_options: Optional[GenerationOptions] = generation_options_var.get()

        streaming_handler: Optional[StreamingHandler] = streaming_handler_var.get()
        custom_callback_handlers = [streaming_handler] if streaming_handler else None

        generation_llm_params = generation_options and generation_options.llm_params
        text = await llm_call(
            llm,
            user_message,
            custom_callback_handlers=custom_callback_handlers,
            llm_params=generation_llm_params,
        )

        text = self.llm_task_manager.parse_task_output(Task.GENERAL, output=text)

        return text

    @action(name="CheckValidFlowExistsAction", is_system_action=True)
    async def check_if_flow_exists(self, state: "State", flow_id: str) -> bool:
        """Return True if a flow with the provided flow_id exists."""
        return flow_id in state.flow_id_states

    @action(name="CheckFlowDefinedAction", is_system_action=True)
    async def check_if_flow_defined(self, state: "State", flow_id: str) -> bool:
        """Return True if a flow with the provided flow_id is defined."""
        return flow_id in state.flow_configs

    @action(name="CheckForActiveEventMatchAction", is_system_action=True)
    async def check_for_active_flow_finished_match(
        self, state: "State", event_name: str, **arguments: Any
    ) -> bool:
        """Return True if there is a flow waiting for the provided event name and parameters."""
        event: Event
        if event_name in InternalEvents.ALL:
            event = InternalEvent(name=event_name, arguments=arguments)
        elif "Action" in event_name:
            event = ActionEvent(name=event_name, arguments=arguments)
        else:
            event = Event(name=event_name, arguments=arguments)
        heads = find_all_active_event_matchers(state, event)
        return len(heads) > 0

    @action(
        name="GenerateFlowFromInstructionsAction",
        is_system_action=True,
        execute_async=True,
    )
    async def generate_flow_from_instructions(
        self,
        state: State,
        instructions: str,
        events: List[dict],
        llm: Optional[BaseLLM] = None,
    ) -> dict:
        """Generate a flow from the provided instructions."""

        if self.instruction_flows_index is None:
            raise RuntimeError("No instruction flows index has been created.")

        # Use action specific llm if registered else fallback to main llm
        generation_llm: Optional[Union[BaseLLM, BaseChatModel]] = (
            llm if llm else self.llm
        )
        log.info("Generating flow for instructions: %s", instructions)

        results = await self.instruction_flows_index.search(
            text=instructions, max_results=5, threshold=None
        )

        examples = ""
        for result in reversed(results):
            examples += f"{result.meta['flow']}\n"

        flow_id = new_uuid()[0:4]
        flow_name = f"dynamic_{flow_id}"

        llm_call_info_var.set(
            LLMCallInfo(task=Task.GENERATE_FLOW_FROM_INSTRUCTIONS.value)
        )

        prompt = self.llm_task_manager.render_task_prompt(
            task=Task.GENERATE_FLOW_FROM_INSTRUCTIONS,
            events=events,
            context={
                "examples": examples,
                "flow_name": flow_name,
                "instructions": instructions,
                "context": state.context,
            },
        )

        # We make this call with temperature 0 to have it as deterministic as possible.
        result = await llm_call(
            generation_llm,
            prompt,
            llm_params={"temperature": self.config.lowest_temperature},
        )

        result = self.llm_task_manager.parse_task_output(
            task=Task.GENERATE_FLOW_FROM_INSTRUCTIONS, output=result
        )

        # TODO: why this is not part of a filter or output_parser?
        #
        lines = _remove_leading_empty_lines(result).split("\n")

        if lines[0].startswith("  "):
            # print(f"Generated flow:\n{result}\n")
            return {
                "name": flow_name,
                "body": f"flow {flow_name}\n" + "\n".join(lines),
            }
        else:
            response = "\n".join(lines)
            log.warning(
                "GenerateFlowFromInstructionsAction\nFAILING-PROMPT ::\n%s\n FAILING-RESPONSE: %s\n",
                prompt,
                response,
            )
            return {
                "name": "bot inform LLM issue",
                "body": 'flow bot inform LLM issue\n  bot say "Sorry! There was an issue in the LLM result form GenerateFlowFromInstructionsAction!"',
            }

    @action(
        name="GenerateFlowFromNameAction", is_system_action=True, execute_async=True
    )
    async def generate_flow_from_name(
        self,
        state: State,
        name: str,
        events: List[dict],
        llm: Optional[BaseLLM] = None,
    ) -> str:
        """Generate a flow from the provided NAME."""

        if self.flows_index is None:
            raise RuntimeError("No flows index has been created.")

        # Use action specific llm if registered else fallback to main llm
        generation_llm: Optional[Union[BaseLLM, BaseChatModel]] = (
            llm if llm else self.llm
        )
        log.info("Generating flow for name: {name}")

        if not self.instruction_flows_index:
            raise RuntimeError("No instruction flows index has been created.")

        results = await self.instruction_flows_index.search(
            text=f"flow {name}", max_results=5, threshold=None
        )

        examples = ""
        for result in reversed(results):
            examples += f"{result.meta['flow']}\n"

        llm_call_info_var.set(LLMCallInfo(task=Task.GENERATE_FLOW_FROM_NAME.value))

        prompt = self.llm_task_manager.render_task_prompt(
            task=Task.GENERATE_FLOW_FROM_NAME,
            events=events,
            context={
                "examples": examples,
                "flow_name": name,
                "context": state.context,
            },
        )

        stop = self.llm_task_manager.get_stop_tokens(Task.GENERATE_FLOW_FROM_NAME)

        # We make this call with temperature 0 to have it as deterministic as possible.
        result = await llm_call(
            generation_llm,
            prompt,
            stop=stop,
            llm_params={"temperature": self.config.lowest_temperature},
        )

        result = self.llm_task_manager.parse_task_output(
            task=Task.GENERATE_FLOW_FROM_NAME, output=result
        )

        lines = _remove_leading_empty_lines(result).split("\n")

        if lines[0].startswith("flow"):
            return f"flow {lines[0][5:]}\n" + "\n".join(lines[1:])
        else:
            return f"flow {name}\n  " + "\n  ".join([line.lstrip() for line in lines])

    @action(
        name="GenerateFlowContinuationAction", is_system_action=True, execute_async=True
    )
    async def generate_flow_continuation(
        self,
        state: State,
        events: List[dict],
        temperature: Optional[float] = None,
        llm: Optional[BaseLLM] = None,
    ) -> dict:
        """Generate a continuation for the flow representing the current conversation."""

        if temperature is None:
            temperature = 0.0

        if self.instruction_flows_index is None:
            raise RuntimeError("No instruction flows index has been created.")

        # Use action specific llm if registered else fallback to main llm
        generation_llm: Optional[Union[BaseLLM, BaseChatModel]] = (
            llm if llm else self.llm
        )

        log.info("Generating flow continuation.")

        colang_history = colang(events)

        # We use the last line from the history to search for relevant flows
        search_text = colang_history.split("\n")[-1]

        if self.flows_index is None:
            raise RuntimeError("No flows index has been created.")
        results = await self.flows_index.search(
            text=search_text, max_results=10, threshold=None
        )

        examples = ""
        for result in reversed(results):
            examples += f"{result.meta['flow']}"
        examples = re.sub(r"#.*$", "", examples)
        examples = examples.strip("\n")

        # TODO: add examples from the actual running flows

        llm_call_info_var.set(LLMCallInfo(task=Task.GENERATE_FLOW_CONTINUATION.value))

        prompt = self.llm_task_manager.render_task_prompt(
            task=Task.GENERATE_FLOW_CONTINUATION,
            events=events,
            context={
                "examples": examples,
                "context": state.context,
            },
        )

        # We make this call with temperature 0 to have it as deterministic as possible.
        result = await llm_call(
            generation_llm, prompt, llm_params={"temperature": temperature}
        )

        # TODO: Currently, we only support generating a bot action as continuation. This could be generalized
        # Colang statements.

        result = self.llm_task_manager.parse_task_output(
            task=Task.GENERATE_FLOW_CONTINUATION, output=result
        )

        lines = _remove_leading_empty_lines(result).split("\n")

        if len(lines) == 0 or (len(lines) == 1 and lines[0] == ""):
            response = "\n".join(lines)
            log.warning(
                "GenerateFlowContinuationAction\nFAILING-PROMPT ::\n%s\n FAILING-RESPONSE: %s\n",
                prompt,
                response,
            )
            return {
                "name": "bot inform LLM issue",
                "body": 'flow bot inform LLM issue\n  bot say "Sorry! There was an issue in the LLM result form GenerateFlowContinuationAction!"',
            }

        line_0 = lines[0].lstrip(" ")
        bot_intent = remove_action_intent_identifiers([line_0])[0].strip(" ")
        if not bot_intent.startswith("bot "):
            bot_intent = get_first_bot_intent(result.splitlines())
        bot_action = get_first_bot_action(result.splitlines())

        if bot_action is None:
            raise LlmResponseError(f"Issue with LLM response: {result}")

        if bot_intent:
            bot_intent = escape_flow_name(bot_intent.strip(" "))

        uuid = new_uuid()[0:8]
        flow_name = f"_dynamic_{uuid} {bot_intent}"
        # TODO: parse potential parameters from flow name with a regex
        flow_parameters: List[Any] = []

        return {
            "name": flow_name,
            "parameters": flow_parameters,
            "body": f'@meta(bot_intent="{bot_intent}")\n'
            + f"flow {flow_name}\n"
            + f"  {bot_action}",
        }

    @action(name="CreateFlowAction", is_system_action=True, execute_async=True)
    async def create_flow(
        self,
        events: List[dict],
        name: str,
        body: str,
        decorators: Optional[str] = None,
    ) -> dict:
        """Create a new flow during runtime."""

        uuid = new_uuid()[0:8]

        name = escape_flow_name(name)
        flow_name = f"_dynamic_{uuid} {name}"
        # TODO: parse potential parameters from flow name with a regex

        body = f"flow {flow_name}\n  " + body
        if decorators:
            body = decorators + "\n" + body

        return {
            "name": flow_name,
            "parameters": [],
            "body": body,
        }

    @action(name="GenerateValueAction", is_system_action=True, execute_async=True)
    async def generate_value(  # pyright: ignore (TODO - different arguments to base-class)
        self,
        state: State,
        instructions: str,
        events: List[dict],
        var_name: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
    ) -> Any:
        """Generate a value in the context of the conversation.

        :param instructions: The instructions to generate the value.
        :param events: The full stream of events so far.
        :param var_name: The name of the variable to generate.
        :param llm: Custom llm model to generate_value
        """
        # Use action specific llm if registered else fallback to main llm
        generation_llm: Optional[Union[BaseLLM, BaseChatModel]] = (
            llm if llm else self.llm
        )

        # We search for the most relevant flows.
        examples = ""
        if self.flows_index:
            results = None
            if var_name:
                results = await self.flows_index.search(
                    text=f"${var_name} = ", max_results=5, threshold=None
                )

            # We add these in reverse order so the most relevant is towards the end.
            if results:
                for result in reversed(results):
                    # If the flow includes "GenerateValueAction", we ignore it as we don't want the LLM
                    # to learn to predict it.
                    if "GenerateValueAction" not in result.text:
                        examples += f"{result.text}\n\n"

        llm_call_info_var.set(
            LLMCallInfo(task=Task.GENERATE_VALUE_FROM_INSTRUCTION.value)
        )

        prompt = self.llm_task_manager.render_task_prompt(
            task=Task.GENERATE_VALUE_FROM_INSTRUCTION,
            events=events,
            context={
                "examples": examples,
                "instructions": instructions,
                "var_name": var_name if var_name else "result",
                "context": state.context,
            },
        )

        stop = self.llm_task_manager.get_stop_tokens(
            Task.GENERATE_USER_INTENT_FROM_USER_ACTION
        )

        result = await llm_call(
            generation_llm, prompt, stop=stop, llm_params={"temperature": 0.1}
        )

        # Parse the output using the associated parser
        result = self.llm_task_manager.parse_task_output(
            Task.GENERATE_VALUE_FROM_INSTRUCTION, output=result
        )

        # We only use the first line for now
        # TODO: support multi-line values?
        value = result.strip().split("\n")[0]

        # Because of conventions from other languages, sometimes the LLM might add
        # a ";" at the end of the line. We remove that
        if value.endswith(";"):
            value = value[:-1]

        # Remove variable name from the left if it appears in the result:
        if isinstance(prompt, str):
            last_prompt_line = prompt.strip().split("\n")[-1]
            value = value.replace(last_prompt_line, "").strip()
        elif isinstance(prompt, list) and isinstance(prompt[-1]["content"], str):
            last_prompt_line = prompt[-1]["content"].strip().split("\n")[-1]
            value = value.replace(last_prompt_line, "").strip()

        log.info("Generated value for $%s: %s", var_name, value)

        try:
            return literal_eval(value)
        except Exception:
            raise Exception(f"Invalid LLM response: `{value}`")

    @action(name="GenerateFlowAction", is_system_action=True, execute_async=True)
    async def generate_flow(
        self,
        state: State,
        events: List[dict],
        llm: Optional[BaseLLM] = None,
        flow_id: Optional[str] = None,
    ) -> dict:
        """Generate the body for a flow."""
        # Use action specific llm if registered else fallback to main llm
        generation_llm: Optional[Union[BaseLLM, BaseChatModel]] = (
            llm if llm else self.llm
        )

        triggering_flow_id = flow_id
        if not triggering_flow_id:
            raise RuntimeError(
                "No flow_id provided to generate flow."
            )  # TODO! Should flow_id be mandatory?

        flow_config = state.flow_configs[triggering_flow_id]
        if not flow_config.source_code:
            raise RuntimeError(f"No source_code in flow_config {flow_config}")
        docstrings = re.findall(r'"""(.*?)"""', flow_config.source_code, re.DOTALL)

        if len(docstrings) > 0:
            docstring = docstrings[0]
            if "one-off" not in docstring:
                self._last_docstring = docstring
        else:
            docstring = self._last_docstring

        render_context = {}
        render_context.update(state.context)
        # TODO: Taking the last element is a shortcut.
        #   a more robust logic needs to be implemented.
        render_context.update(state.flow_id_states[triggering_flow_id][-1].context)

        # We also extract dynamically the list of tools
        tools = []
        tool_names = []
        for flow_config in state.flow_configs.values():
            if flow_config.decorators.get("meta", {}).get("tool") is True:
                # We get rid of the first line, which is the decorator

                if not flow_config.source_code:
                    raise Exception(f"No source_code in flow_config {flow_config}")

                body = flow_config.source_code.split("\n", maxsplit=1)[1]

                # We only need the part up to the docstring
                # TODO: improve the logic below for extracting the "header"
                lines = body.split("\n")
                for i in range(len(lines)):
                    if lines[i].endswith('"""'):
                        lines = lines[0 : i + 1]
                        break

                tools.append("\n".join(lines))
                tool_names.append("`" + flow_config.id + "`")

        tools = textwrap.indent("\n\n".join(tools), "  ")

        render_context["tools"] = tools
        render_context["tool_names"] = ", ".join(tool_names)

        # TODO: add the context of the flow
        flow_nld = self.llm_task_manager._render_string(
            textwrap.dedent(docstring), context=render_context, events=events
        )

        llm_call_info_var.set(
            LLMCallInfo(task=Task.GENERATE_FLOW_CONTINUATION_FROM_NLD.value)
        )

        prompt = self.llm_task_manager.render_task_prompt(
            task=Task.GENERATE_FLOW_CONTINUATION_FROM_NLD,
            events=events,
            context={
                "flow_nld": flow_nld,
            },
        )

        stop = self.llm_task_manager.get_stop_tokens(
            Task.GENERATE_FLOW_CONTINUATION_FROM_NLD
        )

        result = await llm_call(
            generation_llm,
            prompt,
            stop=stop,
            llm_params={"temperature": self.config.lowest_temperature},
        )

        # Parse the output using the associated parser
        result = self.llm_task_manager.parse_task_output(
            Task.GENERATE_FLOW_CONTINUATION_FROM_NLD, output=result
        )

        result = _remove_leading_empty_lines(result)
        lines = result.split("\n")
        if "codeblock" in lines[0]:
            lines = lines[1:]

        if len(lines) == 0 or (len(lines) == 1 and lines[0] == ""):
            return {
                "name": "bot inform LLM issue",
                "body": 'flow bot inform LLM issue\n  bot say "Sorry! There was an issue in the LLM result form GenerateFlowContinuationAction!"',
            }

        # We make sure that we stop at a user action, and replace it with "..."
        for i in range(len(lines)):
            if lines[i].startswith("  user "):
                lines = lines[0:i]
                lines.append("  wait user input")
                lines.append("  ...")
                break
            elif "await " in lines[i]:
                # Force to wait and continue the generation when the result is received
                lines = lines[0 : i + 1]
                lines.append("  ...")
                break
            elif lines[i].strip() == "...":
                # Don't parse anything after "..."
                lines = lines[0 : i + 1]
                break

            elif re.match(r"  await .* -> \$.*", lines[i]):
                # The LLM could be tempted to use the definition syntax when calling the flows
                lines[i] = re.sub(r"  await (.*) -> (\$.*)", r"\2 = await \1", lines[i])

            elif lines[i].strip().startswith("bot say") and "..." in result:
                # Always wait for user input after the bot says something
                lines = lines[0 : i + 1]
                lines.append("  wait user input")
                lines.append("  ...")
                break

            elif "user input" in lines[i] or "user select" in lines[i]:
                lines[i] = "  wait user input"

        uuid = new_uuid()[0:8]

        flow_name = f"_dynamic_{uuid}"
        for i in range(len(lines)):
            if not lines[i].startswith("  "):
                lines[i] = "  " + lines[i]

        body = "\n".join(lines)
        body = f"""flow {flow_name}\n{body}"""

        if verbose.verbose_mode_enabled:
            console.print("[bold]Creating flow:[/]")
            for line in body.split("\n"):
                text = Text(line, style="black on yellow", end="\n")
                text.pad_right(console.width)
                console.print(text)

            console.print("")

        return {"name": flow_name, "parameters": [], "body": body}
