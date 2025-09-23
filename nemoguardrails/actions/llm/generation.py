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

"""A set of actions for generating various types of completions using an LLMs."""

import asyncio
import logging
import random
import re
import sys
import threading
from functools import lru_cache
from time import time
from typing import Callable, List, Optional, Union, cast

from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.actions.llm.utils import (
    flow_to_colang,
    get_first_nonempty_line,
    get_last_bot_intent_event,
    get_last_user_intent_event,
    get_last_user_utterance_event,
    get_multiline_response,
    get_retrieved_relevant_chunks,
    get_top_k_nonempty_lines,
    llm_call,
    strip_quotes,
)
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v2_x.lang.colang_ast import Flow, Spec, SpecOp
from nemoguardrails.colang.v2_x.runtime.eval import eval_expression
from nemoguardrails.context import (
    generation_options_var,
    llm_call_info_var,
    raw_llm_request,
    reasoning_trace_var,
    streaming_handler_var,
)
from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem
from nemoguardrails.kb.kb import KnowledgeBase
from nemoguardrails.llm.prompts import get_prompt
from nemoguardrails.llm.taskmanager import LLMTaskManager, ParsedTaskOutput
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.utils import (
    get_or_create_event_loop,
    new_event_dict,
    new_uuid,
    safe_eval,
)

log = logging.getLogger(__name__)


local_streaming_handlers = {}


class LLMGenerationActions:
    """A container objects for multiple related actions."""

    def __init__(
        self,
        config: RailsConfig,
        llm: Union[BaseLLM, BaseChatModel],
        llm_task_manager: LLMTaskManager,
        get_embedding_search_provider_instance: Callable[
            [Optional[EmbeddingSearchProvider]], EmbeddingsIndex
        ],
        verbose: bool = False,
    ):
        self.config = config
        self.llm = llm
        self.verbose = verbose

        # We extract the user/bot messages from the config as we might alter them.
        self.user_messages = config.user_messages.copy()
        self.bot_messages = config.bot_messages.copy()

        # If we have user messages, we build an index with them
        self.user_message_index = None
        self.bot_message_index = None
        self.flows_index = None

        self.get_embedding_search_provider_instance = (
            get_embedding_search_provider_instance
        )

        # There are still some edge cases not covered by nest_asyncio.
        # Using a separate thread always for now.
        loop = get_or_create_event_loop()
        if True or check_sync_call_from_async_loop():
            t = threading.Thread(target=asyncio.run, args=(self.init(),))
            t.start()
            t.join()
        else:
            loop.run_until_complete(self.init())

        self.llm_task_manager = llm_task_manager

        # We also initialize the environment for rendering bot messages
        self.env = SandboxedEnvironment()

        # If set, in passthrough mode, this function will be used instead of
        # calling the LLM with the user input.
        self.passthrough_fn = None

    async def init(self):
        # For Colang 2.x we need to do some initial processing
        if self.config.colang_version == "2.x":
            self._process_flows()

        await asyncio.gather(
            self._init_user_message_index(),
            self._init_bot_message_index(),
            self._init_flows_index(),
        )

    def _extract_user_message_example(self, flow: Flow):
        """Heuristic to extract user message examples from a flow."""
        elements = [
            item
            for item in flow.elements
            if item["_type"] != "doc_string_stmt" and item["_type"] != "stmt"
        ]
        if len(elements) != 2:
            return

        el = elements[1]
        if isinstance(el, SpecOp):
            if el.op == "match":
                spec = cast(SpecOp, el).spec
                if (
                    not hasattr(spec, "name")
                    or spec.name != "UtteranceUserActionFinished"
                ):
                    return

                if "final_transcript" not in spec.arguments:
                    return

                # Extract the message and remove the double quotes
                message = eval_expression(spec.arguments["final_transcript"], {})
                if isinstance(message, str):
                    self.user_messages[flow.name] = [message]

            elif el.op == "await":
                spec = cast(SpecOp, el).spec
                if isinstance(spec, dict) and spec.get("_type") == "spec_or":
                    specs = spec.get("elements")
                else:
                    assert isinstance(spec, Spec)
                    specs = [spec]

                for spec in specs:
                    if (
                        not spec.name.startswith("user ")
                        or not spec.arguments
                        or not spec.arguments["$0"]
                    ):
                        continue

                    message = eval_expression(spec.arguments["$0"], {})
                    if isinstance(message, str):
                        if flow.name not in self.user_messages:
                            self.user_messages[flow.name] = []
                        self.user_messages[flow.name].append(message)

    def _extract_bot_message_example(self, flow: Flow):
        # Quick heuristic to identify the user utterance examples
        if len(flow.elements) != 2:
            return

        el = flow.elements[1]
        if (
            not isinstance(el, SpecOp)
            or not hasattr(el.spec, "name")
            or el.spec.name != "UtteranceBotAction"
            or "script" not in el.spec.arguments
        ):
            return

        # Extract the message and remove the double quotes
        message = el.spec.arguments["script"][1:-1]

        self.bot_messages[flow.name] = [message]

    def _process_flows(self):
        """Process the provided flows to extract the user utterance examples."""
        flow: Flow
        for flow in self.config.flows:
            if flow.name.startswith("user "):
                self._extract_user_message_example(flow)

            if flow.name.startswith("bot "):
                self._extract_bot_message_example(flow)

    async def _init_user_message_index(self):
        """Initializes the index of user messages."""

        if not self.user_messages:
            return

        items = []
        for intent, utterances in self.user_messages.items():
            for text in utterances:
                items.append(IndexItem(text=text, meta={"intent": intent}))

        # If we have no patterns, we stop.
        if len(items) == 0:
            return

        self.user_message_index = self.get_embedding_search_provider_instance(
            self.config.core.embedding_search_provider
        )

        await self.user_message_index.add_items(items)

        # NOTE: this should be very fast, otherwise needs to be moved to separate thread.
        await self.user_message_index.build()

    async def _init_bot_message_index(self):
        """Initializes the index of bot messages."""

        if not self.bot_messages:
            return

        items = []
        for intent, utterances in self.bot_messages.items():
            for text in utterances:
                items.append(IndexItem(text=intent, meta={"text": text}))

        # If we have no patterns, we stop.
        if len(items) == 0:
            return

        self.bot_message_index = self.get_embedding_search_provider_instance(
            self.config.core.embedding_search_provider
        )
        await self.bot_message_index.add_items(items)

        # NOTE: this should be very fast, otherwise needs to be moved to separate thread.
        await self.bot_message_index.build()

    async def _init_flows_index(self):
        """Initializes the index of flows."""

        if not self.config.flows:
            return

        items = []
        for flow in self.config.flows:
            # We don't include the system flows in the index because we don't want
            # the LLM to predict system actions.
            if flow.get("is_system_flow", False):
                continue

            # TODO: check if the flow has system actions and ignore the flow.

            colang_flow = flow.get("source_code") or flow_to_colang(flow)

            # We index on the full body for now
            # items.append(IndexItem(text=colang_flow, meta={"flow": colang_flow}))

            # EXPERIMENTAL: We create an index entry for every line in the flow
            for line in colang_flow.split("\n"):
                if line.strip() != "":
                    items.append(IndexItem(text=line, meta={"flow": colang_flow}))

        # If we have no patterns, we stop.
        if len(items) == 0:
            return

        self.flows_index = self.get_embedding_search_provider_instance(
            self.config.core.embedding_search_provider
        )
        await self.flows_index.add_items(items)

        # NOTE: this should be very fast, otherwise needs to be moved to separate thread.
        await self.flows_index.build()

    def _get_general_instructions(self):
        """Helper to extract the general instruction."""
        text = ""
        for instruction in self.config.instructions:
            if instruction.type == "general":
                text = instruction.content

                # We stop at the first one for now
                break

        return text

    @lru_cache
    def _get_sample_conversation_two_turns(self):
        """Helper to extract only the two turns from the sample conversation.

        This is needed to be included to "seed" the conversation so that the model
        can follow the format more easily.
        """
        lines = self.config.sample_conversation.split("\n")
        i = 0
        user_count = 0
        while i < len(lines):
            if lines[i].startswith("user "):
                user_count += 1

            if user_count == 3:
                break

            i += 1

        sample_conversation = "\n".join(lines[0:i])

        # Remove any trailing new lines
        sample_conversation = sample_conversation.strip()

        return sample_conversation

    @action(is_system_action=True)
    async def generate_user_intent(
        self,
        events: List[dict],
        context: dict,
        config: RailsConfig,
        llm: Optional[BaseLLM] = None,
        kb: Optional[KnowledgeBase] = None,
    ):
        """Generate the canonical form for what the user said i.e. user intent."""
        # If using a single LLM call, use the specific action defined for this task.
        if self.config.rails.dialog.single_call.enabled:
            return await self.generate_intent_steps_message(
                events=events, llm=llm, kb=kb
            )
        # The last event should be the "StartInternalSystemAction" and the one before it the "UtteranceUserActionFinished".
        event = get_last_user_utterance_event(events)
        assert event["type"] == "UserMessage"

        # Use action specific llm if registered else fallback to main llm
        llm = llm or self.llm

        streaming_handler = streaming_handler_var.get()

        # TODO: check for an explicit way of enabling the canonical form detection

        if self.user_messages:
            # TODO: based on the config we can use a specific canonical forms model
            #  or use the LLM to detect the canonical form. The below implementation
            #  is for the latter.

            log.info("Phase 1 :: Generating user intent")

            # We search for the most relevant similar user utterance
            examples = ""
            potential_user_intents = []
            if isinstance(event["text"], list):
                text = " ".join(
                    [item["text"] for item in event["text"] if item["type"] == "text"]
                )
            else:
                text = event["text"]

            if self.user_message_index is not None:
                threshold = None

                if config.rails.dialog.user_messages:
                    threshold = (
                        config.rails.dialog.user_messages.embeddings_only_similarity_threshold
                    )

                results = await self.user_message_index.search(
                    text=text, max_results=5, threshold=threshold
                )

                # If the option to use only the embeddings is activated, we take the first
                # canonical form.
                if results and config.rails.dialog.user_messages.embeddings_only:
                    intent = results[0].meta["intent"]

                    return ActionResult(
                        events=[new_event_dict("UserIntent", intent=intent)]
                    )

                elif (
                    config.rails.dialog.user_messages.embeddings_only
                    and config.rails.dialog.user_messages.embeddings_only_fallback_intent
                ):
                    intent = (
                        config.rails.dialog.user_messages.embeddings_only_fallback_intent
                    )

                    return ActionResult(
                        events=[new_event_dict("UserIntent", intent=intent)]
                    )
                else:
                    results = await self.user_message_index.search(
                        text=text, max_results=5
                    )
                # We add these in reverse order so the most relevant is towards the end.
                for result in reversed(results):
                    examples += f'user "{result.text}"\n  {result.meta["intent"]}\n\n'
                    if result.meta["intent"] not in potential_user_intents:
                        potential_user_intents.append(result.meta["intent"])

            prompt = self.llm_task_manager.render_task_prompt(
                task=Task.GENERATE_USER_INTENT,
                events=events,
                context={
                    "examples": examples,
                    "potential_user_intents": ", ".join(potential_user_intents),
                },
            )

            # Initialize the LLMCallInfo object
            llm_call_info_var.set(LLMCallInfo(task=Task.GENERATE_USER_INTENT.value))

            # We make this call with temperature 0 to have it as deterministic as possible.
            result = await llm_call(
                llm, prompt, llm_params={"temperature": self.config.lowest_temperature}
            )

            # Parse the output using the associated parser
            result = self.llm_task_manager.parse_task_output(
                Task.GENERATE_USER_INTENT, output=result
            )
            result = result.text

            user_intent = get_first_nonempty_line(result)
            if user_intent is None:
                user_intent = "unknown message"

            if user_intent and user_intent.startswith("user "):
                user_intent = user_intent[5:]

            log.info(
                "Canonical form for user intent: "
                + (user_intent if user_intent else "None")
            )

            if user_intent is None:
                return ActionResult(
                    events=[new_event_dict("UserIntent", intent="unknown message")]
                )
            else:
                return ActionResult(
                    events=[new_event_dict("UserIntent", intent=user_intent)]
                )
        else:
            output_events = []

            # If we are in passthrough mode, we just use the input for prompting
            if self.config.passthrough:
                # We check if we have a raw request. If the guardrails API is using
                # the `generate_events` API, this will not be set.
                raw_prompt = raw_llm_request.get()

                if raw_prompt is None:
                    prompt = event["text"]
                else:
                    if isinstance(raw_prompt, str):
                        # If we're in completion mode, we use directly the last $user_message
                        # as it may have been altered by the input rails.
                        prompt = event["text"]
                    elif isinstance(raw_prompt, list):
                        prompt = raw_prompt.copy()

                        # In this case, if the last message is from the user, we replace the text
                        # just in case the input rails may have altered it.
                        if prompt[-1]["role"] == "user":
                            raw_prompt[-1]["content"] = event["text"]
                    else:
                        raise ValueError(
                            f"Unsupported type for raw prompt: {type(raw_prompt)}"
                        )

                if self.passthrough_fn:
                    raw_output = await self.passthrough_fn(
                        context=context, events=events
                    )

                    # If the passthrough action returns a single value, we consider that
                    # to be the text output
                    if isinstance(raw_output, tuple) or isinstance(raw_output, list):
                        text, passthrough_output = raw_output[0], raw_output[1]
                    else:
                        text = raw_output
                        passthrough_output = None

                    # We record the passthrough output in the context
                    output_events.append(
                        new_event_dict(
                            "ContextUpdate",
                            data={"passthrough_output": passthrough_output},
                        )
                    )
                else:
                    # Initialize the LLMCallInfo object
                    llm_call_info_var.set(LLMCallInfo(task=Task.GENERAL.value))

                    generation_options: GenerationOptions = generation_options_var.get()
                    llm_params = (
                        generation_options and generation_options.llm_params
                    ) or {}
                    text = await llm_call(
                        llm,
                        prompt,
                        custom_callback_handlers=[streaming_handler_var.get()],
                        llm_params=llm_params,
                    )
                    text = self.llm_task_manager.parse_task_output(
                        Task.GENERAL, output=text
                    )

                    text = _process_parsed_output(
                        text, self._include_reasoning_traces()
                    )

            else:
                # Initialize the LLMCallInfo object
                llm_call_info_var.set(LLMCallInfo(task=Task.GENERAL.value))

                if kb:
                    chunks = await kb.search_relevant_chunks(event["text"])
                    relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
                else:
                    # in case there is  no user flow (user message) then we need the context update to work for relevant_chunks
                    relevant_chunks = get_retrieved_relevant_chunks(
                        events, skip_user_message=True
                    )

                # Otherwise, we still create an altered prompt.
                prompt = self.llm_task_manager.render_task_prompt(
                    task=Task.GENERAL,
                    events=events,
                    context={"relevant_chunks": relevant_chunks},
                )

                generation_options: GenerationOptions = generation_options_var.get()
                llm_params = (
                    generation_options and generation_options.llm_params
                ) or {}
                result = await llm_call(
                    llm,
                    prompt,
                    custom_callback_handlers=[streaming_handler_var.get()],
                    stop=["User:"],
                    llm_params=llm_params,
                )

                text = self.llm_task_manager.parse_task_output(
                    Task.GENERAL, output=result
                )

                text = _process_parsed_output(text, self._include_reasoning_traces())
                text = text.strip()
                if text.startswith('"'):
                    text = text[1:-1]

            # In streaming mode, we also push this.
            if streaming_handler:
                await streaming_handler.push_chunk(text)

            if self.config.passthrough:
                from nemoguardrails.actions.llm.utils import (
                    get_and_clear_tool_calls_contextvar,
                )

                tool_calls = get_and_clear_tool_calls_contextvar()

                if tool_calls:
                    output_events.append(
                        new_event_dict("BotToolCalls", tool_calls=tool_calls)
                    )
                else:
                    output_events.append(new_event_dict("BotMessage", text=text))
            else:
                output_events.append(new_event_dict("BotMessage", text=text))

            return ActionResult(events=output_events)

    async def _search_flows_index(self, text, max_results):
        """Search the index of flows."""
        results = await self.flows_index.search(text=text, max_results=10)

        # we filter the results to keep only unique flows
        flows = set()
        final_results = []
        for result in results:
            if result.meta["flow"] not in flows:
                flows.add(result.meta["flow"])
                # For backwards compatibility we also replace the text with the full version
                result.text = result.meta["flow"]
                final_results.append(result)

        return final_results[0:max_results]

    @action(is_system_action=True)
    async def generate_next_step(
        self, events: List[dict], llm: Optional[BaseLLM] = None
    ):
        """Generate the next step in the current conversation flow.

        Currently, only generates a next step after a user intent.
        """
        log.info("Phase 2 :: Generating next step ...")

        # Use action specific llm if registered else fallback to main llm
        llm = llm or self.llm

        # The last event should be the "StartInternalSystemAction" and the one before it the "UserIntent".
        event = get_last_user_intent_event(events)

        # Currently, we only predict next step after a user intent using LLM
        if event["type"] == "UserIntent":
            # If using a single LLM call, use the results computed in the first call.
            if self.config.rails.dialog.single_call.enabled:
                bot_intent_event = event["additional_info"]["bot_intent_event"]
                return ActionResult(events=[bot_intent_event])

            user_intent = event["intent"]

            # We search for the most relevant similar flows
            examples = ""
            if self.flows_index:
                results = await self._search_flows_index(
                    text=user_intent, max_results=5
                )

                # We add these in reverse order so the most relevant is towards the end.
                for result in reversed(results):
                    examples += f"{result.text}\n\n"

            prompt = self.llm_task_manager.render_task_prompt(
                task=Task.GENERATE_NEXT_STEPS,
                events=events,
                context={"examples": examples},
            )

            # Initialize the LLMCallInfo object
            llm_call_info_var.set(LLMCallInfo(task=Task.GENERATE_NEXT_STEPS.value))

            # We use temperature 0 for next step prediction as well
            result = await llm_call(
                llm, prompt, llm_params={"temperature": self.config.lowest_temperature}
            )

            # Parse the output using the associated parser
            result = self.llm_task_manager.parse_task_output(
                Task.GENERATE_NEXT_STEPS, output=result
            )
            result = result.text

            # If we don't have multi-step generation enabled, we only look at the first line.
            if not self.config.enable_multi_step_generation:
                result = get_first_nonempty_line(result)

                if result and result.startswith("bot "):
                    bot_intent = result[4:]

                    # Sometimes, the LLMs add also the message on the same line.
                    # We do some cleaning up if that's the case.
                    if '"' in bot_intent:
                        bot_intent = bot_intent.split('"')[0].strip()

                    # Also, sometimes, there's a comma and more content
                    if "," in bot_intent:
                        bot_intent = bot_intent.split(",")[0].strip()

                    next_step = {"bot": bot_intent}
                else:
                    next_step = {"bot": "general response"}

                # If we have to execute an action, we return the event to start it
                if next_step.get("execute"):
                    return ActionResult(
                        events=[
                            new_event_dict(
                                "StartInternalSystemAction",
                                action_name=next_step["execute"],
                            )
                        ]
                    )
                else:
                    bot_intent = next_step.get("bot")

                    return ActionResult(
                        events=[new_event_dict("BotIntent", intent=bot_intent)]
                    )
            else:
                # Otherwise, we parse the output as a single flow.
                # If we have a parsing error, we try to reduce size of the flow, potentially
                # up to a single step.
                lines = result.split("\n")
                while True:
                    try:
                        parse_colang_file("dynamic.co", content="\n".join(lines))
                        break
                    except Exception as e:
                        # If we could not parse the flow on the last line, we return a general response
                        if len(lines) == 1:
                            log.info("Exception while parsing single line: %s", e)
                            return ActionResult(
                                events=[
                                    new_event_dict(
                                        "BotIntent", intent="general response"
                                    )
                                ]
                            )

                        log.info("Could not parse %s lines, reducing size", len(lines))
                        lines = lines[:-1]

                return ActionResult(
                    events=[
                        # We generate a random UUID as the flow_id
                        new_event_dict(
                            "start_flow",
                            flow_id=new_uuid(),
                            flow_body="\n".join(lines),
                        )
                    ]
                )

        return ActionResult(return_value=None)

    def _render_string(
        self,
        template_str: str,
        context: Optional[dict] = None,
    ) -> str:
        """Render a string using the provided context information.

        Args:
            template_str: The string template to render.
            context: The context for rendering.

        Returns:
            The rendered string.
        """
        # First, if we have any direct usage of variables in the string,
        # we replace with correct Jinja syntax.
        for param in re.findall(r"\$([^ \"'!?\-,;</]*(?:\w|]))", template_str):
            template_str = template_str.replace(f"${param}", "{{" + param + "}}")

        template = self.env.from_string(template_str)

        # First, we extract all the variables from the template.
        variables = meta.find_undeclared_variables(self.env.parse(template_str))

        # This is the context that will be passed to the template when rendering.
        render_context = {}

        # Copy the context variables to the render context.
        if context:
            for variable in variables:
                if variable in context:
                    render_context[variable] = context[variable]

        return template.render(render_context)

    @action(is_system_action=True)
    async def generate_bot_message(
        self, events: List[dict], context: dict, llm: Optional[BaseLLM] = None
    ):
        """Generate a bot message based on the desired bot intent."""
        log.info("Phase 3 :: Generating bot message ...")

        # Use action specific llm if registered else fallback to main llm
        llm = llm or self.llm

        # The last event should be the "StartInternalSystemAction" and the one before it the "BotIntent".
        event = get_last_bot_intent_event(events)
        assert event["type"] == "BotIntent"
        bot_intent = event["intent"]
        context_updates = {}

        streaming_handler = streaming_handler_var.get()

        # when we have 'output rails streaming' enabled
        # we must disable (skip) the output rails which gets executed on $bot_message
        # as it is executed separately in llmrails.py
        # of course, it does not work when passed as context in `run_output_rails_in_streaming`
        # streaming_handler is set when stream_async method is used

        # if streaming_handler and len(self.config.rails.output.flows) > 0:
        if streaming_handler and self.config.rails.output.streaming.enabled:
            context_updates["skip_output_rails"] = True

        if bot_intent in self.config.bot_messages:
            # Choose a message randomly from self.config.bot_messages[bot_message]
            # However, in test mode, we always choose the first one, to keep it predictable.
            if "pytest" in sys.modules:
                bot_utterance = self.bot_messages[bot_intent][0]
            else:
                bot_utterance = random.choice(self.bot_messages[bot_intent])

            log.info("Found existing bot message: " + bot_utterance)

            # We also need to render
            bot_utterance = self._render_string(bot_utterance, context)

            # We skip output rails for predefined messages.
            context_updates["skip_output_rails"] = True

        # Check if the output is supposed to be the content of a context variable
        elif bot_intent and bot_intent[0] == "$" and bot_intent[1:] in context:
            bot_utterance = context[bot_intent[1:]]

        else:
            # Generate the bot message using an LLM call

            # If using a single LLM call, use the results computed in the first call.
            if self.config.rails.dialog.single_call.enabled:
                event = get_last_user_intent_event(events)

                if event["type"] == "UserIntent":
                    bot_message_event = event["additional_info"]["bot_message_event"]

                    # We only need to use the bot message if it corresponds to the
                    # generate bot intent as well.
                    last_bot_intent = get_last_bot_intent_event(events)

                    if (
                        last_bot_intent["intent"]
                        == event["additional_info"]["bot_intent_event"]["intent"]
                    ):
                        text = bot_message_event["text"]
                        # If the bot message is being generated in streaming mode
                        if text.startswith('Bot message: "<<STREAMING['):
                            # Format: `Bot message: "<<STREAMING[...]>>"`
                            # Extract the streaming handler uid and get a reference.
                            streaming_handler_uid = text[26:-4]
                            _streaming_handler = local_streaming_handlers[
                                streaming_handler_uid
                            ]

                            # We pipe the content from this handler to the main one.
                            _streaming_handler.set_pipe_to(streaming_handler)
                            await _streaming_handler.disable_buffering()

                            # And wait for it to finish.
                            # We stop after the closing double quotes for the bot message.
                            _streaming_handler.stop = [
                                '"\n',
                            ]
                            text = await _streaming_handler.wait()
                            return ActionResult(
                                events=[new_event_dict("BotMessage", text=text)]
                            )
                        else:
                            if streaming_handler:
                                await streaming_handler.push_chunk(
                                    bot_message_event["text"]
                                )

                            return ActionResult(events=[bot_message_event])

            # If we are in passthrough mode, we just use the input for prompting
            if self.config.passthrough:
                # If we have a passthrough function, we use that.
                if self.passthrough_fn:
                    prompt = None
                    raw_output = await self.passthrough_fn(
                        context=context, events=events
                    )

                    # If the passthrough action returns a single value, we consider that
                    # to be the text output
                    if isinstance(raw_output, tuple) or isinstance(raw_output, list):
                        result, passthrough_output = raw_output[0], raw_output[1]
                    else:
                        result = raw_output
                        passthrough_output = None

                    # We record the passthrough output in the context
                    context_updates["passthrough_output"] = passthrough_output
                else:
                    # Otherwise, we call the LLM with the prompt coming from the user.

                    t0 = time()

                    # Initialize the LLMCallInfo object
                    llm_call_info_var.set(
                        LLMCallInfo(task=Task.GENERATE_BOT_MESSAGE.value)
                    )

                    # In passthrough mode, we should use the full conversation history
                    # instead of just the last user message to preserve tool message context
                    raw_prompt = raw_llm_request.get()

                    if raw_prompt is not None and isinstance(raw_prompt, list):
                        # Use the full conversation including tool messages
                        prompt = raw_prompt.copy()

                        # Update the last user message if it was altered by input rails
                        user_message = context.get("user_message")
                        if user_message and prompt:
                            for i in reversed(range(len(prompt))):
                                if prompt[i]["role"] == "user":
                                    prompt[i]["content"] = user_message
                                    break
                    else:
                        prompt = context.get("user_message")

                    generation_options: GenerationOptions = generation_options_var.get()
                    llm_params = (
                        generation_options and generation_options.llm_params
                    ) or {}
                    result = await llm_call(
                        llm,
                        prompt,
                        custom_callback_handlers=[streaming_handler],
                        llm_params=llm_params,
                    )

                    result = self.llm_task_manager.parse_task_output(
                        Task.GENERAL, output=result
                    )

                    result = _process_parsed_output(
                        result, self._include_reasoning_traces()
                    )

                    log.info(
                        "--- :: LLM Bot Message Generation passthrough call took %.2f seconds",
                        time() - t0,
                    )
            else:
                # Otherwise, we go through the process of creating the altered prompt,
                # which includes examples, relevant chunks, etc.

                # We search for the most relevant similar bot utterance
                examples = ""
                # NOTE: disabling bot message index when there are no user messages
                if self.config.user_messages and self.bot_message_index:
                    results = await self.bot_message_index.search(
                        text=event["intent"], max_results=5
                    )

                    # We add these in reverse order so the most relevant is towards the end.
                    for result in reversed(results):
                        examples += f'bot {result.text}\n  "{result.meta["text"]}"\n\n'

                # We compute the relevant chunks to be used as context
                relevant_chunks = get_retrieved_relevant_chunks(events)

                prompt_config = get_prompt(self.config, Task.GENERATE_BOT_MESSAGE)
                prompt = self.llm_task_manager.render_task_prompt(
                    task=Task.GENERATE_BOT_MESSAGE,
                    events=events,
                    context={"examples": examples, "relevant_chunks": relevant_chunks},
                )

                t0 = time()

                if streaming_handler:
                    # TODO: Figure out a more generic way to deal with this
                    if prompt_config.output_parser in ["verbose_v1", "bot_message"]:
                        streaming_handler.set_pattern(
                            prefix='Bot message: "', suffix='"'
                        )
                    else:
                        streaming_handler.set_pattern(prefix='  "', suffix='"')

                # Initialize the LLMCallInfo object
                llm_call_info_var.set(LLMCallInfo(task=Task.GENERATE_BOT_MESSAGE.value))

                generation_options: GenerationOptions = generation_options_var.get()
                llm_params = (
                    generation_options and generation_options.llm_params
                ) or {}
                result = await llm_call(
                    llm,
                    prompt,
                    custom_callback_handlers=[streaming_handler],
                    llm_params=llm_params,
                )

                log.info(
                    "--- :: LLM Bot Message Generation call took %.2f seconds",
                    time() - t0,
                )

                # Parse the output using the associated parser
                result = self.llm_task_manager.parse_task_output(
                    Task.GENERATE_BOT_MESSAGE, output=result
                )

                result = _process_parsed_output(
                    result, self._include_reasoning_traces()
                )

                # TODO: catch openai.error.InvalidRequestError from exceeding max token length

                result = get_multiline_response(result)
                result = strip_quotes(result)

            bot_utterance = result

            # Context variable starting with "_" are considered private (not used in tests or logging)
            context_updates["_last_bot_prompt"] = prompt

            log.info(f"Generated bot message: {bot_utterance}")

        if bot_utterance:
            bot_utterance = clean_utterance_content(bot_utterance)
            # In streaming mode, we also push this.
            if streaming_handler:
                await streaming_handler.push_chunk(bot_utterance)

            return ActionResult(
                events=[new_event_dict("BotMessage", text=bot_utterance)],
                context_updates=context_updates,
            )
        else:
            # In streaming mode, we also push this.
            bot_utterance = "I'm not sure what to say."
            if streaming_handler:
                await streaming_handler.push_chunk(bot_utterance)

            return ActionResult(
                events=[new_event_dict("BotMessage", text=bot_utterance)],
                context_updates=context_updates,
            )

    @action(is_system_action=True)
    async def generate_value(
        self,
        instructions: str,
        events: List[dict],
        var_name: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
    ):
        """Generate a value in the context of the conversation.

        :param instructions: The instructions to generate the value.
        :param events: The full stream of events so far.
        :param var_name: The name of the variable to generate. If not specified, it will use
          the `action_result_key` as the name of the variable.
        :param llm: Custom llm model to generate_value
        """
        # Use action specific llm if registered else fallback to main llm
        llm = llm or self.llm

        last_event = events[-1]
        assert last_event["type"] == "StartInternalSystemAction"

        if not var_name:
            var_name = last_event["action_result_key"]

        # We search for the most relevant flows.
        examples = ""
        if self.flows_index:
            results = await self._search_flows_index(
                text=f"${var_name} = ", max_results=5
            )

            # We add these in reverse order so the most relevant is towards the end.
            for result in reversed(results):
                # If the flow includes "= ...", we ignore it as we don't want the LLM
                # to learn to predict "...".
                if not re.findall(r"=\s+\.\.\.", result.text):
                    examples += f"{result.text}\n\n"

        prompt = self.llm_task_manager.render_task_prompt(
            task=Task.GENERATE_VALUE,
            events=events,
            context={
                "examples": examples,
                "instructions": instructions,
                "var_name": var_name,
            },
        )

        # Initialize the LLMCallInfo object
        llm_call_info_var.set(LLMCallInfo(task=Task.GENERATE_VALUE.value))

        result = await llm_call(
            llm, prompt, llm_params={"temperature": self.config.lowest_temperature}
        )

        # Parse the output using the associated parser
        result = self.llm_task_manager.parse_task_output(
            Task.GENERATE_VALUE, output=result
        )
        result = result.text

        # We only use the first line for now
        # TODO: support multi-line values?
        value = result.strip().split("\n")[0]

        # Because of conventions from other languages, sometimes the LLM might add
        # a ";" at the end of the line. We remove that
        if value.endswith(";"):
            value = value[:-1]

        log.info(f"Generated value for ${var_name}: {value}")

        try:
            return safe_eval(value)
        except Exception as e:
            log.error(f"Error evaluating value: {value}. Error: {str(e)}")
            raise ValueError(f"Invalid LLM response: `{value}`")

    @action(is_system_action=True)
    async def generate_intent_steps_message(
        self,
        events: List[dict],
        llm: Optional[BaseLLM] = None,
        kb: Optional[KnowledgeBase] = None,
    ):
        """Generate all three main Guardrails phases with a single LLM call.
        The three phases are: user canonical from (user intent), next flow steps (i.e. bot canonical form)
        and bot message.
        """

        # The last event should be the "StartInternalSystemAction" and the one before it the "UtteranceUserActionFinished".
        event = get_last_user_utterance_event(events)
        assert event["type"] == "UserMessage"

        # Use action specific llm if registered else fallback to main llm
        llm = llm or self.llm

        streaming_handler = streaming_handler_var.get()

        if self.config.user_messages:
            # TODO: based on the config we can use a specific canonical forms model
            #  or use the LLM to detect the canonical form. The below implementation
            #  is for the latter.

            log.info("Generate all three phases in one LLM call...")

            # We search for the most relevant similar user utterance
            examples = []
            potential_user_intents = []
            intent_results = []
            flow_results = {}

            if self.user_message_index:
                # Get the top 10 intents even if we use less in the selected examples.
                # Some of these intents might not have an associated flow and will be
                # skipped from the few-shot examples.
                intent_results = await self.user_message_index.search(
                    text=event["text"], max_results=10
                )

                # We fill in the list of potential user intents
                for result in intent_results:
                    if result.meta["intent"] not in potential_user_intents:
                        potential_user_intents.append(result.meta["intent"])

            if self.flows_index:
                for intent in potential_user_intents:
                    flow_results_intent = await self._search_flows_index(
                        text=intent, max_results=2
                    )
                    flow_results[intent] = flow_results_intent

            # We add the intent to the examples in reverse order
            # so the most relevant is towards the end.
            for result in intent_results:
                # Stop after the first 5 flow examples, in case more than 5 intents
                # have been selected from the index.
                if len(examples) >= 5:
                    break

                intent = result.meta["intent"]
                example = f'user "{result.text}"\n  {intent}\n'

                flow_results_intent = flow_results.get(intent, [])
                found_flow_for_intent = False
                for result_flow in flow_results_intent:
                    # Assumption: each flow should contain at least two lines, the first is the user intent.
                    # Just in case there are some flows with only one line
                    if "\n" not in result_flow.text:
                        continue
                    (flow_user_intent, flow_continuation) = result_flow.text.split(
                        "\n", 1
                    )
                    flow_user_intent = flow_user_intent[5:]
                    if flow_user_intent == intent:
                        found_flow_for_intent = True
                        example += f"{flow_continuation}\n"

                        # Also add the bot message if the last line in the flow is a bot canonical form
                        last_flow_line = flow_continuation
                        if "\n" in flow_continuation:
                            (_, last_flow_line) = flow_continuation.rsplit("\n", 1)
                        if last_flow_line.startswith("bot "):
                            bot_canonical_form = last_flow_line[4:]

                            found_bot_message = False
                            if self.bot_message_index:
                                bot_messages_results = (
                                    await self.bot_message_index.search(
                                        text=bot_canonical_form, max_results=1
                                    )
                                )

                                for bot_message_result in bot_messages_results:
                                    if bot_message_result.text == bot_canonical_form:
                                        found_bot_message = True
                                        example += (
                                            f'  "{bot_message_result.meta["text"]}"\n'
                                        )
                                        # Only use the first bot message for now
                                        break

                            if not found_bot_message:
                                # This is for canonical forms that do not have an associated message.
                                # Create a simple message for the bot canonical form.
                                # In a later version we could generate a message with the LLM at app initialization.
                                example += f"  # On the next line generate a bot message related to {bot_canonical_form}\n"

                        # For now, only use the first flow for each intent.
                        break
                if not found_flow_for_intent:
                    # Skip intents that do not have an associated flow.
                    continue

                example += "\n"
                examples.append(example)

            if kb:
                chunks = await kb.search_relevant_chunks(event["text"])
                relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
            else:
                relevant_chunks = ""

            relevant_chunks = relevant_chunks.strip()

            prompt = self.llm_task_manager.render_task_prompt(
                task=Task.GENERATE_INTENT_STEPS_MESSAGE,
                events=events,
                context={
                    "examples": "\n\n".join(reversed(examples)),
                    "potential_user_intents": ", ".join(potential_user_intents),
                    "relevant_chunks": relevant_chunks,
                },
            )
            prompt_config = get_prompt(self.config, Task.GENERATE_INTENT_STEPS_MESSAGE)

            # We make this call with temperature 0 to have it as deterministic as possible.
            # This is important for canonical forms, but not a great choice for bot messages.

            if streaming_handler:
                # Create a new "inner" streaming handler and save the reference
                _streaming_handler = StreamingHandler()
                local_streaming_handlers[_streaming_handler.uid] = _streaming_handler

                # We buffer the content, so we can get a chance to look at the
                # first k lines.
                await _streaming_handler.enable_buffering()
                asyncio.create_task(
                    llm_call(
                        llm,
                        prompt,
                        custom_callback_handlers=[_streaming_handler],
                        stop=["\nuser ", "\nUser "],
                        llm_params={"temperature": self.config.lowest_temperature},
                    )
                )
                result = await _streaming_handler.wait_top_k_nonempty_lines(k=2)

                # We also mark that the message is still being generated
                # by a streaming handler.
                result += f'\nBot message: "<<STREAMING[{_streaming_handler.uid}]>>"'

                # Moving forward we need to set the expected pattern to correctly
                # parse the message.
                # TODO: Figure out a more generic way to deal with this.
                if prompt_config.output_parser == "verbose_v1":
                    _streaming_handler.set_pattern(prefix='Bot message: "', suffix='"')
                else:
                    _streaming_handler.set_pattern(prefix='  "', suffix='"')
            else:
                # Initialize the LLMCallInfo object
                llm_call_info_var.set(
                    LLMCallInfo(task=Task.GENERATE_INTENT_STEPS_MESSAGE.value)
                )

                generation_options: GenerationOptions = generation_options_var.get()
                additional_params = {
                    **((generation_options and generation_options.llm_params) or {}),
                    "temperature": self.config.lowest_temperature,
                }
                result = await llm_call(llm, prompt, llm_params=additional_params)

            # Parse the output using the associated parser
            result = self.llm_task_manager.parse_task_output(
                Task.GENERATE_INTENT_STEPS_MESSAGE, output=result
            )
            result = result.text

            # TODO: Implement logic for generating more complex Colang next steps (multi-step),
            #  not just a single bot intent.

            # Get the next 2 non-empty lines, these should contain:
            # line 1 - user intent, line 2 - bot intent.
            # Afterwards we have the bot message.
            next_three_lines = get_top_k_nonempty_lines(result, k=2)
            user_intent = next_three_lines[0] if len(next_three_lines) > 0 else None
            bot_intent = next_three_lines[1] if len(next_three_lines) > 1 else None
            bot_message = None
            if bot_intent:
                pos = result.find(bot_intent)
                if pos != -1:
                    # The bot message could be multiline
                    bot_message = result[pos + len(bot_intent) :]
                    bot_message = get_multiline_response(bot_message)
                    bot_message = strip_quotes(bot_message)
                    # Quick hack for degenerated / empty bot messages
                    if bot_message and len(bot_message.strip()) == 0:
                        bot_message = None

            if user_intent:
                if user_intent.startswith("user "):
                    user_intent = user_intent[5:]
                elif user_intent.startswith("User intent: "):
                    user_intent = user_intent[13:]
            else:
                user_intent = "unknown message"

            if bot_intent and bot_intent.startswith("bot "):
                bot_intent = bot_intent[4:]
            elif bot_intent and bot_intent.startswith("Bot intent: "):
                bot_intent = bot_intent[12:]
            else:
                bot_intent = "general response"

            if not bot_message:
                bot_message = "I'm not sure what to say."

            log.info(
                "Canonical form for user intent: "
                + (user_intent if user_intent else "None")
            )
            log.info(
                "Canonical form for bot intent: "
                + (bot_intent if bot_intent else "None")
            )
            log.info(
                f"Generated bot message: " + (bot_message if bot_message else "None")
            )

            additional_info = {
                "bot_intent_event": new_event_dict("BotIntent", intent=bot_intent),
                "bot_message_event": new_event_dict("BotMessage", text=bot_message),
            }
            events = [
                new_event_dict(
                    "UserIntent", intent=user_intent, additional_info=additional_info
                )
            ]

            return ActionResult(events=events)

        else:
            prompt = self.llm_task_manager.render_task_prompt(
                task=Task.GENERAL, events=events
            )

            # Initialize the LLMCallInfo object
            llm_call_info_var.set(LLMCallInfo(task=Task.GENERAL.value))

            # We make this call with temperature 0 to have it as deterministic as possible.
            generation_options: GenerationOptions = generation_options_var.get()
            llm_params = (generation_options and generation_options.llm_params) or {}
            result = await llm_call(llm, prompt, llm_params=llm_params)

            result = self.llm_task_manager.parse_task_output(
                Task.GENERAL, output=result
            )
            result = _process_parsed_output(result, self._include_reasoning_traces())
            text = result.strip()
            if text.startswith('"'):
                text = text[1:-1]

            # In streaming mode, we also push this.
            if streaming_handler:
                await streaming_handler.push_chunk(text)

            return ActionResult(
                events=[new_event_dict("BotMessage", text=text)],
            )

    def _include_reasoning_traces(self) -> bool:
        """Get the configuration value for whether to include reasoning traces in output."""
        return _get_apply_to_reasoning_traces(self.config)


def clean_utterance_content(utterance: str) -> str:
    """
    Clean an utterance by performing the following operations:
     - replacing "\\n" with "\n".

    Args:
        utterance (str): The utterance to clean.

    Returns:
        str: The cleaned utterance.
    """
    if utterance:
        # If "\n" is used inside a predefined message, it will be returned as is as part of the message.
        # It should be translated to an actual \n character.
        utterance = utterance.replace("\\n", "\n")
    return utterance


def _record_reasoning_trace(trace: str) -> None:
    """Store the reasoning trace in context for later retrieval."""
    reasoning_trace_var.set(trace)


def _assemble_response(text: str, trace: Optional[str], include_reasoning: bool) -> str:
    """Combine trace and text if requested, otherwise just return text."""
    return (trace + text) if (trace and include_reasoning) else text


def _process_parsed_output(
    output: ParsedTaskOutput, include_reasoning_trace: bool
) -> str:
    """Record trace, then assemble the final LLM response."""
    if reasoning_trace := output.reasoning_trace:
        _record_reasoning_trace(reasoning_trace)
    return _assemble_response(output.text, reasoning_trace, include_reasoning_trace)


def _get_apply_to_reasoning_traces(config: RailsConfig) -> bool:
    """Get the configuration value for whether to include reasoning traces in output."""
    return config.rails.output.apply_to_reasoning_traces
