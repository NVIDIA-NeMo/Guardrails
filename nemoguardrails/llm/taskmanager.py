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
import re
from ast import literal_eval
from typing import Any, Callable, Dict, List, Optional, Union

from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment

from nemoguardrails.llm.filters import (
    co_v2,
    colang,
    colang_without_identifiers,
    first_turns,
    indent,
    last_turns,
    remove_text_messages,
    to_chat_messages,
    to_intent_messages,
    to_intent_messages_2,
    to_messages,
    to_messages_v2,
    user_assistant_sequence,
    verbose_v1,
)
from nemoguardrails.llm.output_parsers import (
    bot_intent_parser,
    bot_message_parser,
    is_content_safe,
    nemoguard_parse_prompt_safety,
    nemoguard_parse_response_safety,
    user_intent_parser,
    verbose_v1_parser,
)
from nemoguardrails.llm.prompts import get_prompt
from nemoguardrails.llm.types import Task
from nemoguardrails.rails.llm.config import MessageTemplate, RailsConfig


class LLMTaskManager:
    """Interface for interacting with an LLM in a task-oriented way."""

    def __init__(self, config: RailsConfig):
        # Save the config as we need access to instructions and sample conversations.
        self.config = config
        # Initialize the environment for rendering templates.
        self.env = SandboxedEnvironment()

        # Register the default filters.
        self.env.filters["colang"] = colang
        self.env.filters["co_v2"] = co_v2
        self.env.filters["colang_without_identifiers"] = colang_without_identifiers
        self.env.filters["remove_text_messages"] = remove_text_messages
        self.env.filters["first_turns"] = first_turns
        self.env.filters["last_turns"] = last_turns
        self.env.filters["indent"] = indent
        self.env.filters["user_assistant_sequence"] = user_assistant_sequence
        self.env.filters["to_messages"] = to_messages
        self.env.filters["to_messages_v2"] = to_messages_v2
        self.env.filters["to_intent_messages"] = to_intent_messages
        self.env.filters["to_intent_messages_2"] = to_intent_messages_2
        self.env.filters["to_chat_messages"] = to_chat_messages
        self.env.filters["verbose_v1"] = verbose_v1

        self.output_parsers: Dict[Optional[str], Callable] = {
            "user_intent": user_intent_parser,
            "bot_intent": bot_intent_parser,
            "bot_message": bot_message_parser,
            "verbose_v1": verbose_v1_parser,
            "is_content_safe": is_content_safe,
            "nemoguard_parse_prompt_safety": nemoguard_parse_prompt_safety,
            "nemoguard_parse_response_safety": nemoguard_parse_response_safety,
        }

        # The prompt context will hold additional variables that ce also be included
        # in the prompt.
        self.prompt_context = {}

    def _get_general_instructions(self):
        """Helper to extract the general instructions."""
        text = ""
        if self.config.instructions is None:
            return text
        for instruction in self.config.instructions:
            if instruction.type == "general":
                text = instruction.content

                # We stop at the first one for now
                break

        return text

    def _render_string(
        self,
        template_str: str,
        context: Optional[dict] = None,
        events: Optional[List[dict]] = None,
    ) -> str:
        """Render a template using the provided context information.

        :param template_str: The template to render.
        :param context: The context for rendering the prompt.
        :param events: The history of events so far.
        :return: The rendered template.
        :rtype: str.
        """
        template = self.env.from_string(template_str)

        # First, we extract all the variables from the template.
        variables = meta.find_undeclared_variables(self.env.parse(template_str))

        # This is the context that will be passed to the template when rendering.
        render_context = {
            "history": events,
            "general_instructions": self._get_general_instructions(),
            "sample_conversation": self.config.sample_conversation,
            "sample_conversation_two_turns": self.config.sample_conversation,
        }

        # Copy the context variables to the render context.
        if context:
            for variable in variables:
                if variable in context:
                    render_context[variable] = context[variable]

        # Last but not least, if we have variables from the prompt context, we add them
        # to the render context.
        if self.prompt_context:
            for variable in variables:
                if variable in self.prompt_context:
                    value = self.prompt_context[variable]

                    # If it's a callable, we compute the value, otherwise we just use it
                    # as is.
                    if callable(value):
                        value = value()

                    render_context[variable] = value

        return template.render(render_context)

    def _render_messages(
        self,
        message_templates: List[Union[str, MessageTemplate]],
        context: Optional[dict] = None,
        events: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Render a sequence of messages.

        :param message_templates: The message templates to render.
        :param context: The context for rendering the prompt.
        :param events: The history of events so far.
        :return: The rendered messages.
        """
        messages = []

        # We iterate each template and render it.
        # If it's a string, it must be a list of messages in JSON format.
        # If it's a MessageTemplate, we render it as a message.
        for message_template in message_templates:
            if isinstance(message_template, str):
                str_messages = self._render_string(message_template, context=context, events=events)
                try:
                    new_messages = literal_eval(str_messages)
                except SyntaxError:
                    raise ValueError(f"Invalid message template: {message_template}")
                messages.extend(new_messages)
            else:
                content = self._render_string(message_template.content, context=context, events=events)

                # Don't add empty messages.
                if content.strip():
                    messages.append(
                        {
                            "type": message_template.type,
                            "content": content,
                        }
                    )

        return messages

    def _get_messages_text_length(self, messages: List[dict]) -> int:
        """Return the length of the text in the messages for token counting purposes.

        This method calculates text length for token limit checks, using placeholders for base64 images
        instead of counting their full encoded size. This allows multimodal content with large base64
        images to pass the length checks while still preserving the actual content.
        """

        def process_content_for_length(content):
            """Process any content type (string, list, dict) and return its effective text."""
            result_text = ""

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            result_text += item.get("text", "") + "\n"
                        elif item.get("type") == "image_url" and isinstance(item.get("image_url"), dict):
                            # image_url items, only count a placeholder length
                            result_text += "[IMAGE_CONTENT]\n"

            # string content that might contain base64 data
            elif isinstance(content, str):
                base64_pattern = r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+"
                if re.search(base64_pattern, content):
                    # Replace base64 content with placeholder using regex
                    result_text += re.sub(base64_pattern, "[IMAGE_CONTENT]", content) + "\n"
                else:
                    result_text += content + "\n"

            return result_text

        text = ""
        for message in messages:
            content = message.get("content", "")
            text += process_content_for_length(content)

        return len(text)

    def render_task_prompt(
        self,
        task: Union[str, Task],
        context: Optional[dict] = None,
        events: Optional[List[dict]] = None,
        force_string_to_message: Optional[bool] = False,
    ) -> Union[str, List[dict]]:
        """Render the prompt for a specific task.

        :param task: The name of the task.
        :param context: The context for rendering the prompt
        :param events: The history of events so far.
        :param force_string_to_message: Force the string message to a user message.
        This should be used for chat models that receive a single message in the task prompt.

        :return: A string, for completion models, or an array of messages for chat models.

        Note that even chat models can have task prompts defined using a string and not an array of messages.
        In this case, the chat model will through an error. If you want to solve this problem, use the
        force_string_to_message parameter to force the string message to a user message.
        """
        prompt = get_prompt(self.config, task)
        if prompt.content:
            task_prompt = self._render_string(prompt.content, context=context, events=events)
            while prompt.max_length is not None and len(task_prompt) > prompt.max_length:
                if not events:
                    raise Exception(f"Prompt exceeds max length of {prompt.max_length} characters even without history")
                # Remove events from the beginning of the history until the prompt fits.
                events = events[1:]
                task_prompt = self._render_string(prompt.content, context=context, events=events)

            # Check if the output should be a user message, for chat models
            if force_string_to_message:
                return [
                    {
                        "type": "user",
                        "content": task_prompt,
                    }
                ]

            return task_prompt
        else:
            if prompt.messages is None:
                return []
            task_messages = self._render_messages(prompt.messages, context=context, events=events)
            task_prompt_length = self._get_messages_text_length(task_messages)
            while prompt.max_length is not None and task_prompt_length > prompt.max_length:
                if not events:
                    raise Exception(f"Prompt exceeds max length of {prompt.max_length} characters even without history")
                # Remove events from the beginning of the history until the prompt fits.
                events = events[1:]
                if prompt.messages is not None:
                    task_messages = self._render_messages(prompt.messages, context=context, events=events)
                else:
                    task_messages = []
                task_prompt_length = self._get_messages_text_length(task_messages)
            return task_messages

    def parse_task_output(self, task: Task, output: str, forced_output_parser: Optional[str] = None) -> str:
        """Parses the output of a task using the configured output parser.

        Args:
            task (Task): The task for which the output is being parsed.
            output (str): The output string to be parsed.
            forced_output_parser (Optional[str]): An optional parser name to force

        Returns:
            str: The parsed text output.
        """
        prompt = get_prompt(self.config, task)
        parser_name = forced_output_parser or prompt.output_parser
        parser_fn = self.output_parsers.get(parser_name)

        if parser_fn:
            parsed_text = parser_fn(output)
        else:
            logging.info("No output parser found for %s", prompt.output_parser)
            parsed_text = output

        return parsed_text

    def has_output_parser(self, task: Task):
        prompt = get_prompt(self.config, task)
        return prompt.output_parser is not None

    def get_stop_tokens(self, task: Union[str, Task]) -> Optional[List[str]]:
        """Return the stop sequence for the given task."""
        prompt = get_prompt(self.config, task)
        return prompt.stop

    def get_max_tokens(self, task: Union[str, Task]) -> Optional[int]:
        """Return the maximum number of tokens for the given task."""
        prompt = get_prompt(self.config, task)
        return prompt.max_tokens

    def register_filter(self, filter_fn: Callable, name: Optional[str] = None):
        """Register a custom filter for the rails configuration."""
        name = name or filter_fn.__name__
        self.env.filters[name] = filter_fn

    def register_output_parser(self, output_parser: Callable, name: str):
        """Register a custom output parser for the rails configuration."""
        self.output_parsers[name] = output_parser

    def register_prompt_context(self, name: str, value_or_fn: Any):
        """Register a value to be included in the prompt context.

        :name: The name of the variable or function that will be used.
        :value_or_fn: The value or function that will be used to generate the value.
        """
        self.prompt_context[name] = value_or_fn
