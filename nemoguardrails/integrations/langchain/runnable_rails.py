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

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import Tool

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.utils import async_wrap
from nemoguardrails.rails.llm.options import GenerationOptions


class RunnableRails(Runnable[Input, Output]):
    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[BaseLanguageModel] = None,
        tools: Optional[List[Tool]] = None,
        passthrough: bool = True,
        runnable: Optional[Runnable] = None,
        input_key: str = "input",
        output_key: str = "output",
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.passthrough = passthrough
        self.passthrough_runnable = runnable
        self.passthrough_user_input_key = input_key
        self.passthrough_bot_output_key = output_key
        self.verbose = verbose
        self.config: Optional[RunnableConfig] = None

        # We override the config passthrough.
        config.passthrough = passthrough

        self.rails = LLMRails(config=config, llm=llm, verbose=verbose)

        if tools:
            # When tools are used, we disable the passthrough mode.
            self.passthrough = False

            for tool in tools:
                self.rails.register_action(tool, tool.name)

        # If we have a passthrough Runnable, we need to register a passthrough fn
        # that will call it
        if self.passthrough_runnable:
            self._init_passthrough_fn()

    def _init_passthrough_fn(self):
        """Initialize the passthrough function for the LLM rails instance."""

        async def passthrough_fn(context: dict, events: List[dict]):
            # First, we fetch the input from the context
            _input = context.get("passthrough_input")
            async_wrapped_invoke = async_wrap(self.passthrough_runnable.invoke)
            _output = await async_wrapped_invoke(_input, self.config, **self.kwargs)

            # If the output is a string, we consider it to be the output text
            if isinstance(_output, str):
                text = _output
            else:
                text = _output.get(self.passthrough_bot_output_key)

            return text, _output

        self.rails.llm_generation_actions.passthrough_fn = passthrough_fn

    def __or__(self, other):
        if isinstance(other, BaseLanguageModel):
            self.llm = other
            self.rails.update_llm(other)

        elif isinstance(other, Runnable):
            self.passthrough_runnable = other
            self.passthrough = True
            self._init_passthrough_fn()

        return self

    @property
    def InputType(self) -> Any:
        return Any

    @property
    def OutputType(self) -> Any:
        """The type of the output of this runnable as a type annotation."""
        return Any

    def _transform_input_to_rails_format(self, _input):
        messages = []

        if self.passthrough and self.passthrough_runnable:
            # First, we add the raw input in the context variable $passthrough_input
            if isinstance(_input, str):
                text_input = _input
            else:
                text_input = _input.get(self.passthrough_user_input_key)

            messages = [
                {
                    "role": "context",
                    "content": {
                        "passthrough_input": _input,
                        # We also set all the input variables as top level context variables
                        **(_input if isinstance(_input, dict) else {}),
                    },
                },
                {
                    "role": "user",
                    "content": text_input,
                },
            ]

        else:
            if isinstance(_input, ChatPromptValue):
                for msg in _input.messages:
                    if isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
                    elif isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, SystemMessage):
                        messages.append({"role": "system", "content": msg.content})
            elif isinstance(_input, StringPromptValue):
                messages.append({"role": "user", "content": _input.text})
            elif isinstance(_input, dict):
                # If we're provided a dict, then the `input` key will be the one passed
                # to the guardrails.
                if "input" not in _input:
                    raise Exception("No `input` key found in the input dictionary.")

                # TODO: add support for putting the extra keys as context
                user_input = _input["input"]
                if isinstance(user_input, str):
                    messages.append({"role": "user", "content": user_input})
                elif isinstance(user_input, list):
                    # If it's a list of messages
                    for msg in user_input:
                        assert "role" in msg
                        assert "content" in msg
                        messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )
                else:
                    raise Exception(
                        f"Can't handle input of type {type(user_input).__name__}"
                    )

                if "context" in _input:
                    if not isinstance(_input["context"], dict):
                        raise ValueError(
                            "The input `context` key for `RunnableRails` must be a dict."
                        )
                    messages = [
                        {"role": "context", "content": _input["context"]}
                    ] + messages

            else:
                raise Exception(f"Can't handle input of type {type(_input).__name__}")

        return messages

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Invoke this runnable synchronously."""
        input_messages = self._transform_input_to_rails_format(input)
        self.config = config
        self.kwargs = kwargs
        res = self.rails.generate(
            messages=input_messages, options=GenerationOptions(output_vars=True)
        )
        context = res.output_data
        result = res.response

        # If more than one message is returned, we only take the first one.
        # This can happen for advanced use cases, e.g., when the LLM could predict
        # multiple function calls at the same time. We'll deal with these later.
        if isinstance(result, list):
            result = result[0]

        if self.passthrough and self.passthrough_runnable:
            passthrough_output = context.get("passthrough_output")

            # If a rail was triggered (input or dialog), the passthrough_output
            # will not be set. In this case, we only set the output key to the
            # message that was received from the guardrail configuration.
            if passthrough_output is None:
                passthrough_output = {
                    self.passthrough_bot_output_key: result["content"]
                }

            bot_message = context.get("bot_message")

            # We make sure that, if the output rails altered the bot message, we
            # replace it in the passthrough_output
            if isinstance(passthrough_output, str):
                passthrough_output = bot_message
            elif isinstance(passthrough_output, dict):
                passthrough_output[self.passthrough_bot_output_key] = bot_message

            return passthrough_output
        else:
            if isinstance(input, ChatPromptValue):
                return AIMessage(content=result["content"])
            elif isinstance(input, StringPromptValue):
                if isinstance(result, dict):
                    return result["content"]
                else:
                    return result
            elif isinstance(input, dict):
                user_input = input["input"]
                if isinstance(user_input, str):
                    return {"output": result["content"]}
                elif isinstance(user_input, list):
                    return {"output": result}
            else:
                raise ValueError(f"Unexpected input type: {type(input)}")
