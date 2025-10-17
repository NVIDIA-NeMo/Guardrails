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
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.runtime import (
    create_flow_configs_from_flow_list,
)
from nemoguardrails.colang.v2_x.runtime.statemachine import initialize_state
from nemoguardrails.utils import EnhancedJsonEncoder, new_event_dict, new_uuid

# test providers that are known to support token usage reporting during streaming
# use this to simulate realistic behavior in tests: providers in this list will
# return token usage data when stream_usage=True is passed, others won't.
_TEST_PROVIDERS_WITH_TOKEN_USAGE_SUPPORT = ["openai", "azure_openai", "nim"]


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    responses: List
    i: int = 0
    streaming: bool = False
    exception: Optional[Exception] = None
    token_usage: Optional[List[Dict[str, int]]] = None  # Token usage per response
    should_enable_stream_usage: bool = False

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # If an exception is set, raise it
        if self.exception:
            raise self.exception

        if self.i >= len(self.responses):
            raise RuntimeError(
                f"No responses available for query number {self.i + 1} in FakeLLM. "
                "Most likely, too many LLM calls are made or additional responses need to be provided."
            )

        response = self.responses[self.i]
        self.i += 1
        return response

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # If an exception is set, raise it
        if self.exception:
            raise self.exception

        if self.i >= len(self.responses):
            raise RuntimeError(
                f"No responses available for query number {self.i + 1} in FakeLLM. "
                "Most likely, too many LLM calls are made or additional responses need to be provided."
            )

        response = self.responses[self.i]

        self.i += 1

        if self.streaming and run_manager:
            # To mock streaming, we just split in chunk by spaces
            chunks = response.split(" ")
            for i in range(len(chunks)):
                if i < len(chunks) - 1:
                    chunk = chunks[i] + " "
                else:
                    chunk = chunks[i]

                await asyncio.sleep(0.05)
                await run_manager.on_llm_new_token(token=chunk, chunk=chunk)

        return response

    def _get_token_usage_for_response(
        self, response_index: int, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get token usage data for the given response index if conditions are met."""

        llm_output = {}
        if (
            self.token_usage
            and response_index >= 0
            and response_index < len(self.token_usage)
            and (kwargs.get("stream_usage", False) or self.should_enable_stream_usage)
        ):
            llm_output = {"token_usage": self.token_usage[response_index]}
        return llm_output

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Override _generate to provide token usage in LLMResult."""

        from langchain.schema import Generation, LLMResult

        generations = [
            [Generation(text=self._call(prompt, stop, run_manager, **kwargs))]
            for prompt in prompts
        ]

        llm_output = self._get_token_usage_for_response(self.i - 1, kwargs)
        return LLMResult(generations=generations, llm_output=llm_output)

    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Override _agenerate to provide token usage in LLMResult."""
        from langchain.schema import Generation, LLMResult

        generations = [
            [Generation(text=await self._acall(prompt, stop, run_manager, **kwargs))]
            for prompt in prompts
        ]

        llm_output = self._get_token_usage_for_response(self.i - 1, kwargs)
        return LLMResult(generations=generations, llm_output=llm_output)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class TestChat:
    """Helper class for easily writing tests.

    Usage:
        config = RailsConfig.from_path(...)
        chat = TestChat(
            config,
            llm_completions=[
                "Hello! How can I help you today?",
            ],
        )

        chat.user("Hello! How are you?")
        chat.bot("Hello! How can I help you today?")

    """

    # Tell pytest that this class is not meant to hold tests.
    __test__ = False

    def __init__(
        self,
        config: Union[str, RailsConfig],
        llm_completions: Optional[List[str]] = None,
        streaming: bool = False,
        llm_exception: Optional[Exception] = None,
        token_usage: Optional[List[Dict[str, int]]] = None,
    ):
        """Creates a TestChat instance.

        Args:
            config: The Rails configuration
            llm_completions: The completions that should be generated by the fake LLM.
            streaming: Whether to simulate streaming responses.
            llm_exception: An exception to be raised by the LLM (for testing error handling).
            token_usage: Optional token usage data to simulate stream_usage=True behavior.
        """
        self.llm = None
        if llm_completions is not None:
            # check if we should simulate stream_usage=True behavior
            # this mirrors the logic in LLMRails._prepare_model_kwargs
            should_enable_stream_usage = False
            if config.streaming:
                main_model = next(
                    (model for model in config.models if model.type == "main"), None
                )
                if (
                    main_model
                    and main_model.engine in _TEST_PROVIDERS_WITH_TOKEN_USAGE_SUPPORT
                ):
                    should_enable_stream_usage = True

            self.llm = FakeLLM(
                responses=llm_completions,
                streaming=streaming,
                token_usage=token_usage,
                should_enable_stream_usage=should_enable_stream_usage,
            )
            if llm_exception:
                self.llm.exception = llm_exception

        self.config = config
        self.app = LLMRails(config, llm=self.llm)

        # Track the conversation for v1.0
        self.history = []
        self.streaming = streaming

        # Track the conversation for v2.x
        self.input_events = []
        self.state = None

        # For 2.x, we start the main flow when initializing by providing a empty state
        if self.config.colang_version == "2.x":
            self.app.runtime.disable_async_execution = True
            _, self.state = self.app.process_events(
                [],
                self.state,
            )

    def user(self, msg: Union[str, dict]):
        if self.config.colang_version == "1.0":
            self.history.append({"role": "user", "content": msg})
        elif self.config.colang_version == "2.x":
            if isinstance(msg, str):
                uid = new_uuid()
                self.input_events.extend(
                    [
                        new_event_dict("UtteranceUserActionStarted", action_uid=uid),
                        new_event_dict(
                            "UtteranceUserActionFinished",
                            final_transcript=msg,
                            action_uid=uid,
                            is_success=True,
                            event_created_at=(
                                datetime.now(timezone.utc) + timedelta(milliseconds=1)
                            ).isoformat(),
                            action_finished_at=(
                                datetime.now(timezone.utc) + timedelta(milliseconds=1)
                            ).isoformat(),
                        ),
                    ]
                )
            elif "type" in msg:
                self.input_events.append(msg)
            else:
                raise ValueError(
                    f"Invalid user message: {msg}. Must be either str or event"
                )
        else:
            raise Exception(f"Invalid colang version: {self.config.colang_version}")

    def bot(self, expected: Union[str, dict, list[dict]]):
        if self.config.colang_version == "1.0":
            result = self.app.generate(messages=self.history)
            assert result, "Did not receive any result"
            assert (
                result["content"] == expected
            ), f"Expected `{expected}` and received `{result['content']}`"
            self.history.append(result)

        elif self.config.colang_version == "2.x":
            output_msgs = []
            output_events = []
            while self.input_events:
                event = self.input_events.pop(0)
                out_events, output_state = self.app.process_events([event], self.state)
                output_events.extend(out_events)

                # We detect any "StartUtteranceBotAction" events, show the message, and
                # generate the corresponding Finished events as new input events.
                for event in out_events:
                    if event["type"] == "StartUtteranceBotAction":
                        output_msgs.append(event["script"])
                        self.input_events.append(
                            new_event_dict(
                                "UtteranceBotActionStarted",
                                action_uid=event["action_uid"],
                            )
                        )
                        self.input_events.append(
                            new_event_dict(
                                "UtteranceBotActionStarted",
                                action_uid=event["action_uid"],
                            )
                        )
                        self.input_events.append(
                            new_event_dict(
                                "UtteranceBotActionFinished",
                                action_uid=event["action_uid"],
                                is_success=True,
                                final_script=event["script"],
                            )
                        )

                self.state = output_state

            output_msg = "\n".join(output_msgs)
            if isinstance(expected, str):
                assert (
                    output_msg == expected
                ), f"Expected `{expected}` and received `{output_msg}`"
            else:
                if isinstance(expected, dict):
                    expected = [expected]
                assert is_data_in_events(output_events, expected)

        else:
            raise Exception(f"Invalid colang version: {self.config.colang_version}")

    async def bot_async(self, msg: str):
        result = await self.app.generate_async(messages=self.history)
        assert result, "Did not receive any result"
        assert (
            result["content"] == msg
        ), f"Expected `{msg}` and received `{result['content']}`"
        self.history.append(result)

    def __rshift__(self, msg: Union[str, dict]):
        self.user(msg)

    def __lshift__(self, msg: str):
        self.bot(msg)


def clean_events(events: List[dict]):
    """Removes private context parameters (starting with '_') from a list of events
    generated by the runtime for a test case.

    If the context update event will be empty after removing all private context parameters,
    the entire event is removed from the list.

    :param events: The list of events generated by the runtime for a test case.
    """
    for e in events:
        if e["type"] == "ContextUpdate":
            for key in list(e["data"].keys()):
                if key.startswith("_"):
                    del e["data"][key]
    for e in events[:]:
        if e["type"] == "ContextUpdate" and len(e["data"]) == 0:
            events.remove(e)


def event_conforms(event_subset: Dict[str, Any], event_to_test: Dict[str, Any]) -> bool:
    """Tests if the `event_to_test` conforms to the event_subset. Conforming means that for all key,value paris in `event_subset` the value has to match."""
    for key, value in event_subset.items():
        if key not in event_to_test:
            return False

        if isinstance(value, dict) and isinstance(event_to_test[key], dict):
            if not event_conforms(value, event_to_test[key]):
                return False
        elif isinstance(value, list) and isinstance(event_to_test[key], list):
            return all(
                [event_conforms(s, e) for s, e in zip(value, event_to_test[key])]
            )
        elif value != event_to_test[key]:
            return False

    return True


def event_sequence_conforms(
    event_subset_list: Iterable[Dict[str, Any]], event_list: Iterable[Dict[str, Any]]
) -> bool:
    if len(event_subset_list) != len(event_list):
        raise Exception(
            f"Different lengths: {len(event_subset_list)} vs {len(event_list)}"
        )

    for subset, event in zip(event_subset_list, event_list):
        if not event_conforms(subset, event):
            raise Exception(f"Mismatch: {subset} vs {event}")

    return True


def any_event_conforms(
    event_subset: Dict[str, Any], event_list: Iterable[Dict[str, Any]]
) -> bool:
    """Returns true iff one of the events in the list conform to the event_subset provided."""
    return any([event_conforms(event_subset, e) for e in event_list])


def is_data_in_events(
    events: List[Dict[str, Any]], event_data: List[Dict[str, Any]]
) -> bool:
    """Returns 'True' if provided data is contained in event."""
    if len(events) != len(event_data):
        return False

    for event, data in zip(events, event_data):
        if not (
            all(key in event for key in data)
            and all(data[key] == event[key] for key in data)
        ):
            return False
    return True


def _init_state(colang_content, yaml_content: Optional[str] = None) -> State:
    config = create_flow_configs_from_flow_list(
        parse_colang_file(
            filename="",
            content=colang_content,
            include_source_mapping=True,
            version="2.x",
        )["flows"]
    )

    rails_config = None
    if yaml_content:
        rails_config = RailsConfig.from_content(colang_content, yaml_content)
    json.dump(config, sys.stdout, indent=4, cls=EnhancedJsonEncoder)
    state = State(flow_states=[], flow_configs=config, rails_config=rails_config)
    initialize_state(state)
    print("---------------------------------")
    json.dump(state.flow_configs, sys.stdout, indent=4, cls=EnhancedJsonEncoder)

    return state
