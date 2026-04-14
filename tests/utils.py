# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, Iterable, List, Optional, Union

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.runtime import (
    create_flow_configs_from_flow_list,
)
from nemoguardrails.colang.v2_x.runtime.statemachine import initialize_state
from nemoguardrails.types import LLMResponse, LLMResponseChunk, UsageInfo
from nemoguardrails.utils import EnhancedJsonEncoder, new_event_dict, new_uuid

# test providers that are known to support token usage reporting
# providers in this list will return token usage data in tests, others won't.
_TEST_PROVIDERS_WITH_TOKEN_USAGE = ["openai", "azure_openai", "nim"]


class FakeLLMModel:
    """Framework-agnostic fake LLM for testing. Implements LLMModel protocol."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        llm_responses: Optional[List[LLMResponse]] = None,
        streaming: bool = False,
        exception: Optional[Exception] = None,
        token_usage: Optional[List[Dict[str, int]]] = None,
        should_return_token_usage: bool = False,
    ):
        if llm_responses is not None:
            self._llm_responses = llm_responses
        elif responses is not None:
            self._llm_responses = [LLMResponse(content=r) for r in responses]
        else:
            self._llm_responses = []
        self.responses = responses or [r.content for r in self._llm_responses]
        self.i = 0
        self.streaming = streaming
        self.exception = exception
        self.token_usage = token_usage
        self.should_return_token_usage = should_return_token_usage

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def provider_name(self) -> Optional[str]:
        return "test"

    @property
    def provider_url(self) -> Optional[str]:
        return None

    def _next_response(self) -> LLMResponse:
        if self.exception:
            raise self.exception
        if self.i >= len(self._llm_responses):
            raise RuntimeError(
                f"No responses available for query number {self.i + 1} in FakeLLMModel. "
                "Most likely, too many LLM calls are made or additional responses need to be provided."
            )
        response = self._llm_responses[self.i]
        self.i += 1
        return response

    def _get_usage(self) -> Optional[UsageInfo]:
        idx = self.i - 1
        if self.token_usage and self.should_return_token_usage and 0 <= idx < len(self.token_usage):
            u = self.token_usage[idx]
            return UsageInfo(
                input_tokens=u.get("prompt_tokens", u.get("input_tokens", 0)),
                output_tokens=u.get("completion_tokens", u.get("output_tokens", 0)),
                total_tokens=u.get("total_tokens", 0),
            )
        return None

    async def generate_async(self, prompt, *, stop=None, **kwargs) -> LLMResponse:
        import copy

        response = copy.copy(self._next_response())
        usage = self._get_usage()
        if usage:
            response.usage = usage
        return response

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        response = self._next_response()
        text = response.content
        chunks = text.split(" ")
        for j, chunk in enumerate(chunks):
            content = chunk + " " if j < len(chunks) - 1 else chunk
            await asyncio.sleep(0.05)
            yield LLMResponseChunk(delta_content=content)
        # Final yield point so concurrent consumers (asyncio.create_task) can
        # process the last chunk before the caller continues after the async for.
        await asyncio.sleep(0)


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
        llm: Optional[Any] = None,
    ):
        """Creates a TestChat instance.

        Args:
            config: The Rails configuration
            llm_completions: The completions that should be generated by the fake LLM.
            streaming: Whether to simulate streaming responses.
            llm_exception: An exception to be raised by the LLM (for testing error handling).
            token_usage: Optional token usage data for simulating token usage reporting.
        """
        if llm is not None:
            self.llm = llm
        elif llm_completions is not None:
            main_model = next((model for model in config.models if model.type == "main"), None)
            should_return_token_usage = bool(main_model and main_model.engine in _TEST_PROVIDERS_WITH_TOKEN_USAGE)

            self.llm = FakeLLMModel(
                responses=llm_completions,
                streaming=streaming,
                exception=llm_exception,
                token_usage=token_usage,
                should_return_token_usage=should_return_token_usage,
            )
        else:
            self.llm = None

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
                            event_created_at=(datetime.now(timezone.utc) + timedelta(milliseconds=1)).isoformat(),
                            action_finished_at=(datetime.now(timezone.utc) + timedelta(milliseconds=1)).isoformat(),
                        ),
                    ]
                )
            elif "type" in msg:
                self.input_events.append(msg)
            else:
                raise ValueError(f"Invalid user message: {msg}. Must be either str or event")
        else:
            raise Exception(f"Invalid colang version: {self.config.colang_version}")

    def bot(self, expected: Union[str, dict, list[dict]]):
        if self.config.colang_version == "1.0":
            result = self.app.generate(messages=self.history)
            assert result, "Did not receive any result"
            assert result["content"] == expected, f"Expected `{expected}` and received `{result['content']}`"
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
                assert output_msg == expected, f"Expected `{expected}` and received `{output_msg}`"
            else:
                if isinstance(expected, dict):
                    expected = [expected]
                assert is_data_in_events(output_events, expected)

        else:
            raise Exception(f"Invalid colang version: {self.config.colang_version}")

    async def bot_async(self, msg: str):
        result = await self.app.generate_async(messages=self.history)
        assert result, "Did not receive any result"
        assert result["content"] == msg, f"Expected `{msg}` and received `{result['content']}`"
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
            return all([event_conforms(s, e) for s, e in zip(value, event_to_test[key])])
        elif value != event_to_test[key]:
            return False

    return True


def event_sequence_conforms(event_subset_list: Iterable[Dict[str, Any]], event_list: Iterable[Dict[str, Any]]) -> bool:
    if len(event_subset_list) != len(event_list):
        raise Exception(f"Different lengths: {len(event_subset_list)} vs {len(event_list)}")

    for subset, event in zip(event_subset_list, event_list):
        if not event_conforms(subset, event):
            raise Exception(f"Mismatch: {subset} vs {event}")

    return True


def any_event_conforms(event_subset: Dict[str, Any], event_list: Iterable[Dict[str, Any]]) -> bool:
    """Returns true iff one of the events in the list conform to the event_subset provided."""
    return any([event_conforms(event_subset, e) for e in event_list])


def is_data_in_events(events: List[Dict[str, Any]], event_data: List[Dict[str, Any]]) -> bool:
    """Returns 'True' if provided data is contained in event."""
    if len(events) != len(event_data):
        return False

    for event, data in zip(events, event_data):
        if not (all(key in event for key in data) and all(data[key] == event[key] for key in data)):
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
