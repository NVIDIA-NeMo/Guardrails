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

import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import nemoguardrails.rails.llm.llmrails
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.cli.chat import extract_scene_text_content, parse_events_inputs
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.utils import new_event_dict, new_uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ChatState:
    state: Optional[State] = None
    waiting_user_input: bool = False
    paused: bool = False
    running_timer_tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    input_events: List[dict] = field(default_factory=list)
    output_events: List[dict] = field(default_factory=list)
    output_state: Optional[State] = None
    events_counter = 0
    first_time: bool = False


class ChatInterface:
    def __init__(self, rails_app: LLMRails):
        self.chat_state = ChatState()
        self.rails_app = rails_app
        self.input_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        asyncio.create_task(self.run())

        # Ensure that the semaphore is assigned to the same loop that we just created
        nemoguardrails.rails.llm.llmrails.process_events_semaphore = asyncio.Semaphore(
            1
        )
        self.output_summary: list[str] = []
        self.should_terminate = False
        self.enable_input = asyncio.Event()
        self.enable_input.set()

    # Start an asynchronous timer
    async def _start_timer(
        self, timer_name: str, delay_seconds: float, action_uid: str
    ):
        await asyncio.sleep(delay_seconds)
        self.chat_state.input_events.append(
            new_event_dict(
                "TimerBotActionFinished",
                action_uid=action_uid,
                is_success=True,
                timer_name=timer_name,
            )
        )
        self.chat_state.running_timer_tasks.pop(action_uid)

        # Pause here until chat is resumed
        while self.chat_state.paused:
            await asyncio.sleep(0.1)

        if self.chat_state.waiting_user_input:
            await self._process_input_events()

    def _add_to_output_summary(self, message: str):
        self.output_summary.append(message)

    def _process_output(self):
        """Helper to process the output events."""

        # We detect any "StartUtteranceBotAction" events, show the message, and
        # generate the corresponding Finished events as new input events.
        for event in self.chat_state.output_events:
            if event["type"] == "StartUtteranceBotAction":
                self._add_to_output_summary(f"{event['script']}")
                self.chat_state.input_events.append(
                    new_event_dict(
                        "UtteranceBotActionStarted",
                        action_uid=event["action_uid"],
                    )
                )
                self.chat_state.input_events.append(
                    new_event_dict(
                        "UtteranceBotActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                        final_script=event["script"],
                    )
                )
            elif event["type"] == "StartGestureBotAction":
                self._add_to_output_summary(f"Gesture: {event['gesture']}")

                self.chat_state.input_events.append(
                    new_event_dict(
                        "GestureBotActionStarted",
                        action_uid=event["action_uid"],
                    )
                )
                self.chat_state.input_events.append(
                    new_event_dict(
                        "GestureBotActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartPostureBotAction":
                self._add_to_output_summary(f"Posture: {event['posture']}")

                self.chat_state.input_events.append(
                    new_event_dict(
                        "PostureBotActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopPostureBotAction":
                self._add_to_output_summary("bot posture (stop)")

                self.chat_state.input_events.append(
                    new_event_dict(
                        "PostureBotActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartVisualInformationSceneAction":
                options = extract_scene_text_content(event["content"])
                self._add_to_output_summary(
                    f"Scene information: {event['title']}{options}"
                )

                self.chat_state.input_events.append(
                    new_event_dict(
                        "VisualInformationSceneActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopVisualInformationSceneAction":
                self._add_to_output_summary(
                    f"scene information (stop): (action_uid={event['action_uid']})"
                )

                self.chat_state.input_events.append(
                    new_event_dict(
                        "VisualInformationSceneActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartVisualFormSceneAction":
                self._add_to_output_summary(f"Scene form: {event['prompt']}")

                self.chat_state.input_events.append(
                    new_event_dict(
                        "VisualFormSceneActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopVisualFormSceneAction":
                self._add_to_output_summary(
                    f"scene form (stop): (action_uid={event['action_uid']})"
                )
                self.chat_state.input_events.append(
                    new_event_dict(
                        "VisualFormSceneActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartVisualChoiceSceneAction":
                options = extract_scene_text_content(event["options"])
                self._add_to_output_summary(f"Scene choice: {event['prompt']}{options}")

                self.chat_state.input_events.append(
                    new_event_dict(
                        "VisualChoiceSceneActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopVisualChoiceSceneAction":
                self._add_to_output_summary(
                    f"scene choice (stop): (action_uid={event['action_uid']})"
                )
                self.chat_state.input_events.append(
                    new_event_dict(
                        "VisualChoiceSceneActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartTimerBotAction":
                action_uid = event["action_uid"]
                timer = self._start_timer(
                    event["timer_name"], event["duration"], action_uid
                )
                # Manage timer tasks
                if action_uid not in self.chat_state.running_timer_tasks:
                    task = asyncio.create_task(timer)
                    self.chat_state.running_timer_tasks.update({action_uid: task})
                self.chat_state.input_events.append(
                    new_event_dict(
                        "TimerBotActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopTimerBotAction":
                action_uid = event["action_uid"]
                if action_uid in self.chat_state.running_timer_tasks:
                    self.chat_state.running_timer_tasks[action_uid].cancel()
                    self.chat_state.running_timer_tasks.pop(action_uid)

            elif event["type"] == "TimerBotActionFinished":
                action_uid = event["action_uid"]
                if action_uid in self.chat_state.running_timer_tasks:
                    self.chat_state.running_timer_tasks[action_uid].cancel()
                    self.chat_state.running_timer_tasks.pop(action_uid)
            elif event["type"].endswith("Exception"):
                if event["type"].endswith("Exception"):
                    self._add_to_output_summary(f"Event: {event}")
            elif event["type"] == "LocalAsyncCounter":
                # self._add_to_output_summary(f"Event: {event}")
                pass
            else:
                self._add_to_output_summary(f"Event: {event['type']}")

        # TODO: deserialize the output state
        # state = State.from_dict(output_state)
        # Simulate serialization for testing
        # data = pickle.dumps(output_state)
        # output_state = pickle.loads(data)
        self.chat_state.state = self.chat_state.output_state

    async def _process_input_events(self):
        while self.chat_state.input_events or self.chat_state.first_time:
            # We need to copy input events to prevent race condition
            input_events_copy = self.chat_state.input_events.copy()
            self.chat_state.input_events = []
            (
                self.chat_state.output_events,
                self.chat_state.output_state,
            ) = await self.rails_app.process_events_async(
                input_events_copy, self.chat_state.state
            )

            self._process_output()
            # If we don't have a check task, we start it
            if self.check_task is None:
                self.check_task = asyncio.create_task(self._check_local_async_actions())

            self.chat_state.first_time = False

    async def _check_local_async_actions(self):
        while True:
            # We only run the check when we wait for user input, but not the first time.
            if not self.chat_state.waiting_user_input or self.chat_state.first_time:
                await asyncio.sleep(0.1)
                continue

            if len(self.chat_state.input_events) == 0:
                self.chat_state.input_events = [new_event_dict("CheckLocalAsync")]

            # We need to copy input events to prevent race condition
            input_events_copy = self.chat_state.input_events.copy()
            self.chat_state.input_events = []
            (
                self.chat_state.output_events,
                self.chat_state.output_state,
            ) = await self.rails_app.process_events_async(
                input_events_copy, self.chat_state.state
            )

            # Process output_events and potentially generate new input_events
            self._process_output()

            if (
                len(self.chat_state.output_events) == 1
                and self.chat_state.output_events[0]["type"] == "LocalAsyncCounter"
                and self.chat_state.output_events[0]["counter"] == 0
            ):
                # If there are no pending actions, we stop
                self.check_task.cancel()
                self.check_task = None
                self.enable_input.set()
                return

            self.chat_state.output_events.clear()
            await asyncio.sleep(0.2)

    async def run(self):
        # Start the task for checking async actions
        self.check_task = asyncio.create_task(self._check_local_async_actions())

        self.chat_state.first_time = True
        while not self.should_terminate:
            if self.chat_state.first_time:
                self.chat_state.input_events = []
            else:
                self.chat_state.waiting_user_input = True
                await self.enable_input.wait()

                user_message = ""
                if not self.input_queue.empty():
                    user_message = self.input_queue.get_nowait()
                self.enable_input.clear()
                self.chat_state.events_counter = 0
                self.chat_state.waiting_user_input = False
                if user_message == "":
                    self.chat_state.input_events = [new_event_dict("CheckLocalAsync")]
                elif user_message.startswith("/"):
                    # Non-UtteranceBotAction actions
                    event_input = user_message.lstrip("/")
                    event = parse_events_inputs(event_input)
                    if event is None:
                        self._add_to_output_summary(f"Invalid event: {event_input}")
                    else:
                        self.chat_state.input_events = [event]
                else:
                    action_uid = new_uuid()
                    self.chat_state.input_events = [
                        new_event_dict(
                            "UtteranceUserActionStarted",
                            action_uid=action_uid,
                        ),
                        new_event_dict(
                            "UtteranceUserActionFinished",
                            final_transcript=user_message,
                            action_uid=action_uid,
                            is_success=True,
                        ),
                    ]

            await self._process_input_events()

    def user(self, message: str) -> None:
        self.input_queue.put_nowait(message)

    async def process(self, message: str, wait_time=1.0) -> str:
        self.output_summary = []
        self.user(message)
        await asyncio.sleep(wait_time)
        response = "\n".join(self.output_summary)
        return response
