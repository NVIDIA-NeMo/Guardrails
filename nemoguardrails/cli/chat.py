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
from typing import Dict, List, Optional, cast

import aiohttp
from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.cli import debugger
from nemoguardrails.colang.v2_x.runtime.eval import eval_expression
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.logging import verbose
from nemoguardrails.logging.verbose import console
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.utils import new_event_dict, new_uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"

enable_input = asyncio.Event()
enable_input.set()


async def _run_chat_v1_0(
    config_path: Optional[str] = None,
    verbose: bool = False,
    streaming: bool = False,
    server_url: Optional[str] = None,
    config_id: Optional[str] = None,
):
    """Asynchronously run a chat session in the terminal.

    Args:
        config_path (Optional[str]): The path to the configuration file. Defaults to None.
        verbose (bool): Whether to run in verbose mode. Defaults to False.
        streaming (bool): Whether to enable streaming mode. Defaults to False.
        server_url (Optional[str]): The URL of the chat server. Defaults to None.
        config_id (Optional[str]): The configuration ID. Defaults to None.
    """
    if config_path is None and server_url is None:
        raise RuntimeError(
            "At least one of `config_path` or `server-url` must be provided."
        )

    if not server_url:
        rails_config = RailsConfig.from_path(config_path)
        rails_app = LLMRails(rails_config, verbose=verbose)
        if streaming and not rails_config.streaming_supported:
            console.print(
                f"WARNING: The config `{config_path}` does not support streaming. "
                "Falling back to normal mode."
            )
            streaming = False
    else:
        rails_app = None

    history = []
    # And go into the default listening loop.
    while True:
        console.print("")
        user_message = input("> ")

        history.append({"role": "user", "content": user_message})

        if not server_url:
            # If we have streaming from a locally loaded config, we initialize the handler.
            if streaming and not server_url and rails_app.main_llm_supports_streaming:
                streaming_handler = StreamingHandler(enable_print=True)
            else:
                streaming_handler = None

            bot_message = await rails_app.generate_async(
                messages=history, streaming_handler=streaming_handler
            )

            if not streaming or not rails_app.main_llm_supports_streaming:
                # We print bot messages in green.
                console.print("[green]" + f"{bot_message['content']}" + "[/]")
        else:
            data = {
                "config_id": config_id,
                "messages": history,
                "stream": streaming,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server_url}/v1/chat/completions",
                    json=data,
                ) as response:
                    # If the response is streaming, we show each chunk as it comes
                    if response.headers.get("Transfer-Encoding") == "chunked":
                        bot_message_text = ""
                        async for chunk in response.content.iter_any():
                            chunk = chunk.decode("utf-8")
                            console.print("[green]" + f"{chunk}" + "[/]", end="")
                            bot_message_text += chunk
                        console.print("")

                        bot_message = {"role": "assistant", "content": bot_message_text}
                    else:
                        result = await response.json()
                        bot_message = result["messages"][0]

                        # We print bot messages in green.
                        console.print("[green]" + f"{bot_message['content']}" + "[/]")

        history.append(bot_message)


@dataclass
class ChatState:
    state: Optional[State] = None
    waiting_user_input: bool = False
    paused: bool = False
    running_timer_tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    input_events: List[dict] = field(default_factory=list)
    output_events: List[dict] = field(default_factory=list)
    output_state: Optional[State] = None
    session: PromptSession = PromptSession()
    status = console.status("[bold green]Working ...[/]")
    events_counter = 0
    first_time: bool = False


async def _run_chat_v2_x(rails_app: LLMRails):
    """Simple chat loop for v2.x using the stateful events API."""
    chat_state = ChatState()

    def watcher(*args):
        nonlocal chat_state
        chat_state.events_counter += 1
        chat_state.status.update(
            f"[bold green]Working ({chat_state.events_counter} events processed)...[/]"
        )

    rails_app.runtime.watchers.append(watcher)

    # Set the runtime for the debugger to work correctly.
    debugger.set_chat_state(chat_state)
    debugger.set_runtime(cast(RuntimeV2_x, rails_app.runtime))

    # Start an asynchronous timer
    async def _start_timer(timer_name: str, delay_seconds: float, action_uid: str):
        nonlocal chat_state
        await asyncio.sleep(delay_seconds)
        chat_state.input_events.append(
            new_event_dict(
                "TimerBotActionFinished",
                action_uid=action_uid,
                is_success=True,
                timer_name=timer_name,
            )
        )
        chat_state.running_timer_tasks.pop(action_uid)

        # Pause here until chat is resumed
        while chat_state.paused:
            await asyncio.sleep(0.1)

        if chat_state.waiting_user_input:
            await _process_input_events()

    def _process_output():
        """Helper to process the output events."""
        nonlocal chat_state

        # We detect any "StartUtteranceBotAction" events, show the message, and
        # generate the corresponding Finished events as new input events.
        for event in chat_state.output_events:
            if event["type"] == "StartUtteranceBotAction":
                # We print bot messages in green.
                if not verbose.verbose_mode_enabled:
                    console.print(f"\n[green]{event['script']}[/]\n")
                else:
                    if not verbose.debug_mode_enabled:
                        console.print(f"\n[#f0f0f0 on #008800]{event['script']}[/]\n")
                    else:
                        console.print(
                            "[black on #008800]"
                            + f"bot utterance: {event['script']}"
                            + "[/]"
                        )

                chat_state.input_events.append(
                    new_event_dict(
                        "UtteranceBotActionStarted",
                        action_uid=event["action_uid"],
                    )
                )
                chat_state.input_events.append(
                    new_event_dict(
                        "UtteranceBotActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                        final_script=event["script"],
                    )
                )
            elif event["type"] == "StartGestureBotAction":
                # We print gesture messages in green.
                if not verbose.verbose_mode_enabled:
                    console.print(
                        "[black on blue]" + f"Gesture: {event['gesture']}" + "[/]"
                    )
                else:
                    console.print(
                        "[black on blue]" + f"bot gesture: {event['gesture']}" + "[/]"
                    )

                chat_state.input_events.append(
                    new_event_dict(
                        "GestureBotActionStarted",
                        action_uid=event["action_uid"],
                    )
                )
                chat_state.input_events.append(
                    new_event_dict(
                        "GestureBotActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartPostureBotAction":
                # We print posture messages in green.
                if not verbose.verbose_mode_enabled:
                    console.print(
                        "[black on blue]" + f"Posture: {event['posture']}." + "[/]"
                    )
                else:
                    console.print(
                        "[black on blue]"
                        + f"bot posture (start): (posture={event['posture']}, action_uid={event['action_uid']}))"
                        + "[/]"
                    )
                chat_state.input_events.append(
                    new_event_dict(
                        "PostureBotActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopPostureBotAction":
                if verbose.verbose_mode_enabled:
                    console.print(
                        "[black on blue]"
                        + f"bot posture (stop): (action_uid={event['action_uid']})"
                        + "[/]"
                    )

                chat_state.input_events.append(
                    new_event_dict(
                        "PostureBotActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartVisualInformationSceneAction":
                # We print scene messages in green.
                if not verbose.verbose_mode_enabled:
                    options = extract_scene_text_content(event["content"])
                    console.print(
                        "[black on magenta]"
                        + f"Scene information: {event['title']}{options}"
                        + "[/]"
                    )
                else:
                    console.print(
                        "[black on magenta]"
                        + f"scene information (start): (title={event['title']}, action_uid={event['action_uid']}, content={event['content']})"
                        + "[/]"
                    )

                chat_state.input_events.append(
                    new_event_dict(
                        "VisualInformationSceneActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopVisualInformationSceneAction":
                if verbose.verbose_mode_enabled:
                    console.print(
                        "[black on magenta]"
                        + f"scene information (stop): (action_uid={event['action_uid']})"
                        + "[/]"
                    )

                chat_state.input_events.append(
                    new_event_dict(
                        "VisualInformationSceneActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartVisualFormSceneAction":
                # We print scene messages in green.
                if not verbose.verbose_mode_enabled:
                    console.print(
                        "[black on magenta]" + f"Scene form: {event['prompt']}" + "[/]"
                    )
                else:
                    console.print(
                        "[black on magenta]"
                        + f"scene form (start): (prompt={event['prompt']}, action_uid={event['action_uid']}, inputs={event['inputs']})"
                        + "[/]"
                    )
                chat_state.input_events.append(
                    new_event_dict(
                        "VisualFormSceneActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopVisualFormSceneAction":
                if verbose.verbose_mode_enabled:
                    console.print(
                        "[black on magenta]"
                        + f"scene form (stop): (action_uid={event['action_uid']})"
                        + "[/]"
                    )
                chat_state.input_events.append(
                    new_event_dict(
                        "VisualFormSceneActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartVisualChoiceSceneAction":
                # We print scene messages in green.
                if not verbose.verbose_mode_enabled:
                    options = extract_scene_text_content(event["options"])
                    console.print(
                        "[black on magenta]"
                        + f"Scene choice: {event['prompt']}{options}"
                        + "[/]"
                    )
                else:
                    console.print(
                        "[black on magenta]"
                        + f"scene choice (start): (prompt={event['prompt']}, action_uid={event['action_uid']}, options={event['options']})"
                        + "[/]"
                    )
                chat_state.input_events.append(
                    new_event_dict(
                        "VisualChoiceSceneActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopVisualChoiceSceneAction":
                if verbose.verbose_mode_enabled:
                    console.print(
                        "[black on magenta]"
                        + f"scene choice (stop): (action_uid={event['action_uid']})"
                        + "[/]"
                    )
                chat_state.input_events.append(
                    new_event_dict(
                        "VisualChoiceSceneActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                    )
                )

            elif event["type"] == "StartTimerBotAction":
                action_uid = event["action_uid"]
                timer = _start_timer(event["timer_name"], event["duration"], action_uid)
                # Manage timer tasks
                if action_uid not in chat_state.running_timer_tasks:
                    task = asyncio.create_task(timer)
                    chat_state.running_timer_tasks.update({action_uid: task})
                chat_state.input_events.append(
                    new_event_dict(
                        "TimerBotActionStarted",
                        action_uid=event["action_uid"],
                    )
                )

            elif event["type"] == "StopTimerBotAction":
                action_uid = event["action_uid"]
                if action_uid in chat_state.running_timer_tasks:
                    chat_state.running_timer_tasks[action_uid].cancel()
                    chat_state.running_timer_tasks.pop(action_uid)

            elif event["type"] == "TimerBotActionFinished":
                action_uid = event["action_uid"]
                if action_uid in chat_state.running_timer_tasks:
                    chat_state.running_timer_tasks[action_uid].cancel()
                    chat_state.running_timer_tasks.pop(action_uid)
            elif event["type"].endswith("Exception"):
                if event["type"].endswith("Exception"):
                    console.print("[red]" + f"Event: {event}" + "[/]")
            elif event["type"] == "LocalAsyncCounter":
                # if verbose.verbose_mode_enabled:
                #     console.print(Styles.GREY + f"Event: {event}" + "[/]")
                pass
            else:
                console.print(f"Event: {event['type']}")

        # TODO: deserialize the output state
        # state = State.from_dict(output_state)
        # Simulate serialization for testing
        # data = pickle.dumps(output_state)
        # output_state = pickle.loads(data)
        chat_state.state = chat_state.output_state

    async def _check_local_async_actions():
        nonlocal chat_state, check_task

        while True:
            # We only run the check when we wait for user input, but not the first time.
            if not chat_state.waiting_user_input or chat_state.first_time:
                await asyncio.sleep(0.1)
                continue

            if len(chat_state.input_events) == 0:
                chat_state.input_events = [new_event_dict("CheckLocalAsync")]

            # We need to copy input events to prevent race condition
            input_events_copy = chat_state.input_events.copy()
            chat_state.input_events = []
            (
                chat_state.output_events,
                chat_state.output_state,
            ) = await rails_app.process_events_async(
                input_events_copy, chat_state.state
            )

            # Process output_events and potentially generate new input_events
            _process_output()

            if (
                len(chat_state.output_events) == 1
                and chat_state.output_events[0]["type"] == "LocalAsyncCounter"
                and chat_state.output_events[0]["counter"] == 0
            ):
                # If there are no pending actions, we stop
                check_task.cancel()
                check_task = None
                debugger.set_output_state(chat_state.output_state)
                chat_state.status.stop()
                enable_input.set()
                return

            chat_state.output_events.clear()

            await asyncio.sleep(0.2)

    async def _process_input_events():
        nonlocal chat_state, check_task
        while chat_state.input_events or chat_state.first_time:
            # We need to copy input events to prevent race condition
            input_events_copy = chat_state.input_events.copy()
            chat_state.input_events = []
            (
                chat_state.output_events,
                chat_state.output_state,
            ) = await rails_app.process_events_async(
                input_events_copy, chat_state.state
            )
            debugger.set_output_state(chat_state.output_state)

            _process_output()
            # If we don't have a check task, we start it
            if check_task is None:
                check_task = asyncio.create_task(_check_local_async_actions())

            chat_state.first_time = False

    # Start the task for checking async actions
    check_task = asyncio.create_task(_check_local_async_actions())

    # And go into the default listening loop.
    chat_state.first_time = True
    with patch_stdout(raw=True):
        while True:
            if chat_state.first_time:
                chat_state.input_events = []
            else:
                chat_state.waiting_user_input = True
                await enable_input.wait()

                user_message: str = await chat_state.session.prompt_async(
                    HTML("<prompt>\n> </prompt>"),
                    style=Style.from_dict(
                        {
                            "prompt": "fg:#ffff00",
                            "": "fg:#ffff00",
                        }
                    ),
                )
                enable_input.clear()
                chat_state.events_counter = 0
                chat_state.status.start()
                chat_state.waiting_user_input = False
                if user_message == "":
                    chat_state.input_events = [new_event_dict("CheckLocalAsync")]

                # System commands
                elif user_message.startswith("!"):
                    command = user_message[1:]
                    debugger.run_command(command)
                    chat_state.status.stop()
                    enable_input.set()
                    continue

                elif user_message.startswith("/"):
                    # Non-UtteranceBotAction actions
                    event_input = user_message.lstrip("/")
                    event = parse_events_inputs(event_input)
                    if event is None:
                        console.print(
                            "[white on red]" + f"Invalid event: {event_input}" + "[/]"
                        )
                    else:
                        chat_state.input_events = [event]
                else:
                    action_uid = new_uuid()
                    chat_state.input_events = [
                        new_event_dict(
                            "UtteranceUserActionStarted",
                            action_uid=action_uid,
                        ),
                        new_event_dict(
                            "UtteranceUserActionFinished",
                            final_transcript=user_message,
                            action_uid=new_uuid(),
                            is_success=True,
                        ),
                    ]

            await _process_input_events()


def extract_scene_text_content(content_list: List[dict]) -> str:
    """Extract the text content of a scene event as a string."""
    content = ""
    content = "\n".join([item["text"] for item in content_list if "text" in item])
    if content:
        content = "\n" + content
    return content


def parse_events_inputs(input_str: str) -> Optional[dict]:
    """Parses a event string and creates an event dictionary."""
    # Split the string to extract the event type and the parameters
    event_parts = input_str.split("(", 1)
    if len(event_parts) == 1:
        event_type = event_parts[0]
        params_str = None
    elif len(event_parts) >= 2:
        event_type, params_str = event_parts
        params_str = params_str.rstrip(")")
    else:
        return None

    adjusted_type = event_type
    parts = event_type.split(".")
    if len(parts) == 1:
        adjusted_type = parts[-1]
    elif len(parts) == 2:
        action = parts[-1]
        rest = parts[:-1]
        adjusted_type = "".join(rest) + action
    else:
        return None

    # Prepare the dictionary with the event type
    event_dict = {"type": adjusted_type}

    # If there are parameters to process
    if params_str:
        # Parse the parameters string
        params_dict = {}
        # Split parameters by commas not enclosed in brackets (to handle nested structures)
        params = []
        bracket_level = 0
        current_param: List[str] = []
        for char in params_str:
            if char in ("{", "[", "("):
                bracket_level += 1
            elif char in ("}", "]", ")"):
                bracket_level -= 1
            if char == "," and bracket_level == 0:
                params.append("".join(current_param).strip())
                current_param = []
            else:
                current_param.append(char)
        if current_param:
            params.append("".join(current_param).strip())

        # Process each parameter
        for param in params:
            param_parts = param.split("=", 1)
            if len(param_parts) == 1:
                return None
            key, value = param_parts
            params_dict[key.strip()] = eval_expression(value.strip(), {})

        event_dict.update(params_dict)

    return event_dict


def run_chat(
    config_path: Optional[str] = None,
    verbose: bool = False,
    verbose_llm_calls: bool = False,
    streaming: bool = False,
    server_url: Optional[str] = None,
    config_id: Optional[str] = None,
):
    """Run a chat session in the terminal.

    Args:
        config_path (Optional[str]): The path to the configuration file. Defaults to None.
        verbose (bool): Whether to run in verbose mode. Defaults to False.
        verbose_llm_calls (bool): Whether to print the prompts and the completions. Defaults to False.
        streaming (bool): Whether to enable streaming mode. Defaults to False.
        server_url (Optional[str]): The URL of the chat server. Defaults to None.
        config_id (Optional[str]): The configuration ID. Defaults to None.
    """
    rails_config = RailsConfig.from_path(config_path)

    if verbose and verbose_llm_calls:
        console.print(
            "NOTE: use the `--verbose-no-llm` option to exclude the LLM prompts "
            "and completions from the log.\n"
        )

    console.print("Starting the chat (Press Ctrl + C twice to quit) ...")

    if rails_config.colang_version == "1.0":
        asyncio.run(
            _run_chat_v1_0(
                config_path=config_path,
                verbose=verbose,
                streaming=streaming,
                server_url=server_url,
                config_id=config_id,
            )
        )
    elif rails_config.colang_version == "2.x":
        rails_app = LLMRails(rails_config, verbose=verbose)
        asyncio.run(_run_chat_v2_x(rails_app))
    else:
        raise Exception(f"Invalid colang version: {rails_config.colang_version}")
