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
import json
import logging
from ast import literal_eval
from datetime import datetime

from rich.logging import RichHandler
from rich.text import Text

from nemoguardrails.logging.simplify_formatter import SimplifyFormatter
from nemoguardrails.utils import console

# Global state variable to track if the verbose mode has already been enabled
verbose_mode_enabled = False

# Whether to log the LLM calls verbosely.
verbose_llm_calls = False

# Whether the debug mode is enabled or not.
debug_mode_enabled = False


class VerboseHandler(logging.StreamHandler):
    """A log handler for verbose mode."""

    def __init__(self, *args, **kwargs):
        super(VerboseHandler, self).__init__(*args, **kwargs)

    def emit(self, record) -> None:
        msg = self.format(record)

        # We check if we're using the spacial syntax with " :: " which denotes a title.
        if " :: " in msg:
            title, body = msg.split(" :: ", 1)
            title = title.strip()

            skip_print = False

            # We remove the title for completion messages and stop the blinking cursor.
            if title == "Completion":
                skip_print = True
                if verbose_llm_calls:
                    console.print("")
                    id_str = getattr(record, "id", None)
                    id_display = f"({id_str[:5]}..)" if id_str else ""
                    console.print(f"[cyan]LLM {title} {id_display}[/]")
                    for line in body.split("\n"):
                        text = Text(line, style="black on #006600", end="\n")
                        text.pad_right(console.width)
                        console.print(text)
                    console.print("")

            # For prompts, we also start the blinking cursor.
            elif title in ["Prompt", "Prompt Messages"]:
                if verbose_llm_calls:
                    skip_print = True
                    console.print("")
                    id_str = getattr(record, "id", None)
                    id_display = f"({id_str[:5]}..)" if id_str else ""
                    task_str = getattr(record, "task", "unknown")
                    console.print(f"[cyan]LLM Prompt {id_display} - {task_str}[/]")

                    for line in body.split("\n"):
                        if line.strip() == "[/]":
                            continue

                        if line.startswith("[cyan]") and line.endswith("[/]"):
                            text = Text(line[6:-3], style="maroon", end="\n")
                        else:
                            text = Text(line, style="black on #909090", end="\n")

                        text.pad_right(console.width)
                        console.print(text)
                    console.print("")

            elif title.startswith("Colang Log ("):
                title = f"[green]{title[11:]}[/]"

            elif title == "Event":
                # For events, we also color differently the type of event.
                event_name, body = body.split(" ", 1)
                title = f"[blue]{title}[/] [bold]{event_name}[/]"

            else:
                if title == "Processing event" and body.startswith("{"):
                    try:
                        event_dict = literal_eval(body)
                        event_type = event_dict["type"]

                        if "ActionStarted" in event_type:
                            skip_print = True
                        elif event_dict["type"] not in [
                            "CheckLocalAsync",
                            "LocalAsyncCounter",
                        ]:
                            del event_dict["type"]
                            del event_dict["uid"]
                            body = json.dumps(event_dict)

                            # We're adding a new line before action events, to
                            # make it more readable.
                            if event_type.startswith("Start") and event_type.endswith(
                                "Action"
                            ):
                                title = f"[magenta][bold]Start[/]{event_type[5:]}[/]"
                            elif event_type.startswith("Stop") and event_type.endswith(
                                "Action"
                            ):
                                title = f"[magenta][bold]Stop[/]{event_type[4:]}[/]"
                            elif event_type.endswith("ActionUpdated"):
                                title = f"[magenta]{event_type[:-7]}[bold]Updated[/][/]"
                            elif event_type.endswith("ActionFinished"):
                                if event_type == "UtteranceUserActionFinished":
                                    title = f"[magenta]{event_type[:-8]}[bold]Finished[/][/]"
                                else:
                                    title = f"[magenta]{event_type[:-8]}[bold]Finished[/][/]"
                            elif event_type.endswith("ActionFailed"):
                                title = f"[magenta]{event_type[:-6]}[bold]Failed[/][/]"
                            elif event_type == "BotThinking":
                                title = f"[yellow bold]{event_type}[/]"
                            else:
                                title = event_type
                        else:
                            skip_print = True
                    except Exception:
                        title = f"[red bold]{title}[/]"
                elif title == "Running action":
                    skip_print = True
                elif title == "Matching head":
                    skip_print = True
                    # TODO: activate this once the source line is sorted
                    # flow_name = re.findall(r"flow='(.*?)'", body)[0]
                    # flow_pos = re.findall(r"pos=(\d+)", body)[0]
                    # msg = f"[yellow][bold]{flow_name}[/]:{flow_pos}[/]"
                    #
                    # ignored_flows = ["_bot_say", "await_flow_by_name"]
                    # if flow_pos == "0" or flow_name in ignored_flows:
                    #     skip_print = True
                else:
                    if title == "---":
                        title = f"[#555555]{body}[/]"
                        body = ""
                    else:
                        title = f"[#707070]{title}[/] [#555555]{body}[/]"
                        body = ""

            if not skip_print:
                current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                msg = f"[dim]{current_time}[/] | "

                if body:
                    msg += f"[dim]{title}[/] | "
                    msg += f"[dim]{body}[/]"
                else:
                    msg += f"[dim]{title}[/]"

                console.print(msg, highlight=False, no_wrap=False)


def set_verbose(
    verbose: bool,
    llm_calls: bool = False,
    debug: bool = False,
    debug_level: str = "INFO",
    simplify: bool = False,
):
    """Configure the verbose mode.

    The verbose mode is meant to be user-friendly. It provides additional information
    about what is happening under the hood.

    The verbose debug mode provides detailed logs, and it's meant for debugging purposes.

    Args:
        verbose: Whether the verbose mode should be enabled or not.
        llm_calls: Whether to log the prompt and response from the LLM calls (default False).
        debug: Whether the debug mode should be enabled or not (default False).
        debug_level: The log level to be used for debug mode (default INFO).
        simplify: Whether the output should be simplified to optimize for readability.
    """
    global verbose_mode_enabled
    global debug_mode_enabled
    global verbose_llm_calls

    if verbose and not verbose_mode_enabled:
        root_logger = logging.getLogger()

        # We make sure that the root logger is at least INFO so that we can see the messages from the VerboseHandler.
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

        # We make sure the log level for the default root console handler is set to WARNING.
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.WARNING)

        # In debug mode we add the RichHandler, otherwise we add the VerboseHandler.
        if debug:
            root_logger = logging.getLogger()
            rich_handler = RichHandler(markup=True, rich_tracebacks=True)

            # If needed, simplify further the verbose output
            if simplify:
                rich_handler.setFormatter(SimplifyFormatter())

            root_logger.addHandler(rich_handler)
            root_logger.setLevel(debug_level)
        else:
            # Next, we also add an instance of the VerboseHandler.
            verbose_handler = VerboseHandler()
            verbose_handler.setLevel(logging.INFO)
            verbose_handler.setFormatter(SimplifyFormatter())
            root_logger.addHandler(verbose_handler)

        # Also, we make sure the sentence_transformers log level is set to WARNING.
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

        verbose_mode_enabled = True
        verbose_llm_calls = llm_calls
        debug_mode_enabled = debug
        console.print("Entered verbose mode.")
