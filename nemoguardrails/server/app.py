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

"""Chat UI for Guardrails Server."""

import logging
import os
from typing import List, Optional

import chainlit as cl

from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.server.api import _get_rails, _has_config_file, app, challenges

log = logging.getLogger(__name__)


@cl.set_starters
async def set_starters(user: Optional[cl.User] = None) -> List[cl.Starter]:
    """Show challenges as starter prompts when available."""
    if not challenges:
        return []

    return [
        cl.Starter(
            label=challenge.get("name", challenge.get("content", "")[:40]),
            message=challenge.get("content", ""),
            icon=challenge.get("icon", None),
        )
        for challenge in challenges
    ]


def _discover_configs() -> List[str]:
    """Return the list of available guardrails configuration IDs."""
    if app.single_config_mode and app.single_config_id:
        return [app.single_config_id]

    if not hasattr(app, "rails_config_path") or not os.path.isdir(app.rails_config_path):
        return []

    return sorted(
        f
        for f in os.listdir(app.rails_config_path)
        if os.path.isdir(os.path.join(app.rails_config_path, f))
        and not f.startswith(".")
        and not f.startswith("_")
        and _has_config_file(os.path.join(app.rails_config_path, f))
    )


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session with a config selector dropdown."""
    cl.user_session.set("messages", [])

    configs = _discover_configs()

    if not configs:
        cl.user_session.set("config_id", None)
        await cl.Message(
            content="No guardrails configurations available. Please configure the server with a rails config path."
        ).send()
        return

    default = app.default_config_id or app.single_config_id
    initial = default if default in configs else configs[0]

    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="config_id",
                label="Guardrails Configuration",
                values=configs,
                initial_value=initial,
            )
        ]
    ).send()

    selected = settings["config_id"]
    cl.user_session.set("config_id", selected)


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle configuration changes from the settings panel."""
    cl.user_session.set("config_id", settings["config_id"])
    cl.user_session.set("messages", [])
    await cl.Message(content=f"Switched to configuration: **{settings['config_id']}**").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process an incoming user message through guardrails."""
    config_id = cl.user_session.get("config_id")

    if not config_id:
        await cl.Message(content="No guardrails configuration selected. Please select one from settings.").send()
        return

    messages = cl.user_session.get("messages") or []
    messages.append({"role": "user", "content": message.content})

    try:
        llm_rails = await _get_rails([config_id])
    except Exception:
        log.exception("Failed to load rails config '%s'", config_id)
        messages.pop()
        cl.user_session.set("messages", messages)
        await cl.Message(content=f"Error loading guardrails configuration '{config_id}'. Check server logs.").send()
        return

    response_msg = cl.Message(content="")
    await response_msg.send()

    full_response = ""
    streaming_unsupported = False
    try:
        try:
            async for chunk in llm_rails.stream_async(messages=messages):
                if isinstance(chunk, str) and chunk:
                    full_response += chunk
                    await response_msg.stream_token(chunk)
        except StreamingNotSupportedError:
            streaming_unsupported = True

        if streaming_unsupported:
            result = await llm_rails.generate_async(messages=messages)
            full_response = result.get("content", str(result)) if isinstance(result, dict) else str(result)
            response_msg.content = full_response
            await response_msg.update()

        messages.append({"role": "assistant", "content": full_response})
        cl.user_session.set("messages", messages)

    except Exception:
        log.exception("Error generating response for config '%s'", config_id)
        messages.pop()
        cl.user_session.set("messages", messages)
        response_msg.content = f"An error occurred for configuration '{config_id}'. Check server logs."
        await response_msg.update()
