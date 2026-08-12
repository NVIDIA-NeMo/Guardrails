# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Conversion utilities between Anthropic Messages API and NeMo ChatMessage formats.

Used by both the server endpoint (messages.py) and the native Anthropic model
adapter (anthropic_chat.py).
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from nemoguardrails.types import ChatMessage, Role, ToolCall, ToolCallFunction


def anthropic_content_to_text(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract plain text from an Anthropic content value.

    Handles both string content and structured content-block arrays.
    Only ``type: "text"`` blocks are extracted; other block types
    (``tool_use``, ``tool_result``, ``thinking``, etc.) are ignored.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
    return "\n".join(parts) if parts else ""


def anthropic_tool_use_to_openai(content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic ``tool_use`` content blocks to OpenAI-style tool call dicts.

    Used by the ``/v1/messages/checks`` endpoint to normalize assistant tool
    calls into the OpenAI format that NeMo's internal pipeline expects.
    """
    tool_calls = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            args = block.get("input", {})
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    },
                }
            )
    return tool_calls


def anthropic_to_nemo_messages(
    system: Optional[Union[str, List[Dict[str, Any]]]] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> List[ChatMessage]:
    """Convert Anthropic Messages API input into NeMo ``ChatMessage`` objects.

    Handles all Anthropic content block types:

    - ``text`` blocks become message content.
    - ``tool_use`` blocks (in assistant messages) become ``ToolCall`` objects.
    - ``thinking`` / ``redacted_thinking`` blocks are preserved on the
      ``reasoning`` field for multi-turn extended thinking.
    - ``tool_result`` blocks (in user messages) become ``role=TOOL`` messages
      so that tool-result-only turns bypass input rails instead of being
      treated as empty user messages.
    """
    result: List[ChatMessage] = []

    if system:
        system_text = anthropic_content_to_text(system) if not isinstance(system, str) else system
        result.append(ChatMessage(role=Role.SYSTEM, content=system_text))

    for msg in messages or []:
        role_str = msg.get("role", "user")
        content = msg.get("content", "")

        if role_str == "user":
            if isinstance(content, list):
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if tool_results:
                    # Anthropic sends tool results inside user messages. We must
                    # emit them as role=TOOL so they don't create empty user
                    # messages that would trigger input rails and get blocked.
                    text = anthropic_content_to_text(content)
                    if text:
                        result.append(ChatMessage(role=Role.USER, content=text))
                    for tr in tool_results:
                        tr_content = tr.get("content", "")
                        tr_text = (
                            anthropic_content_to_text(tr_content)
                            if isinstance(tr_content, list)
                            else str(tr_content)
                            if tr_content
                            else ""
                        )
                        result.append(
                            ChatMessage(role=Role.TOOL, content=tr_text, tool_call_id=tr.get("tool_use_id", ""))
                        )
                else:
                    text = anthropic_content_to_text(content)
                    result.append(ChatMessage(role=Role.USER, content=text))
            else:
                text = anthropic_content_to_text(content)
                result.append(ChatMessage(role=Role.USER, content=text))

        elif role_str == "assistant":
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                thinking_blocks = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            args = block.get("input", {})
                            tool_calls.append(
                                ToolCall(
                                    id=block.get("id", ""),
                                    type="function",
                                    function=ToolCallFunction(
                                        name=block.get("name", ""),
                                        arguments=args if isinstance(args, dict) else {},
                                    ),
                                )
                            )
                        elif block.get("type") in ("thinking", "redacted_thinking"):
                            thinking_blocks.append(block)
                result.append(
                    ChatMessage(
                        role=Role.ASSISTANT,
                        content="\n".join(text_parts) if text_parts else "",
                        tool_calls=tool_calls if tool_calls else None,
                        reasoning=thinking_blocks if thinking_blocks else None,
                    )
                )
            else:
                result.append(ChatMessage(role=Role.ASSISTANT, content=str(content) if content else ""))

        elif role_str == "tool_result":
            tool_use_id = msg.get("tool_use_id", "")
            text = anthropic_content_to_text(content)
            result.append(ChatMessage(role=Role.TOOL, content=text, tool_call_id=tool_use_id))

    return result


def nemo_to_anthropic_messages(
    messages: List[ChatMessage],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert NeMo ``ChatMessage`` objects back to Anthropic Messages API format.

    Returns a ``(system, messages)`` tuple matching the Anthropic API shape.
    System messages are extracted into the top-level ``system`` parameter.
    ``TOOL`` messages are wrapped as ``tool_result`` blocks inside user
    messages, per the Anthropic API convention. Thinking/reasoning blocks
    are placed before text blocks, as required by the API.
    """
    system_text: Optional[str] = None
    anthropic_messages: List[Dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_text = msg.content if isinstance(msg.content, str) else anthropic_content_to_text(msg.content or "")
            continue

        if msg.role == Role.USER:
            content: Union[str, List[Dict[str, Any]]]
            if isinstance(msg.content, list):
                content = msg.content
            else:
                content = str(msg.content) if msg.content else ""
            anthropic_messages.append({"role": "user", "content": content})

        elif msg.role == Role.ASSISTANT:
            blocks: List[Dict[str, Any]] = []
            if msg.reasoning:
                if isinstance(msg.reasoning, list):
                    blocks.extend(msg.reasoning)
                elif isinstance(msg.reasoning, str):
                    blocks.append({"type": "thinking", "thinking": msg.reasoning})
            if msg.content:
                text = str(msg.content) if isinstance(msg.content, str) else anthropic_content_to_text(msg.content)
                if text:
                    blocks.append({"type": "text", "text": text})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tc.function.arguments,
                        }
                    )
            anthropic_messages.append(
                {"role": "assistant", "content": blocks if blocks else [{"type": "text", "text": ""}]}
            )

        elif msg.role == Role.TOOL:
            content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": str(msg.content) if msg.content else ""}]
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": content_blocks,
                        }
                    ],
                }
            )

    return system_text, anthropic_messages
