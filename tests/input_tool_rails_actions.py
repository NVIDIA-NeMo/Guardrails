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

"""Utility functions for input tool rails tests.

This module contains the utility functions that were previously implemented as
fake test functions in the test file. They provide the actual implementation
for tool input validation, safety checking, and sanitization.
"""

import logging
import re
from typing import Optional

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


@action(is_system_action=True)
async def self_check_tool_input(
    tool_message: str = None,
    tool_name: str = None,
    tool_call_id: str = None,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Test implementation of basic tool input validation.

    This action performs basic validation of tool results coming from tools:
    - Checks if tool results are valid and safe
    - Validates the structure and content
    - Performs basic security checks on tool responses

    Args:
        tool_message: The content returned by the tool
        tool_name: Name of the tool that returned this result
        tool_call_id: ID linking to the original tool call
        context: Optional context information

    Returns:
        bool: True if tool input is valid, False to block
    """
    tool_message = tool_message or (context.get("tool_message") if context else "")
    tool_name = tool_name or (context.get("tool_name") if context else "")
    tool_call_id = tool_call_id or (context.get("tool_call_id") if context else "")

    config = context.get("config") if context else None
    allowed_tools = getattr(config, "allowed_tools", None) if config else None

    log.debug(f"Validating tool input from {tool_name}: {tool_message[:100]}...")

    if allowed_tools and tool_name not in allowed_tools:
        log.warning(f"Tool '{tool_name}' not in allowed tools list: {allowed_tools}")
        return False

    if not tool_message:
        log.warning(f"Empty tool message from {tool_name}")
        return False

    if not tool_name:
        log.warning("Tool message received without tool name")
        return False

    if not tool_call_id:
        log.warning(f"Tool message from {tool_name} missing tool_call_id")
        return False

    max_length = getattr(config, "max_tool_message_length", 10000) if config else 10000
    if len(tool_message) > max_length:
        log.warning(f"Tool message from {tool_name} exceeds max length: {len(tool_message)} > {max_length}")
        return False

    return True


@action(is_system_action=True)
async def validate_tool_input_safety(
    tool_message: str = None,
    tool_name: str = None,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Test implementation of tool input safety validation.

    This action checks tool results for potentially dangerous content:
    - Detects sensitive information patterns
    - Flags potentially harmful content
    - Prevents dangerous data from being processed

    Args:
        tool_message: The content returned by the tool
        tool_name: Name of the tool that returned this result
        context: Optional context information

    Returns:
        bool: True if tool input is safe, False to block
    """
    tool_message = tool_message or (context.get("tool_message") if context else "")
    tool_name = tool_name or (context.get("tool_name") if context else "")

    if not tool_message:
        return True

    log.debug(f"Validating safety of tool input from {tool_name}")

    dangerous_patterns = [
        "password",
        "secret",
        "api_key",
        "private_key",
        "token",
        "credential",
        "<script",
        "javascript:",
        "data:text/html",
        "eval(",
        "exec(",
        "__import__",
        "subprocess",
        "DROP TABLE",
        "DELETE FROM",
        "UPDATE.*SET",
    ]

    tool_message_lower = tool_message.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in tool_message_lower:
            log.warning(f"Potentially dangerous content in tool response from {tool_name}: pattern '{pattern}' found")
            return False

    return True


@action(is_system_action=True)
async def sanitize_tool_input(
    tool_message: str = None,
    tool_name: str = None,
    context: Optional[dict] = None,
    **kwargs,
) -> str:
    """Test implementation of tool input sanitization.

    This action cleans and sanitizes tool results:
    - Removes or masks sensitive information
    - Truncates overly long responses
    - Escapes potentially dangerous content

    Args:
        tool_message: The content returned by the tool
        tool_name: Name of the tool that returned this result
        context: Optional context information

    Returns:
        str: Sanitized tool message content
    """
    tool_message = tool_message or (context.get("tool_message") if context else "")
    tool_name = tool_name or (context.get("tool_name") if context else "")

    if not tool_message:
        return tool_message

    log.debug(f"Sanitizing tool input from {tool_name}")

    sanitized = tool_message

    sanitized = re.sub(
        r'(api[_-]?key|token|secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{16,})["\']?',
        r"\1: [REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )

    sanitized = re.sub(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", r"[USER]@\2", sanitized)

    config = context.get("config") if context else None
    max_length = getattr(config, "max_tool_message_length", 10000) if config else 10000

    if len(sanitized) > max_length:
        log.info(f"Truncating tool response from {tool_name}: {len(sanitized)} -> {max_length}")
        sanitized = sanitized[: max_length - 50] + "... [TRUNCATED]"

    return sanitized
