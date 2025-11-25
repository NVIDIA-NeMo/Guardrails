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

"""Utilities for converting between LangChain messages and dictionary format."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def get_message_role(msg: BaseMessage) -> str:
    """Get the role string for a BaseMessage."""
    if isinstance(msg, AIMessage):
        return "assistant"
    elif isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, SystemMessage):
        return "system"
    elif isinstance(msg, ToolMessage):
        return "tool"
    else:
        return getattr(msg, "type", "user")


def get_message_class(msg_type: str) -> Type[BaseMessage]:
    """Get the appropriate message class for a given type/role."""
    if msg_type == "user":
        return HumanMessage
    elif msg_type in ["bot", "assistant"]:
        return AIMessage
    elif msg_type in ["system", "developer"]:
        return SystemMessage
    elif msg_type == "tool":
        return ToolMessage
    else:
        raise ValueError(f"Unknown message type: {msg_type}")


def message_to_dict(msg: BaseMessage) -> Dict[str, Any]:
    """
    Convert a BaseMessage to dictionary format, preserving all model fields.

    Args:
        msg: The BaseMessage to convert

    Returns:
        Dictionary representation with role, content, and all other fields
    """
    result = {"role": get_message_role(msg), "content": msg.content}

    if isinstance(msg, ToolMessage):
        result["tool_call_id"] = msg.tool_call_id

    exclude_fields = {"type", "content", "example"}

    if hasattr(msg, "model_fields"):
        for field_name in msg.model_fields:
            if field_name not in exclude_fields and field_name not in result:
                value = getattr(msg, field_name, None)
                if value is not None:
                    result[field_name] = value

    return result


def dict_to_message(msg_dict: Dict[str, Any]) -> BaseMessage:
    """
    Convert a dictionary to the appropriate BaseMessage type.

    Args:
        msg_dict: Dictionary with role/type, content, and optional fields

    Returns:
        The appropriate BaseMessage instance
    """
    msg_type = msg_dict.get("type") or msg_dict.get("role")
    if not msg_type:
        raise ValueError("Message dictionary must have 'type' or 'role' field")

    content = msg_dict.get("content", "")
    message_class = get_message_class(msg_type)

    exclude_keys = {"role", "type", "content"}

    valid_fields = set(message_class.model_fields.keys()) if hasattr(message_class, "model_fields") else set()

    kwargs = {k: v for k, v in msg_dict.items() if k not in exclude_keys and k in valid_fields and v is not None}

    if message_class == ToolMessage:
        kwargs["tool_call_id"] = msg_dict.get("tool_call_id", "")

    return message_class(content=content, **kwargs)


def messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Convert a list of BaseMessage objects to dictionary format.

    Args:
        messages: List of BaseMessage objects

    Returns:
        List of dictionary representations
    """
    return [message_to_dict(msg) for msg in messages]


def dicts_to_messages(msg_dicts: List[Dict[str, Any]]) -> List[BaseMessage]:
    """
    Convert a list of dictionaries to BaseMessage objects.

    Args:
        msg_dicts: List of message dictionaries

    Returns:
        List of appropriate BaseMessage instances
    """
    return [dict_to_message(msg_dict) for msg_dict in msg_dicts]


def is_message_type(obj: Any, message_type: Type[BaseMessage]) -> bool:
    """Check if an object is an instance of a specific message type."""
    return isinstance(obj, message_type)


def is_base_message(obj: Any) -> bool:
    """Check if an object is any type of BaseMessage."""
    return isinstance(obj, BaseMessage)


def is_ai_message(obj: Any) -> bool:
    """Check if an object is an AIMessage."""
    return isinstance(obj, AIMessage)


def is_human_message(obj: Any) -> bool:
    """Check if an object is a HumanMessage."""
    return isinstance(obj, HumanMessage)


def is_system_message(obj: Any) -> bool:
    """Check if an object is a SystemMessage."""
    return isinstance(obj, SystemMessage)


def is_tool_message(obj: Any) -> bool:
    """Check if an object is a ToolMessage."""
    return isinstance(obj, ToolMessage)


def all_base_messages(items: List[Any]) -> bool:
    """Check if all items in a list are BaseMessage instances."""
    return all(isinstance(item, BaseMessage) for item in items)


def create_ai_message(
    content: str,
    tool_calls: Optional[list] = None,
    additional_kwargs: Optional[dict] = None,
    response_metadata: Optional[dict] = None,
    id: Optional[str] = None,
    name: Optional[str] = None,
    usage_metadata: Optional[dict] = None,
    **extra_kwargs,
) -> AIMessage:
    """Create an AIMessage with optional fields."""
    kwargs = {}
    if tool_calls is not None:
        kwargs["tool_calls"] = tool_calls
    if additional_kwargs is not None:
        kwargs["additional_kwargs"] = additional_kwargs
    if response_metadata is not None:
        kwargs["response_metadata"] = response_metadata
    if id is not None:
        kwargs["id"] = id
    if name is not None:
        kwargs["name"] = name
    if usage_metadata is not None:
        kwargs["usage_metadata"] = usage_metadata

    valid_fields = set(AIMessage.model_fields.keys()) if hasattr(AIMessage, "model_fields") else set()
    for key, value in extra_kwargs.items():
        if key in valid_fields and key not in kwargs:
            kwargs[key] = value

    return AIMessage(content=content, **kwargs)


def create_ai_message_chunk(content: str, **metadata) -> AIMessageChunk:
    """Create an AIMessageChunk with optional metadata."""
    return AIMessageChunk(content=content, **metadata)


def create_human_message(
    content: str,
    additional_kwargs: Optional[dict] = None,
    response_metadata: Optional[dict] = None,
    id: Optional[str] = None,
    name: Optional[str] = None,
) -> HumanMessage:
    """Create a HumanMessage with optional fields."""
    kwargs = {}
    if additional_kwargs is not None:
        kwargs["additional_kwargs"] = additional_kwargs
    if response_metadata is not None:
        kwargs["response_metadata"] = response_metadata
    if id is not None:
        kwargs["id"] = id
    if name is not None:
        kwargs["name"] = name

    return HumanMessage(content=content, **kwargs)


def create_system_message(
    content: str,
    additional_kwargs: Optional[dict] = None,
    response_metadata: Optional[dict] = None,
    id: Optional[str] = None,
    name: Optional[str] = None,
) -> SystemMessage:
    """Create a SystemMessage with optional fields."""
    kwargs = {}
    if additional_kwargs is not None:
        kwargs["additional_kwargs"] = additional_kwargs
    if response_metadata is not None:
        kwargs["response_metadata"] = response_metadata
    if id is not None:
        kwargs["id"] = id
    if name is not None:
        kwargs["name"] = name

    return SystemMessage(content=content, **kwargs)


def create_tool_message(
    content: str,
    tool_call_id: str,
    name: Optional[str] = None,
    additional_kwargs: Optional[dict] = None,
    response_metadata: Optional[dict] = None,
    id: Optional[str] = None,
    artifact: Optional[Any] = None,
    status: Optional[str] = None,
) -> ToolMessage:
    """Create a ToolMessage with optional fields."""
    kwargs = {"tool_call_id": tool_call_id}
    if name is not None:
        kwargs["name"] = name
    if additional_kwargs is not None:
        kwargs["additional_kwargs"] = additional_kwargs
    if response_metadata is not None:
        kwargs["response_metadata"] = response_metadata
    if id is not None:
        kwargs["id"] = id
    if artifact is not None:
        kwargs["artifact"] = artifact
    if status is not None:
        kwargs["status"] = status

    return ToolMessage(content=content, **kwargs)
