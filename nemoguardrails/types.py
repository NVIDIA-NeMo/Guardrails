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

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCallFunction:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolCall:
    id: str
    type: str = "function"
    function: ToolCallFunction = field(default_factory=lambda: ToolCallFunction(name="", arguments={}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


@dataclass
class UsageInfo:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error", "other"]


_STANDARD_MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name", "provider_metadata"}

_ROLE_ALIASES = {
    "bot": Role.ASSISTANT,
    "assistant": Role.ASSISTANT,
    "human": Role.USER,
    "user": Role.USER,
    "developer": Role.SYSTEM,
    "system": Role.SYSTEM,
    "tool": Role.TOOL,
}


@dataclass
class ChatMessage:
    role: Role
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    provider_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_user(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role=Role.USER, content=content, **kwargs)

    @classmethod
    def from_assistant(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role=Role.ASSISTANT, content=content, **kwargs)

    @classmethod
    def from_system(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role=Role.SYSTEM, content=content, **kwargs)

    @classmethod
    def from_tool(cls, content: str, tool_call_id: str, **kwargs) -> "ChatMessage":
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role.value}

        if self.content is not None:
            payload["content"] = self.content

        if self.tool_calls is not None:
            payload["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id

        if self.name is not None:
            payload["name"] = self.name

        if self.provider_metadata:
            payload["provider_metadata"] = self.provider_metadata

        return payload

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChatMessage":
        """Create a ChatMessage from a dict.

        Accepts both the canonical nested tool call format
        (``{"function": {"name": ..., "arguments": ...}}``) and the legacy
        flat format (``{"name": ..., "args": ...}``). JSON string arguments
        are parsed automatically. Role aliases like "bot", "human", and
        "developer" are mapped to canonical Role values. Unknown keys are
        captured into ``provider_metadata``.
        """

        raw_role = d.get("role")
        if raw_role is None:
            raise ValueError("Missing required key: 'role'")
        role = _ROLE_ALIASES.get(raw_role)
        if role is None:
            raise ValueError(f"Unknown role: {raw_role}")

        tool_calls = None
        raw_tool_calls = d.get("tool_calls")
        if raw_tool_calls is not None:
            tool_calls = []
            for tc in raw_tool_calls:
                func_data = tc.get("function")
                if func_data is not None:
                    raw_args = func_data.get("arguments", {})
                else:
                    raw_args = tc.get("args", {})
                    func_data = {"name": tc.get("name", "")}

                if isinstance(raw_args, str):
                    try:
                        args_dict = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raise ValueError(f"Tool call arguments are not valid JSON: {raw_args!r}")
                    if not isinstance(args_dict, dict):
                        raise ValueError(
                            f"Tool call arguments must be a JSON object, got {type(args_dict).__name__}: {raw_args!r}"
                        )
                else:
                    if not isinstance(raw_args, dict):
                        raise ValueError(
                            f"Tool call arguments must be a dict, got {type(raw_args).__name__}: {raw_args!r}"
                        )
                    args_dict = raw_args

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=ToolCallFunction(
                            name=func_data.get("name", ""),
                            arguments=args_dict,
                        ),
                    )
                )

        extra = {k: v for k, v in d.items() if k not in _STANDARD_MESSAGE_KEYS}
        provider_metadata = {**extra, **d.get("provider_metadata", {})}

        return cls(
            role=role,
            content=d.get("content"),
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            provider_metadata=provider_metadata,
        )


@dataclass
class LLMResponse:
    content: str
    reasoning: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    model: Optional[str] = None
    finish_reason: Optional[FinishReason] = None
    stop_sequence: Optional[str] = None
    request_id: Optional[str] = None
    usage: Optional[UsageInfo] = None
    provider_metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponseChunk:
    delta_content: Optional[str] = None
    delta_reasoning: Optional[str] = None
    delta_tool_calls: Optional[List[ToolCall]] = None
    model: Optional[str] = None
    finish_reason: Optional[FinishReason] = None
    request_id: Optional[str] = None
    usage: Optional[UsageInfo] = None
    provider_metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class LLMModel(Protocol):
    """Protocol that all LLM backends must implement.

    Adapters wrap provider-specific SDKs (LangChain, LiteLLM, OpenAI, etc.)
    behind this interface so the core pipeline remains framework-agnostic.

    ``prompt`` accepts either a plain string or a list of ``ChatMessage``
    objects. Adapters convert ``ChatMessage`` to whatever their SDK expects.
    ``**kwargs`` are forwarded to the underlying SDK (e.g. temperature,
    max_tokens).
    """

    async def generate(
        self,
        prompt: Union[str, List["ChatMessage"]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> "LLMResponse": ...

    def stream(
        self,
        prompt: Union[str, List["ChatMessage"]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator["LLMResponseChunk"]:
        """Implementations must be async generator functions (use ``yield``)."""
        ...

    @property
    def model_name(self) -> str: ...

    @property
    def provider_name(self) -> Optional[str]: ...

    @property
    def provider_url(self) -> Optional[str]: ...


@runtime_checkable
class LLMFramework(Protocol):
    """Protocol for pluggable LLM framework backends.

    Each framework (LangChain, LiteLLM, etc.) implements this protocol to
    provide a factory for creating ``LLMModel`` instances.

    ``model_kwargs`` carries all provider-specific configuration. Framework
    implementations extract what they need (e.g. LangChain pops ``mode``
    to choose between chat and text completion models).
    """

    def create_model(
        self,
        model_name: str,
        provider_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMModel: ...
