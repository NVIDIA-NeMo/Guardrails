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

"""OpenAI API schema definitions for the NeMo Guardrails server."""

import os
from typing import Annotated, Any, List, Literal, Optional, Union

from openai.types.chat.chat_completion import ChatCompletion
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from nemoguardrails.rails.llm.options import GenerationOptions, RailType


class GuardrailsDataOutput(BaseModel):
    """Guardrails-specific output data."""

    config_id: Optional[str] = Field(
        default=None,
        description="The guardrails configuration ID associated with this response.",
    )
    llm_output: Optional[dict] = Field(default=None, description="Additional LLM output data.")
    output_data: Optional[dict] = Field(default=None, description="Additional output data.")
    log: Optional[dict] = Field(default=None, description="Generation log data.")


class GuardrailsChatCompletion(ChatCompletion):
    """OpenAI API response body with NeMo-Guardrails extensions."""

    guardrails: Optional[GuardrailsDataOutput] = Field(default=None, description="Guardrails specific output data.")


class _OpenAIChatMessageBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _OpenAIChatMessageRoleSchema(BaseModel):
    role: Literal["developer", "system", "user", "assistant", "tool", "function", "context"]


class _OpenAIPromptCacheBreakpointSchema(_OpenAIChatMessageBase):
    mode: Literal["explicit"]


class _OpenAIImageURLSchema(_OpenAIChatMessageBase):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None


class _OpenAITextContentPartSchema(_OpenAIChatMessageBase):
    text: str
    type: Literal["text"]
    prompt_cache_breakpoint: Optional[_OpenAIPromptCacheBreakpointSchema] = None


class _OpenAIImageContentPartSchema(_OpenAIChatMessageBase):
    image_url: _OpenAIImageURLSchema
    type: Literal["image_url"]
    prompt_cache_breakpoint: Optional[_OpenAIPromptCacheBreakpointSchema] = None


class _OpenAIFileSchema(_OpenAIChatMessageBase):
    file_data: Optional[str] = None
    file_id: Optional[str] = None
    filename: Optional[str] = None


class _OpenAIFileContentPartSchema(_OpenAIChatMessageBase):
    file: _OpenAIFileSchema
    type: Literal["file"]
    prompt_cache_breakpoint: Optional[_OpenAIPromptCacheBreakpointSchema] = None


class _OpenAIRefusalContentPartSchema(_OpenAIChatMessageBase):
    refusal: str
    type: Literal["refusal"]


_OpenAITextContentSchema = Union[str, List[_OpenAITextContentPartSchema]]
_OpenAIUserContentPartSchema = Union[
    _OpenAITextContentPartSchema,
    _OpenAIImageContentPartSchema,
    _OpenAIFileContentPartSchema,
]


class _OpenAIDeveloperMessageSchema(_OpenAIChatMessageBase):
    content: _OpenAITextContentSchema
    role: Literal["developer"]
    name: Optional[str] = None


class _OpenAISystemMessageSchema(_OpenAIChatMessageBase):
    content: _OpenAITextContentSchema
    role: Literal["system"]
    name: Optional[str] = None


class _OpenAIUserMessageSchema(_OpenAIChatMessageBase):
    content: Union[str, List[_OpenAIUserContentPartSchema]]
    role: Literal["user"]
    name: Optional[str] = None


class _OpenAIFunctionCallSchema(_OpenAIChatMessageBase):
    arguments: str
    name: str


class _OpenAIFunctionToolCallSchema(_OpenAIChatMessageBase):
    id: str
    function: _OpenAIFunctionCallSchema
    type: Literal["function"]


class _OpenAIAssistantMessageSchema(_OpenAIChatMessageBase):
    role: Literal["assistant"]
    content: Optional[Union[str, List[Union[_OpenAITextContentPartSchema, _OpenAIRefusalContentPartSchema]]]] = None
    function_call: Optional[_OpenAIFunctionCallSchema] = None
    name: Optional[str] = None
    tool_calls: Optional[List[_OpenAIFunctionToolCallSchema]] = None
    refusal: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_content_or_call(cls, value: Any) -> Any:
        if (
            isinstance(value, dict)
            and value.get("content") is None
            and value.get("function_call") is None
            and not value.get("tool_calls")
            and value.get("refusal") is None
        ):
            raise ValidationError.from_exception_data(
                cls.__name__,
                [{"type": "missing", "loc": ("content",), "input": value}],
            )
        return value


class _OpenAIToolMessageSchema(_OpenAIChatMessageBase):
    content: _OpenAITextContentSchema
    role: Literal["tool"]
    tool_call_id: str


class _OpenAIFunctionMessageSchema(_OpenAIChatMessageBase):
    content: Optional[str]
    name: str
    role: Literal["function"]


class _GuardrailsContextMessageSchema(_OpenAIChatMessageBase):
    content: dict[str, Any]
    role: Literal["context"]


_OpenAIChatMessageInput = Union[
    _OpenAIDeveloperMessageSchema,
    _OpenAISystemMessageSchema,
    _OpenAIUserMessageSchema,
    _OpenAIAssistantMessageSchema,
    _OpenAIToolMessageSchema,
    _OpenAIFunctionMessageSchema,
    _GuardrailsContextMessageSchema,
]

_OPENAI_CHAT_MESSAGE_SCHEMAS: dict[str, type[BaseModel]] = {
    "developer": _OpenAIDeveloperMessageSchema,
    "system": _OpenAISystemMessageSchema,
    "user": _OpenAIUserMessageSchema,
    "assistant": _OpenAIAssistantMessageSchema,
    "tool": _OpenAIToolMessageSchema,
    "function": _OpenAIFunctionMessageSchema,
    "context": _GuardrailsContextMessageSchema,
}


def _validate_openai_chat_message(message: Any) -> Any:
    role = _OpenAIChatMessageRoleSchema.model_validate(message).role
    _OPENAI_CHAT_MESSAGE_SCHEMAS[role].model_validate(message)
    return message


OpenAIChatMessage = Annotated[
    dict[str, Any],
    BeforeValidator(
        _validate_openai_chat_message,
        json_schema_input_type=_OpenAIChatMessageInput,
    ),
]


class OpenAIChatCompletionRequest(BaseModel):
    """Standard OpenAI chat completion request parameters."""

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_custom_tools(cls, data: Any) -> Any:
        if isinstance(data, dict):
            tools = data.get("tools")
            if isinstance(tools, list) and any(
                isinstance(tool, dict) and tool.get("type") == "custom" for tool in tools
            ):
                raise ValueError("Custom tools are not supported.")

            tool_choice = data.get("tool_choice")
            if isinstance(tool_choice, dict):
                allowed_tools = tool_choice.get("tools")
                if tool_choice.get("type") == "custom" or (
                    isinstance(allowed_tools, list)
                    and any(isinstance(tool, dict) and tool.get("type") == "custom" for tool in allowed_tools)
                ):
                    raise ValueError("Custom tools are not supported.")
        return data

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_audio(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "modalities" in data and not isinstance(data["modalities"], list):
                raise ValueError("The 'modalities' parameter must be a list.")
            modalities = data.get("modalities", [])
            if "audio" in data or "audio" in modalities:
                raise ValueError("Audio input and output are not supported.")
        return data

    messages: Optional[List[OpenAIChatMessage]] = Field(
        default=None,
        description="The list of messages in the current conversation.",
    )
    model: str = Field(
        ...,
        description="The LLM model to use for chat completion (e.g., 'gpt-4o', 'llama-3.1-8b').",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent as server-sent events.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature to use.",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling parameter.",
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Stop sequences.",
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description="Presence penalty parameter.",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description="Frequency penalty parameter.",
    )
    logit_bias: Optional[dict] = Field(
        default=None,
        description="Logit bias parameter.",
    )
    logprobs: Optional[bool] = Field(
        default=None,
        description="Log probabilities parameter.",
    )
    tools: Optional[list[dict]] = Field(
        default=None,
        description="Tools parameter.",
    )
    tool_choice: Optional[str | dict] = Field(
        default=None,
        description="Tool choice parameter.",
    )
    parallel_tool_calls: Optional[bool] = Field(
        default=None,
        description="Whether to allow parallel tool calls during tool use.",
    )


class GuardrailsDataInput(BaseModel):
    """Guardrails-specific options for the request."""

    config_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("DEFAULT_CONFIG_ID", None),
        description="The guardrails configuration ID to use.",
    )
    config_ids: Optional[List[str]] = Field(
        default=None,
        description="List of configuration IDs to combine.",
        validate_default=True,
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=255,
        description="The ID of an existing thread to continue.",
    )
    context: Optional[dict] = Field(
        default=None,
        description="Additional context data for the conversation.",
    )
    options: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="Additional generation options.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_caller_supplied_state(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("state") is not None:
            raise ValueError("Caller-supplied state is not accepted over HTTP.")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_config_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("config_id") is not None and data.get("config_ids") is not None:
                raise ValueError("Only one of config_id or config_ids should be specified")
        return data

    @field_validator("config_ids", mode="before")
    @classmethod
    def ensure_config_ids(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None and info.data.get("config_id"):
            return [info.data["config_id"]]
        return v


class GuardrailsChatCompletionRequest(OpenAIChatCompletionRequest):
    """OpenAI chat completion request with NeMo Guardrails extensions."""

    guardrails: GuardrailsDataInput = Field(
        default_factory=GuardrailsDataInput,
        description="Guardrails specific options for the request.",
    )


class OpenAIModel(BaseModel):
    """Standard OpenAI model."""

    id: str = Field(..., description="The model identifier.")
    created: int = Field(..., description="The unix timestamp in seconds of the model's creation.")
    object: Literal["model"] = Field("model", description="The object type which is always 'model'.")
    owned_by: str | None = Field(..., description="The organization that owns the model.")


class OpenAIModelsList(BaseModel):
    """Standard OpenAI models list response."""

    data: list[OpenAIModel] = Field(..., description="List of OpenAI model objects.")


class GuardrailCheckDataInput(GuardrailsDataInput):
    """Guardrails input options specific to the /v1/checks endpoint."""

    rail_types: Optional[List[RailType]] = Field(
        default=None,
        description="Rail types to run. When omitted, auto-detected from message roles.",
    )


class GuardrailCheckRequest(OpenAIChatCompletionRequest):
    """Request body for the /v1/checks endpoint."""

    guardrails: GuardrailCheckDataInput = Field(
        default_factory=GuardrailCheckDataInput,
        description="Guardrails specific options for the request.",
    )


class GuardrailCheckResponse(BaseModel):
    """Response from the /v1/checks endpoint."""

    status: str = Field(..., description="Overall check result: passed, modified, or blocked.")
    content: str = Field(..., description="Content after rails processing.")
    rail: Optional[str] = Field(default=None, description="Name of the blocking rail, if any.")
