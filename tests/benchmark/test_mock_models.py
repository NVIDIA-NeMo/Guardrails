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

import pytest
from pydantic import ValidationError

from nemoguardrails.benchmark.mock_llm_server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Message,
    Model,
    ModelsResponse,
    Usage,
)


class TestMessage:
    """Test the Message model."""

    def test_message_creation(self):
        """Test creating a Message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_missing_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            Message(role="user")  # Missing content

        with pytest.raises(ValidationError):
            Message(content="Hello")  # Missing role


class TestChatCompletionRequest:
    """Test the ChatCompletionRequest model."""

    def test_chat_completion_request_minimal(self):
        """Test creating ChatCompletionRequest with minimal fields."""
        req = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.model == "gpt-3.5-turbo"
        assert len(req.messages) == 1
        assert req.temperature == 1.0  # Default
        assert req.n == 1  # Default
        assert req.stream is False  # Default

    def test_chat_completion_request_validation(self):
        """Test validation of ChatCompletionRequest fields."""
        # Test temperature bounds
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[Message(role="user", content="Hi")],
                temperature=3.0,  # > 2.0
            )

        # Test n bounds
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[Message(role="user", content="Hi")],
                n=200,  # > 128
            )

    def test_chat_completion_request_stop_variants(self):
        """Test stop parameter can be string or list."""
        req1 = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role="user", content="Hi")],
            stop="END",
        )
        assert req1.stop == "END"

        req2 = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role="user", content="Hi")],
            stop=["END", "STOP"],
        )
        assert req2.stop == ["END", "STOP"]


class TestCompletionRequest:
    """Test the CompletionRequest model."""

    def test_completion_request_minimal(self):
        """Test creating CompletionRequest with minimal fields."""
        req = CompletionRequest(
            model="text-davinci-003",
            prompt="Hello",
        )
        assert req.model == "text-davinci-003"
        assert req.prompt == "Hello"
        assert req.max_tokens == 16  # Default
        assert req.temperature == 1.0  # Default

    def test_completion_request_prompt_string(self):
        """Test CompletionRequest with string prompt."""
        req = CompletionRequest(model="test-model", prompt="Test prompt")
        assert req.prompt == "Test prompt"
        assert isinstance(req.prompt, str)

    def test_completion_request_prompt_list(self):
        """Test CompletionRequest with list of prompts."""
        req = CompletionRequest(model="test-model", prompt=["Prompt 1", "Prompt 2"])
        assert req.prompt == ["Prompt 1", "Prompt 2"]
        assert isinstance(req.prompt, list)

    def test_completion_request_all_fields(self):
        """Test creating CompletionRequest with all fields."""
        req = CompletionRequest(
            model="text-davinci-003",
            prompt=["Prompt 1", "Prompt 2"],
            max_tokens=50,
            temperature=0.8,
            top_p=0.95,
            n=3,
            stream=True,
            logprobs=5,
            echo=True,
            stop=["STOP"],
            presence_penalty=0.6,
            frequency_penalty=0.4,
            best_of=2,
            logit_bias={"token1": 1.0},
            user="user456",
        )
        assert req.model == "text-davinci-003"
        assert req.prompt == ["Prompt 1", "Prompt 2"]
        assert req.max_tokens == 50
        assert req.logprobs == 5
        assert req.echo is True
        assert req.best_of == 2

    def test_completion_request_validation(self):
        """Test validation of CompletionRequest fields."""
        # Test logprobs bounds
        with pytest.raises(ValidationError):
            CompletionRequest(
                model="test-model",
                prompt="Hi",
                logprobs=10,  # > 5
            )


class TestUsage:
    """Test the Usage model."""

    def test_usage_creation(self):
        """Test creating a Usage model."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_usage_missing_fields(self):
        """Test that missing fields raise validation error."""
        with pytest.raises(ValidationError):
            Usage(prompt_tokens=10, completion_tokens=20)  # Missing total_tokens


class TestChatCompletionChoice:
    """Test the ChatCompletionChoice model."""

    def test_chat_completion_choice_creation(self):
        """Test creating a ChatCompletionChoice."""
        choice = ChatCompletionChoice(
            index=0,
            message=Message(role="assistant", content="Response"),
            finish_reason="stop",
        )
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content == "Response"
        assert choice.finish_reason == "stop"


class TestCompletionChoice:
    """Test the CompletionChoice model."""

    def test_completion_choice_creation(self):
        """Test creating a CompletionChoice."""
        choice = CompletionChoice(text="Generated text", index=0, logprobs=None, finish_reason="length")
        assert choice.text == "Generated text"
        assert choice.index == 0
        assert choice.logprobs is None
        assert choice.finish_reason == "length"

    def test_completion_choice_with_logprobs(self):
        """Test CompletionChoice with logprobs."""
        choice = CompletionChoice(
            text="Text",
            index=0,
            logprobs={"tokens": ["test"], "token_logprobs": [-0.5]},
            finish_reason="stop",
        )
        assert choice.logprobs is not None
        assert "tokens" in choice.logprobs


class TestChatCompletionResponse:
    """Test the ChatCompletionResponse model."""

    def test_chat_completion_response_creation(self):
        """Test creating a ChatCompletionResponse."""
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert response.created == 1234567890
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 15

    def test_chat_completion_response_multiple_choices(self):
        """Test ChatCompletionResponse with multiple choices."""
        response = ChatCompletionResponse(
            id="chatcmpl-456",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content="Response 1"),
                    finish_reason="stop",
                ),
                ChatCompletionChoice(
                    index=1,
                    message=Message(role="assistant", content="Response 2"),
                    finish_reason="stop",
                ),
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )
        assert len(response.choices) == 2
        assert response.choices[0].message.content == "Response 1"
        assert response.choices[1].message.content == "Response 2"


class TestCompletionResponse:
    """Test the CompletionResponse model."""

    def test_completion_response_creation(self):
        """Test creating a CompletionResponse."""
        response = CompletionResponse(
            id="cmpl-789",
            object="text_completion",
            created=1234567890,
            model="text-davinci-003",
            choices=[CompletionChoice(text="Completed text", index=0, logprobs=None, finish_reason="stop")],
            usage=Usage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
        )
        assert response.id == "cmpl-789"
        assert response.object == "text_completion"
        assert response.created == 1234567890
        assert response.model == "text-davinci-003"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 25


class TestModel:
    """Test the Model model."""

    def test_model_creation(self):
        """Test creating a Model."""
        model = Model(id="gpt-3.5-turbo", object="model", created=1677610602, owned_by="openai")
        assert model.id == "gpt-3.5-turbo"
        assert model.object == "model"
        assert model.created == 1677610602
        assert model.owned_by == "openai"


class TestModelsResponse:
    """Test the ModelsResponse model."""

    def test_models_response_creation(self):
        """Test creating a ModelsResponse."""
        response = ModelsResponse(
            object="list",
            data=[
                Model(
                    id="gpt-3.5-turbo",
                    object="model",
                    created=1677610602,
                    owned_by="openai",
                ),
                Model(id="gpt-4", object="model", created=1687882410, owned_by="openai"),
            ],
        )
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "gpt-3.5-turbo"
        assert response.data[1].id == "gpt-4"

    def test_models_response_empty(self):
        """Test ModelsResponse with no models."""
        response = ModelsResponse(object="list", data=[])
        assert response.object == "list"
        assert len(response.data) == 0
