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

"""Unit tests for the Guardrails class.

These tests mock the underlying LLMRails instantiation and verify that the Guardrails
class correctly delegates method calls with properly formatted parameters.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.guardrails.guardrails import Guardrails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationResponse


@pytest.fixture
def mock_rails_config():
    """Create a mock RailsConfig for testing."""
    config = MagicMock(spec=RailsConfig)
    return config


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    return llm


class TestGuardrailsInit:
    """Tests for Guardrails.__init__ method."""

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_init_without_llm(self, mock_llmrails_class, mock_rails_config):
        """Test initialization without providing an LLM."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config, verbose=False)

        # Verify LLMRails was instantiated with config only
        mock_llmrails_class.assert_called_once_with(mock_rails_config, None, False)

        # Verify attributes are set correctly
        assert guardrails.config == mock_rails_config
        assert guardrails.llm is None
        assert guardrails.verbose is False
        assert guardrails.llmrails == mock_llmrails_instance

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_init_with_llm(self, mock_llmrails_class, mock_rails_config, mock_llm):
        """Test initialization with a custom LLM."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        guardrails = Guardrails(config=mock_rails_config, llm=mock_llm, verbose=True)

        # Verify LLMRails was instantiated with both config and llm
        mock_llmrails_class.assert_called_once_with(mock_rails_config, mock_llm, True)

        # Verify attributes are set correctly
        assert guardrails.config == mock_rails_config
        assert guardrails.llm == mock_llm
        assert guardrails.verbose is True
        assert guardrails.llmrails == mock_llmrails_instance


class TestConvertToMessages:
    """Tests for the _convert_to_messages static method."""

    def test_prompt_string(self):
        """Test conversion of string prompt to LLMMessages."""
        result = Guardrails._convert_to_messages(prompt="Hello, how are you?")

        expected = [{"role": "user", "content": "Hello, how are you?"}]
        assert result == expected

    def test_empty_string_prompt(self):
        """Test conversion of empty string prompt raises ValueError."""
        # Empty string is falsy, so it should raise an error
        with pytest.raises(ValueError, match="Neither prompt nor messages provided"):
            Guardrails._convert_to_messages(prompt="")

    def test_messages_single_message(self):
        """Test conversion with single message."""
        messages = [{"role": "user", "content": "What is the weather?"}]
        result = Guardrails._convert_to_messages(messages=messages)

        expected = [{"role": "user", "content": "What is the weather?"}]
        assert result == expected

    def test_messages_multiple_messages(self):
        """Test conversion with multiple messages."""
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Tell me more."},
        ]
        result = Guardrails._convert_to_messages(messages=messages)

        expected = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Tell me more."},
        ]
        assert result == expected

    def test_empty_messages_list(self):
        """Test conversion with empty messages list raises ValueError."""
        # Empty list is falsy, so it should raise an error
        messages = []
        with pytest.raises(ValueError, match="Neither prompt nor messages provided"):
            Guardrails._convert_to_messages(messages=messages)

    def test_prompt_takes_priority_over_messages(self):
        """Test that messages parameter takes priority when both are provided."""
        messages = [{"role": "user", "content": "From messages"}]
        result = Guardrails._convert_to_messages(prompt="From prompt", messages=messages)

        # Messages should take priority
        expected = [{"role": "user", "content": "From messages"}]
        assert result == expected

    def test_neither_prompt_nor_messages_raises_error(self):
        """Test that providing neither prompt nor messages raises ValueError."""
        with pytest.raises(ValueError, match="Neither prompt nor messages provided"):
            Guardrails._convert_to_messages()

    def test_multiline_string_prompt(self):
        """Test conversion of multiline string prompt."""
        multiline_prompt = """Line 1
Line 2
Line 3"""
        result = Guardrails._convert_to_messages(prompt=multiline_prompt)

        expected = [{"role": "user", "content": multiline_prompt}]
        assert result == expected

    def test_string_prompt_with_special_characters(self):
        """Test conversion of string prompt with special characters."""
        special_prompt = "Hello! @#$%^&*() How's the weather? \"quoted\" 'text'"
        result = Guardrails._convert_to_messages(prompt=special_prompt)

        expected = [{"role": "user", "content": special_prompt}]
        assert result == expected


class TestGenerate:
    """Tests for the synchronous generate method."""

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_with_string_prompt(self, mock_llmrails_class, mock_rails_config):
        """Test generate method with a string prompt."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = "Generated response"

        guardrails = Guardrails(config=mock_rails_config)
        result = guardrails.generate(prompt="Hello!")

        # Verify generate was called with correct messages
        expected_messages = [{"role": "user", "content": "Hello!"}]
        mock_llmrails_instance.generate.assert_called_once_with(messages=expected_messages)
        assert result == "Generated response"

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_with_messages(self, mock_llmrails_class, mock_rails_config):
        """Test generate method with a list of messages."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = "Response to conversation"

        guardrails = Guardrails(config=mock_rails_config)
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Tell me more."},
        ]
        result = guardrails.generate(messages=messages)

        # Verify generate was called with converted messages
        expected_messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Tell me more."},
        ]
        mock_llmrails_instance.generate.assert_called_once_with(messages=expected_messages)
        assert result == "Response to conversation"

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_with_kwargs(self, mock_llmrails_class, mock_rails_config):
        """Test generate method with additional kwargs."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = "Response"
        generation_options = {"temperature": 0.7, "max_tokens": 100}

        guardrails = Guardrails(config=mock_rails_config)
        result = guardrails.generate(prompt="Test", options=generation_options)

        # Verify kwargs were passed through
        expected_messages = [{"role": "user", "content": "Test"}]
        mock_llmrails_instance.generate.assert_called_once_with(messages=expected_messages, options=generation_options)
        assert result == "Response"

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_returns_dict(self, mock_llmrails_class, mock_rails_config):
        """Test generate method when LLMRails returns a dict."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = {
            "content": "Response",
            "metadata": {"tokens": 100},
        }

        guardrails = Guardrails(config=mock_rails_config)
        result = guardrails.generate(prompt="Test")

        assert result == {"content": "Response", "metadata": {"tokens": 100}}

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_returns_generation_response(self, mock_llmrails_class, mock_rails_config):
        """Test generate method when LLMRails returns GenerationResponse."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_response = GenerationResponse(response="Response text")
        mock_llmrails_instance.generate.return_value = mock_response

        guardrails = Guardrails(config=mock_rails_config)
        result = guardrails.generate(prompt="Test")

        assert result == mock_response

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_returns_tuple(self, mock_llmrails_class, mock_rails_config):
        """Test generate method when LLMRails returns a tuple."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = (
            {"response": "text"},
            {"state": "data"},
        )

        guardrails = Guardrails(config=mock_rails_config)
        result = guardrails.generate(prompt="Test")

        assert result == ({"response": "text"}, {"state": "data"})

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_empty_string(self, mock_llmrails_class, mock_rails_config):
        """Test generate method with an empty string prompt raises ValueError."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config)

        with pytest.raises(ValueError, match="Neither prompt nor messages provided"):
            guardrails.generate(prompt="")

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_neither_prompt_nor_messages_raises_error(self, mock_llmrails_class, mock_rails_config):
        """Test that calling generate with neither prompt nor messages raises ValueError."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config)

        with pytest.raises(ValueError, match="Neither prompt nor messages provided"):
            guardrails.generate()


class TestGenerateAsync:
    """Tests for the asynchronous generate_async method."""

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_generate_async_with_string_prompt(self, mock_llmrails_class, mock_rails_config):
        """Test generate_async method with a string prompt."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate_async = AsyncMock(return_value="Async response")

        guardrails = Guardrails(config=mock_rails_config)
        result = await guardrails.generate_async(prompt="Hello async!")

        # Verify generate_async was called with correct messages
        expected_messages = [{"role": "user", "content": "Hello async!"}]
        mock_llmrails_instance.generate_async.assert_awaited_once_with(messages=expected_messages)
        assert result == "Async response"

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_generate_async_with_messages(self, mock_llmrails_class, mock_rails_config):
        """Test generate_async method with a list of messages."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate_async = AsyncMock(return_value="Async conversation response")

        guardrails = Guardrails(config=mock_rails_config)
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
        ]
        result = await guardrails.generate_async(messages=messages)

        # Verify generate_async was called with converted messages
        expected_messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
        ]
        mock_llmrails_instance.generate_async.assert_awaited_once_with(messages=expected_messages)
        assert result == "Async conversation response"

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_generate_async_with_kwargs(self, mock_llmrails_class, mock_rails_config):
        """Test generate_async method with additional kwargs."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate_async = AsyncMock(return_value="Response")

        guardrails = Guardrails(config=mock_rails_config)
        result = await guardrails.generate_async(prompt="Test", temperature=0.5, top_p=0.9)

        # Verify kwargs were passed through
        expected_messages = [{"role": "user", "content": "Test"}]
        mock_llmrails_instance.generate_async.assert_awaited_once_with(
            messages=expected_messages, temperature=0.5, top_p=0.9
        )
        assert result == "Response"


class TestStreamAsync:
    """Tests for the asynchronous stream_async method."""

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_with_string_prompt(self, mock_llmrails_class, mock_rails_config):
        """Test stream_async method with a string prompt."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        # Create an async iterator mock
        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=mock_rails_config)

        # Collect all chunks
        chunks = []
        async for chunk in guardrails.stream_async(prompt="Stream this"):
            chunks.append(chunk)

        # Verify stream_async was called with correct messages
        expected_messages = [{"role": "user", "content": "Stream this"}]
        mock_llmrails_instance.stream_async.assert_called_once_with(messages=expected_messages)
        assert chunks == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_with_messages(self, mock_llmrails_class, mock_rails_config):
        """Test stream_async method with a list of messages."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            yield "Response "
            yield "to "
            yield "conversation"

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=mock_rails_config)
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ]

        chunks = []
        async for chunk in guardrails.stream_async(messages=messages):
            chunks.append(chunk)

        expected_messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ]
        mock_llmrails_instance.stream_async.assert_called_once_with(messages=expected_messages)
        assert chunks == ["Response ", "to ", "conversation"]

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_with_kwargs(self, mock_llmrails_class, mock_rails_config):
        """Test stream_async method with additional kwargs."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            yield "chunk"

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=mock_rails_config)

        chunks = []
        async for chunk in guardrails.stream_async(prompt="Test", temperature=0.8):
            chunks.append(chunk)

        # Verify kwargs were passed through
        expected_messages = [{"role": "user", "content": "Test"}]
        mock_llmrails_instance.stream_async.assert_called_once_with(messages=expected_messages, temperature=0.8)

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_dict_chunks(self, mock_llmrails_class, mock_rails_config):
        """Test stream_async when it yields dict chunks."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            yield {"type": "start", "data": "beginning"}
            yield {"type": "content", "data": "middle"}
            yield {"type": "end", "data": "finish"}

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=mock_rails_config)

        chunks = []
        async for chunk in guardrails.stream_async(prompt="Stream dict"):
            chunks.append(chunk)

        assert chunks == [
            {"type": "start", "data": "beginning"},
            {"type": "content", "data": "middle"},
            {"type": "end", "data": "finish"},
        ]

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_empty_stream(self, mock_llmrails_class, mock_rails_config):
        """Test stream_async when stream is empty."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            # Empty stream
            if False:
                yield

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=mock_rails_config)

        chunks = []
        async for chunk in guardrails.stream_async(prompt="Empty stream"):
            chunks.append(chunk)

        assert chunks == []

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_single_chunk(self, mock_llmrails_class, mock_rails_config):
        """Test stream_async with a single chunk."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            yield "single chunk"

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=mock_rails_config)

        chunks = []
        async for chunk in guardrails.stream_async(prompt="Single chunk test"):
            chunks.append(chunk)

        assert chunks == ["single chunk"]

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_neither_prompt_nor_messages_raises_error(self, mock_llmrails_class, mock_rails_config):
        """Test that stream_async with neither prompt nor messages raises ValueError."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config)

        with pytest.raises(ValueError, match="Neither prompt nor messages provided"):
            # Need to iterate to trigger the error
            async for _ in guardrails.stream_async():
                pass


class TestIntegration:
    """Integration tests verifying end-to-end behavior."""

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_multiple_calls_same_instance(self, mock_llmrails_class, mock_rails_config):
        """Test that the same Guardrails instance can be used for multiple calls."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate_async = AsyncMock(side_effect=["Response 1", "Response 2", "Response 3"])

        guardrails = Guardrails(config=mock_rails_config)

        result1 = await guardrails.generate_async(prompt="First call")
        result2 = await guardrails.generate_async(prompt="Second call")
        result3 = await guardrails.generate_async(prompt="Third call")

        assert result1 == "Response 1"
        assert result2 == "Response 2"
        assert result3 == "Response 3"
        assert mock_llmrails_instance.generate_async.await_count == 3

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_with_custom_llm_initialization(self, mock_llmrails_class, mock_rails_config, mock_llm):
        """Test that custom LLM is properly passed through to LLMRails."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config, llm=mock_llm)

        # Verify the custom LLM was passed to LLMRails
        mock_llmrails_class.assert_called_once_with(mock_rails_config, mock_llm, False)
        assert guardrails.llm == mock_llm

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_with_additional_parameters(self, mock_llmrails_class, mock_rails_config):
        """Test that additional parameters can be passed through kwargs."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = "Response"

        guardrails = Guardrails(config=mock_rails_config)

        result = guardrails.generate(
            prompt="Test",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )

        # Verify all kwargs were passed through
        expected_messages = [{"role": "user", "content": "Test"}]
        mock_llmrails_instance.generate.assert_called_once_with(
            messages=expected_messages,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )
        assert result == "Response"


class TestUtilityMethods:
    """Tests for utility methods: explain() and update_llm()."""

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_explain_returns_explain_info(self, mock_llmrails_class, mock_rails_config):
        """Test that explain() returns the ExplainInfo from LLMRails."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        # Create a mock ExplainInfo object
        mock_explain_info = MagicMock()
        mock_llmrails_instance.explain.return_value = mock_explain_info

        guardrails = Guardrails(config=mock_rails_config)
        result = guardrails.explain()

        # Verify explain was called on underlying LLMRails
        mock_llmrails_instance.explain.assert_called_once()

        # Verify the ExplainInfo object is returned
        assert result == mock_explain_info

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_explain_delegates_to_llmrails(self, mock_llmrails_class, mock_rails_config):
        """Test that explain() delegates to llmrails.explain()."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config)
        guardrails.explain()

        # Verify the delegation happened
        mock_llmrails_instance.explain.assert_called_once_with()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_update_llm_updates_instance_llm(self, mock_llmrails_class, mock_rails_config, mock_llm):
        """Test that update_llm() updates the Guardrails llm attribute."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config)

        # Initially llm is None
        assert guardrails.llm is None

        # Create a new mock LLM
        new_llm = MagicMock()
        guardrails.update_llm(new_llm)

        # Verify the llm attribute was updated
        assert guardrails.llm == new_llm

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_update_llm_delegates_to_llmrails(self, mock_llmrails_class, mock_rails_config):
        """Test that update_llm() calls llmrails.update_llm()."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config)

        new_llm = MagicMock()
        guardrails.update_llm(new_llm)

        # Verify update_llm was called on underlying LLMRails with the new LLM
        mock_llmrails_instance.update_llm.assert_called_once_with(new_llm)

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_update_llm_with_initial_llm(self, mock_llmrails_class, mock_rails_config, mock_llm):
        """Test update_llm() when Guardrails was initialized with an LLM."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        # Initialize with an LLM
        initial_llm = MagicMock()
        guardrails = Guardrails(config=mock_rails_config, llm=initial_llm)

        assert guardrails.llm == initial_llm

        # Update to a new LLM
        new_llm = MagicMock()
        guardrails.update_llm(new_llm)

        # Verify the llm attribute was updated
        assert guardrails.llm == new_llm
        assert guardrails.llm != initial_llm

        # Verify update_llm was called on underlying LLMRails
        mock_llmrails_instance.update_llm.assert_called_once_with(new_llm)

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_update_llm_multiple_times(self, mock_llmrails_class, mock_rails_config):
        """Test that update_llm() can be called multiple times."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=mock_rails_config)

        # Update LLM multiple times
        llm1 = MagicMock()
        llm2 = MagicMock()
        llm3 = MagicMock()

        guardrails.update_llm(llm1)
        assert guardrails.llm == llm1

        guardrails.update_llm(llm2)
        assert guardrails.llm == llm2

        guardrails.update_llm(llm3)
        assert guardrails.llm == llm3

        # Verify update_llm was called three times on underlying LLMRails
        assert mock_llmrails_instance.update_llm.call_count == 3
        mock_llmrails_instance.update_llm.assert_any_call(llm1)
        mock_llmrails_instance.update_llm.assert_any_call(llm2)
        mock_llmrails_instance.update_llm.assert_any_call(llm3)

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_explain_after_generation(self, mock_llmrails_class, mock_rails_config):
        """Test explain() works after a generation call."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = "Response"

        mock_explain_info = MagicMock()
        mock_explain_info.llm_calls = ["call1", "call2"]
        mock_llmrails_instance.explain.return_value = mock_explain_info

        guardrails = Guardrails(config=mock_rails_config)

        # Generate a response
        guardrails.generate(prompt="Test")

        # Then get explain info
        explain_info = guardrails.explain()

        assert explain_info == mock_explain_info
        assert explain_info.llm_calls == ["call1", "call2"]
        mock_llmrails_instance.explain.assert_called_once()
