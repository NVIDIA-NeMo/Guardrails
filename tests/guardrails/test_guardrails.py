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
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationOptions
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG, NEMOGUARDS_CONFIG

# Valid IORails input/output rails for has_only_iorails_flows tests
_IORAILS_BASE_RAILS = {
    "input": {"flows": ["content safety check input $model=content_safety"]},
    "output": {"flows": ["content safety check output $model=content_safety"]},
}


def _make_iorails_config(rails: dict, extra_prompts: list | None = None) -> RailsConfig:
    """Build a RailsConfig with the given rails section."""
    prompts = list(NEMOGUARDS_CONFIG["prompts"])
    if extra_prompts:
        prompts.extend(extra_prompts)
    return RailsConfig.from_content(
        config={
            "models": [
                {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
                {"type": "content_safety", "engine": "nim", "model": "nvidia/llama-3.1-nemoguard-8b-content-safety"},
            ],
            "rails": rails,
            "prompts": prompts,
        }
    )


@pytest.fixture
def _nemoguards_rails_config():
    """Create a real RailsConfig matching the nemoguards_v2 example config."""
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
def _content_safety_rails_config():
    """Create a real RailsConfig matching the nemoguards_v2 example config."""
    return RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    return llm


class TestGuardrailsRouting:
    """Tests to check the routing of requests to Guardrails between LLMRails and IORails"""

    @pytest.mark.asyncio
    @patch.object(LLMRails, "__init__", return_value=None)
    async def test_use_iorails_false_uses_llmrails_only(self, mock_llmrails_init, _content_safety_rails_config):
        """Test if Guardrails is initialized with `use_iorails` == False and an IORails-compatible config
        all calls go to LLMRails.

        We patch __init__ (rather than the class itself) so that IORails and LLMRails remain real
        classes. This lets the isinstance() checks in guardrails.py work correctly, while still
        giving us uninitialized instances whose methods we can replace with mocks.
        """

        async with Guardrails(config=_content_safety_rails_config, verbose=False, use_iorails=False) as guardrails:
            # Content-safety config is supported by IORails, but use_iorails=False overrides
            assert guardrails._has_only_iorails_flows()
            assert isinstance(guardrails.rails_engine, LLMRails)

            # Set up mocks on the real (but uninitialized) LLMRails instance
            explain_info = ExplainInfo()
            mock_new_llm = MagicMock()

            async def mock_stream():
                yield "chunk1"

            guardrails.rails_engine.generate = MagicMock(return_value="generate() response")
            guardrails.rails_engine.generate_async = AsyncMock(return_value="generate_async() response")
            guardrails.rails_engine.explain = MagicMock(return_value=explain_info)
            guardrails.rails_engine.stream_async = MagicMock(return_value=mock_stream())
            guardrails.rails_engine.update_llm = MagicMock()

            # Call all methods
            messages = [{"role": "user", "content": "Hi how are you"}]
            assert guardrails.generate(messages=messages) == "generate() response"
            assert await guardrails.generate_async(messages=messages) == "generate_async() response"
            chunks = [chunk async for chunk in guardrails.stream_async(messages=messages)]
            assert chunks == ["chunk1"]
            assert guardrails.explain() is explain_info
            guardrails.update_llm(mock_new_llm)

            # Verify all calls went to LLMRails
            guardrails.rails_engine.generate.assert_called_once_with(messages=messages)
            guardrails.rails_engine.generate_async.assert_called_once_with(messages=messages)
            guardrails.rails_engine.stream_async.assert_called_once_with(messages=messages)
            guardrails.rails_engine.explain.assert_called_once()
            guardrails.rails_engine.update_llm.assert_called_once_with(mock_new_llm)

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_use_iorails_true_iorails_config(
        self, mock_iorails_init, mock_start, mock_stop, _content_safety_rails_config
    ):
        """Test if Guardrails is initialized with `use_iorails` == True, and a config that
        can be run by IORails, that calls are routed to IORails where implemented and exceptions
        are raised where not.

        We patch __init__ (rather than the class itself) so that IORails and LLMRails remain real
        classes. This lets the isinstance() checks in guardrails.py work correctly, while still
        giving us uninitialized instances whose methods we can replace with mocks.
        start/stop are also patched because the __init__ patch leaves the instance without
        _running, so the real methods would raise AttributeError during startup/shutdown.
        """

        async with Guardrails(config=_content_safety_rails_config, verbose=False, use_iorails=True) as guardrails:
            assert guardrails._has_only_iorails_flows()
            assert isinstance(guardrails.rails_engine, IORails)

            # Mock generate (sync) and generate_async on IORails
            guardrails.rails_engine.generate = MagicMock(return_value="iorails generate response")
            guardrails.rails_engine.generate_async = AsyncMock(return_value="iorails generate_async response")

            messages = [{"role": "user", "content": "Hi how are you"}]
            mock_new_llm = MagicMock()

            # Mock stream_async on the IORails instance
            async def mock_stream():
                yield "iorails chunk"

            guardrails.rails_engine.stream_async = MagicMock(return_value=mock_stream())

            assert guardrails.generate(messages=messages) == "iorails generate response"

            response = await guardrails.generate_async(messages=messages)
            assert response == "iorails generate_async response"

            chunks = [chunk async for chunk in guardrails.stream_async(messages=messages)]
            assert chunks == ["iorails chunk"]

            with pytest.raises(NotImplementedError, match="IORails doesn't support explain()"):
                guardrails.explain()

            with pytest.raises(NotImplementedError, match="IORails doesn't support update_llm()"):
                guardrails.update_llm(mock_new_llm)

            guardrails.rails_engine.generate.assert_called_once_with(messages=messages)
            guardrails.rails_engine.generate_async.assert_called_once_with(messages=messages)
            guardrails.rails_engine.stream_async.assert_called_once_with(
                messages=messages,
                options=None,
                include_metadata=False,
            )

    @pytest.mark.asyncio
    @patch.object(LLMRails, "__init__", return_value=None)
    async def test_use_iorails_true_llmrails_config(self, mock_llmrails_init):
        """Test if Guardrails is initialized with `use_iorails` == True but the RailsConfig
        requires LLMRails all calls still go to LLMRails.

        We use a config with 'self check input' which is NOT supported by IORails.
        We patch __init__ (rather than the class itself) so that IORails and LLMRails remain real
        classes. This lets the isinstance() checks in guardrails.py work correctly, while still
        giving us uninitialized instances whose methods we can replace with mocks.
        """
        unsupported_config = _make_iorails_config(
            rails={
                "input": {"flows": ["self check input"]},
                "output": {"flows": ["content safety check output $model=content_safety"]},
            },
            extra_prompts=[{"task": "self_check_input", "content": "placeholder"}],
        )

        async with Guardrails(config=unsupported_config, verbose=False, use_iorails=True) as guardrails:
            assert not guardrails._has_only_iorails_flows()
            assert isinstance(guardrails.rails_engine, LLMRails)

            # Set up mocks on the real (but uninitialized) LLMRails instance
            explain_info = ExplainInfo()
            mock_new_llm = MagicMock()

            async def mock_stream():
                yield "chunk1"

            guardrails.rails_engine.generate = MagicMock(return_value="generate() response")
            guardrails.rails_engine.generate_async = AsyncMock(return_value="generate_async() response")
            guardrails.rails_engine.explain = MagicMock(return_value=explain_info)
            guardrails.rails_engine.stream_async = MagicMock(return_value=mock_stream())
            guardrails.rails_engine.update_llm = MagicMock()

            # Call all methods
            messages = [{"role": "user", "content": "Hi how are you"}]
            assert guardrails.generate(messages=messages) == "generate() response"
            assert await guardrails.generate_async(messages=messages) == "generate_async() response"
            chunks = [chunk async for chunk in guardrails.stream_async(messages=messages)]
            assert chunks == ["chunk1"]
            assert guardrails.explain() is explain_info
            guardrails.update_llm(mock_new_llm)

            # Verify all calls went to LLMRails
            guardrails.rails_engine.generate.assert_called_once_with(messages=messages)
            guardrails.rails_engine.generate_async.assert_called_once_with(messages=messages)
            guardrails.rails_engine.stream_async.assert_called_once_with(messages=messages)
            guardrails.rails_engine.explain.assert_called_once()
            guardrails.rails_engine.update_llm.assert_called_once_with(mock_new_llm)


class TestGuardrailsInit:
    """Tests for Guardrails.__init__ method."""

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_init_without_llm(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test initialization without providing an LLM."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, verbose=False, use_iorails=False)

        # Verify LLMRails was instantiated with config only
        mock_llmrails_class.assert_called_once_with(_nemoguards_rails_config, None, False)

        # Verify attributes are set correctly
        assert guardrails.config == _nemoguards_rails_config
        assert guardrails.verbose is False
        assert guardrails.rails_engine == mock_llmrails_instance

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_init_with_llm(self, mock_llmrails_class, _nemoguards_rails_config, mock_llm):
        """Test initialization with a custom LLM."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        guardrails = Guardrails(config=_nemoguards_rails_config, llm=mock_llm, verbose=True, use_iorails=False)

        # Verify LLMRails was instantiated with both config and llm
        mock_llmrails_class.assert_called_once_with(_nemoguards_rails_config, mock_llm, True)

        # Verify attributes are set correctly
        assert guardrails.config == _nemoguards_rails_config
        assert guardrails.verbose is True
        assert guardrails.rails_engine == mock_llmrails_instance

    @patch.object(IORails, "__init__", return_value=None)
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_init_with_llm_forces_llmrails(
        self, mock_llmrails_class, mock_iorails_init, _content_safety_rails_config, mock_llm
    ):
        """Passing `llm` forces LLMRails even with use_iorails=True and an IORails-compatible config."""
        mock_llmrails_class.return_value = MagicMock()
        Guardrails(config=_content_safety_rails_config, llm=mock_llm, use_iorails=True)
        mock_llmrails_class.assert_called_once_with(_content_safety_rails_config, mock_llm, False)
        mock_iorails_init.assert_not_called()

    @patch.object(IORails, "__init__", return_value=None)
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_init_with_llm_use_iorails_false_uses_llmrails(
        self, mock_llmrails_class, mock_iorails_init, _content_safety_rails_config, mock_llm
    ):
        """Passing `llm` with use_iorails=False routes to LLMRails even on an IORails-compatible config."""
        mock_llmrails_class.return_value = MagicMock()
        Guardrails(config=_content_safety_rails_config, llm=mock_llm, use_iorails=False)
        mock_llmrails_class.assert_called_once_with(_content_safety_rails_config, mock_llm, False)
        mock_iorails_init.assert_not_called()

    @patch.object(IORails, "__init__", return_value=None)
    def test_init_without_llm_uses_iorails(self, mock_iorails_init, _content_safety_rails_config):
        """Omitting `llm` (the default) selects IORails on an IORails-compatible config."""
        guardrails = Guardrails(config=_content_safety_rails_config, use_iorails=True)
        assert isinstance(guardrails.rails_engine, IORails)
        mock_iorails_init.assert_called_once_with(_content_safety_rails_config)


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
        assert result == messages

    def test_messages_multiple_messages(self):
        """Test conversion with multiple messages."""
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Tell me more."},
        ]
        result = Guardrails._convert_to_messages(messages=messages)

        assert result == messages

    def test_messages_with_system_message(self):
        """Test conversion with system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = Guardrails._convert_to_messages(messages=messages)

        assert result == messages

    def test_empty_messages_list(self):
        """Test conversion with empty messages list raises ValueError."""
        # Empty list is falsy, so it should raise an error
        messages = []
        with pytest.raises(ValueError, match="Neither prompt nor messages provided"):
            Guardrails._convert_to_messages(messages=messages)

    def test_messages_take_priority_over_prompt(self):
        """Test that messages parameter takes priority when both are provided."""
        messages = [{"role": "user", "content": "From messages"}]
        result = Guardrails._convert_to_messages(prompt="From prompt", messages=messages)
        assert result == messages

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


class TestGenerateAsync:
    """Tests for the asynchronous generate_async method."""

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_generate_async_with_string_prompt(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test generate_async method with a string prompt using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate_async = AsyncMock(return_value="Async response")

        async with Guardrails(config=_nemoguards_rails_config, use_iorails=False) as guardrails:
            result = await guardrails.generate_async(prompt="Hello async!")

            # Verify generate_async was called with correct messages
            expected_messages = [{"role": "user", "content": "Hello async!"}]
            mock_llmrails_instance.generate_async.assert_awaited_once_with(messages=expected_messages)
            assert result == "Async response"

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_generate_async_with_messages(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test generate_async method with a list of messages using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate_async = AsyncMock(return_value="Async conversation response")

        async with Guardrails(config=_nemoguards_rails_config, use_iorails=False) as guardrails:
            messages = [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"},
            ]
            result = await guardrails.generate_async(messages=messages)

            mock_llmrails_instance.generate_async.assert_awaited_once_with(messages=messages)
            assert result == "Async conversation response"

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_generate_async_with_kwargs(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test generate_async method with additional kwargs using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate_async = AsyncMock(return_value="Response")

        async with Guardrails(config=_nemoguards_rails_config, use_iorails=False) as guardrails:
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
    async def test_stream_async_with_string_prompt(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test stream_async method with a string prompt using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        # Create an async iterator mock
        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        chunks = []
        async for chunk in guardrails.stream_async(prompt="Stream this"):
            chunks.append(chunk)

        # Verify stream_async was called with correct messages
        expected_messages = [{"role": "user", "content": "Stream this"}]
        mock_llmrails_instance.stream_async.assert_called_once_with(messages=expected_messages)
        assert chunks == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_with_messages(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test stream_async method with a list of messages using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            yield "Response "
            yield "to "
            yield "conversation"

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ]

        chunks = []
        async for chunk in guardrails.stream_async(messages=messages):
            chunks.append(chunk)

        mock_llmrails_instance.stream_async.assert_called_once_with(messages=messages)
        assert chunks == ["Response ", "to ", "conversation"]

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_with_kwargs(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test stream_async method with additional kwargs using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            yield "chunk"

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        chunks = []
        async for chunk in guardrails.stream_async(prompt="Test", temperature=0.8):
            chunks.append(chunk)

        # Verify kwargs were passed through
        expected_messages = [{"role": "user", "content": "Test"}]
        mock_llmrails_instance.stream_async.assert_called_once_with(messages=expected_messages, temperature=0.8)

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_dict_chunks(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test stream_async when it yields dict chunks using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        return_chunks = [
            {"type": "start", "data": "beginning"},
            {"type": "content", "data": "middle"},
            {"type": "end", "data": "finish"},
        ]

        async def mock_stream():
            yield return_chunks[0]
            yield return_chunks[1]
            yield return_chunks[2]

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        chunks = []
        async for chunk in guardrails.stream_async(prompt="Stream dict"):
            chunks.append(chunk)

        assert chunks == return_chunks

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_empty_stream(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test stream_async when stream is empty using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            # Empty stream
            if False:
                yield

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        chunks = []
        async for chunk in guardrails.stream_async(prompt="Empty stream"):
            chunks.append(chunk)

        assert chunks == []

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_single_chunk(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test stream_async with a single chunk using context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        async def mock_stream():
            yield "single chunk"

        mock_llmrails_instance.stream_async.return_value = mock_stream()

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        chunks = []
        async for chunk in guardrails.stream_async(prompt="Single chunk test"):
            chunks.append(chunk)

        assert chunks == ["single chunk"]

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_stream_async_neither_prompt_nor_messages_raises_error(
        self, mock_llmrails_class, _nemoguards_rails_config
    ):
        """Test that stream_async with neither prompt nor messages raises ValueError."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        with pytest.raises(ValueError, match="Neither prompt nor messages provided"):
            # Error raised during stream creation, before iteration
            guardrails.stream_async()


class TestIntegration:
    """Integration tests verifying end-to-end behavior."""

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_multiple_calls_same_instance(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test that the same Guardrails instance can be used for multiple calls with context manager."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate_async = AsyncMock(side_effect=["Response 1", "Response 2", "Response 3"])

        async with Guardrails(config=_nemoguards_rails_config, use_iorails=False) as guardrails:
            result1 = await guardrails.generate_async(prompt="First call")
            result2 = await guardrails.generate_async(prompt="Second call")
            result3 = await guardrails.generate_async(prompt="Third call")

            assert result1 == "Response 1"
            assert result2 == "Response 2"
            assert result3 == "Response 3"
            assert mock_llmrails_instance.generate_async.await_count == 3

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_with_custom_llm_initialization(self, mock_llmrails_class, _nemoguards_rails_config, mock_llm):
        """Test that custom LLM is properly passed through to LLMRails."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, llm=mock_llm, use_iorails=False)

        # Verify the custom LLM was passed to LLMRails
        mock_llmrails_class.assert_called_once_with(_nemoguards_rails_config, mock_llm, False)

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_with_additional_parameters(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test that additional parameters can be passed through kwargs."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = "Response"

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

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
    def test_explain_delegates_to_llmrails(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test that explain() delegates to llmrails.explain()."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        guardrails.explain()

        # Verify the delegation happened
        mock_llmrails_instance.explain.assert_called_once_with()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_update_llm_delegates_new_llm(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test that update_llm() delegates the new LLM to LLMRails."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

        new_llm = MagicMock()
        guardrails.update_llm(new_llm)

        mock_llmrails_instance.update_llm.assert_called_once_with(new_llm)

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_update_llm_with_initial_llm(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test update_llm() when Guardrails was initialized with an LLM."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        # Initialize with an LLM
        initial_llm = MagicMock()
        guardrails = Guardrails(config=_nemoguards_rails_config, llm=initial_llm, use_iorails=False)

        # Verify initial LLM was passed to LLMRails
        mock_llmrails_class.assert_called_once_with(_nemoguards_rails_config, initial_llm, False)

        # Update to a new LLM
        new_llm = MagicMock()
        guardrails.update_llm(new_llm)

        # Verify update_llm was called on underlying LLMRails
        mock_llmrails_instance.update_llm.assert_called_once_with(new_llm)

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_update_llm_called_multiple_times(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test that update_llm() can be called multiple times."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

        # Update LLM multiple times
        llm1 = MagicMock()
        llm2 = MagicMock()
        llm3 = MagicMock()

        guardrails.update_llm(llm1)
        guardrails.update_llm(llm2)
        guardrails.update_llm(llm3)

        # Verify update_llm was called three times on underlying LLMRails
        assert mock_llmrails_instance.update_llm.call_count == 3
        mock_llmrails_instance.update_llm.assert_any_call(llm1)
        mock_llmrails_instance.update_llm.assert_any_call(llm2)
        mock_llmrails_instance.update_llm.assert_any_call(llm3)

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_explain_after_generation(self, mock_llmrails_class, _nemoguards_rails_config):
        """Test explain() works after a generation call."""
        mock_llmrails_instance = MagicMock()
        mock_llmrails_class.return_value = mock_llmrails_instance
        mock_llmrails_instance.generate.return_value = "Response"

        mock_explain_info = MagicMock()
        mock_explain_info.llm_calls = ["call1", "call2"]
        mock_llmrails_instance.explain.return_value = mock_explain_info

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

        # Generate a response
        guardrails.generate(prompt="Test")

        # Then get explain info
        explain_info = guardrails.explain()

        assert explain_info == mock_explain_info
        assert explain_info.llm_calls == ["call1", "call2"]
        mock_llmrails_instance.explain.assert_called_once()


class TestGuardrailsAttributes:
    """Tests for the llm, runtime, and config attribute accessors on Guardrails.

    Under LLMRails, llm/runtime delegate to the underlying instance.
    Under IORails, llm/runtime raise NotImplementedError. config is a plain
    attribute on Guardrails and is available under both engines.
    """

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_llm_property_delegates_to_llmrails(self, mock_llmrails_class, _nemoguards_rails_config):
        """guardrails.llm returns the underlying LLMRails.llm."""
        sentinel_llm = MagicMock()
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.llm = sentinel_llm
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        assert guardrails.llm is sentinel_llm

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_runtime_property_delegates_to_llmrails(self, mock_llmrails_class, _nemoguards_rails_config):
        """guardrails.runtime returns the underlying LLMRails.runtime."""
        sentinel_runtime = MagicMock()
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.runtime = sentinel_runtime
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        assert guardrails.runtime is sentinel_runtime

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_llm_property_reflects_update_llm(self, mock_llmrails_class, _nemoguards_rails_config):
        """After update_llm() swaps the LLM on LLMRails, guardrails.llm reads through
        to the new value (no caching on the facade)."""
        mock_llmrails_instance = MagicMock()
        initial_llm = MagicMock(name="initial")
        mock_llmrails_instance.llm = initial_llm
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        assert guardrails.llm is initial_llm

        # Simulate update_llm flipping the underlying attribute
        new_llm = MagicMock(name="new")
        mock_llmrails_instance.llm = new_llm
        guardrails.update_llm(new_llm)
        assert guardrails.llm is new_llm

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_config_attribute_on_llmrails(self, mock_llmrails_class, _nemoguards_rails_config):
        """guardrails.config is the same RailsConfig instance passed in."""
        mock_llmrails_class.return_value = MagicMock()
        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        assert guardrails.config is _nemoguards_rails_config

    @patch.object(IORails, "__init__", return_value=None)
    def test_llm_property_raises_on_iorails(self, mock_iorails_init, _content_safety_rails_config):
        """guardrails.llm raises NotImplementedError when running on IORails."""
        guardrails = Guardrails(config=_content_safety_rails_config, use_iorails=True)
        assert isinstance(guardrails.rails_engine, IORails)

        with pytest.raises(NotImplementedError, match="IORails doesn't support llm attribute access"):
            _ = guardrails.llm

    @patch.object(IORails, "__init__", return_value=None)
    def test_runtime_property_raises_on_iorails(self, mock_iorails_init, _content_safety_rails_config):
        """guardrails.runtime raises NotImplementedError when running on IORails."""
        guardrails = Guardrails(config=_content_safety_rails_config, use_iorails=True)
        assert isinstance(guardrails.rails_engine, IORails)

        with pytest.raises(NotImplementedError, match="IORails doesn't support runtime attribute access"):
            _ = guardrails.runtime

    @patch.object(IORails, "__init__", return_value=None)
    def test_config_attribute_on_iorails(self, mock_iorails_init, _content_safety_rails_config):
        """guardrails.config is accessible regardless of which engine is in use."""
        guardrails = Guardrails(config=_content_safety_rails_config, use_iorails=True)
        assert isinstance(guardrails.rails_engine, IORails)
        assert guardrails.config is _content_safety_rails_config


class TestGuardrailsLifecycle:
    """Test that startup/shutdown delegate to the rails engine."""

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_startup_calls_start_on_iorails(self, mock_init, mock_start, mock_stop, _content_safety_rails_config):
        """startup() delegates to IORails.start().
        start/stop are patched because the __init__ patch leaves the instance without
        _running, so the real methods would raise AttributeError.
        """
        guardrails = Guardrails(config=_content_safety_rails_config, verbose=False, use_iorails=True)
        assert isinstance(guardrails.rails_engine, IORails)

        await guardrails.startup()
        mock_start.assert_called_once()

        await guardrails.shutdown()
        mock_stop.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(LLMRails, "__init__", return_value=None)
    async def test_startup_skips_start_on_llmrails(self, mock_init, _nemoguards_rails_config):
        """startup() does not call start() on LLMRails (it has no start method)."""
        guardrails = Guardrails(config=_nemoguards_rails_config, verbose=False, use_iorails=False)
        assert isinstance(guardrails.rails_engine, LLMRails)

        # Should not raise even though LLMRails has no start/stop
        await guardrails.startup()
        await guardrails.shutdown()

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_startup_is_idempotent(self, mock_init, mock_start, mock_stop, _content_safety_rails_config):
        """Calling startup() twice only starts engines once."""
        guardrails = Guardrails(config=_content_safety_rails_config, verbose=False, use_iorails=True)
        await guardrails.startup()
        await guardrails.startup()
        mock_start.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_shutdown_without_startup_is_noop(
        self, mock_init, mock_start, mock_stop, _content_safety_rails_config
    ):
        """Calling shutdown() without startup() does not call stop."""
        guardrails = Guardrails(config=_content_safety_rails_config, verbose=False, use_iorails=True)
        await guardrails.shutdown()
        mock_stop.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_generate_async_lazy_starts(self, mock_init, mock_start, mock_stop, _content_safety_rails_config):
        """generate_async() calls startup() automatically if not already started."""
        guardrails = Guardrails(config=_content_safety_rails_config, verbose=False, use_iorails=True)
        guardrails._rails_engine.generate_async = AsyncMock(return_value={"role": "assistant", "content": "hi"})
        assert not guardrails._started
        await guardrails.generate_async(messages=[{"role": "user", "content": "hello"}])
        assert guardrails._started
        mock_start.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_stream_async_lazy_starts(self, mock_init, mock_start, mock_stop, _content_safety_rails_config):
        """stream_async() calls startup() automatically if not already started."""

        async def mock_stream():
            yield "hello"

        guardrails = Guardrails(config=_content_safety_rails_config, verbose=False, use_iorails=True)
        guardrails._rails_engine.stream_async = MagicMock(return_value=mock_stream())

        assert not guardrails._started
        chunks = [chunk async for chunk in guardrails.stream_async(messages=[{"role": "user", "content": "hi"}])]

        assert guardrails._started
        mock_start.assert_called_once()
        assert chunks == ["hello"]

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock, side_effect=RuntimeError("engine down"))
    @patch.object(IORails, "__init__", return_value=None)
    async def test_startup_failure_leaves_not_started(
        self, mock_init, mock_start, mock_stop, _content_safety_rails_config
    ):
        """If IORails.start() fails during startup(), _started stays False."""
        guardrails = Guardrails(config=_content_safety_rails_config, verbose=False, use_iorails=True)

        with pytest.raises(RuntimeError, match="engine down"):
            await guardrails.startup()

        assert not guardrails._started
        mock_start.assert_called_once()


class TestHasOnlyIORailsFlows:
    """Check all the permutations of configs with `has_only_iorails_flows()`"""

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_content_safety_has_only_iorails_flows(self, mock_llmrails_class, _content_safety_rails_config):
        """Check if we have config rails we don't use iorails"""
        guardrails = Guardrails(config=_content_safety_rails_config)
        assert guardrails._has_only_iorails_flows()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_nemoguards_has_only_iorails_flows(self, mock_llmrails_class, _nemoguards_rails_config):
        """Nemoguards config (content safety + topic safety + jailbreak) is supported by IORails."""
        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        assert guardrails._has_only_iorails_flows()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_has_only_iorails_flows_unsupported_retrieval_rails(self, mock_llmrails_class):
        """Check if we have retrieval rails we don't use iorails"""
        config = _make_iorails_config({**_IORAILS_BASE_RAILS, "retrieval": {"flows": ["check facts"]}})
        guardrails = Guardrails(config=config)
        assert not guardrails._has_only_iorails_flows()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_has_only_iorails_flows_unsupported_dialog_rails(self, mock_llmrails_class):
        """Check if we have dialog rails we don't use iorails"""
        config = _make_iorails_config({**_IORAILS_BASE_RAILS, "dialog": {}})
        guardrails = Guardrails(config=config)
        assert not guardrails._has_only_iorails_flows()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_has_only_iorails_flows_unsupported_actions_rails(self, mock_llmrails_class):
        """Check if we have actions rails we don't use iorails"""
        config = _make_iorails_config({**_IORAILS_BASE_RAILS, "actions": {"instant_actions": ["some_action"]}})
        guardrails = Guardrails(config=config)
        assert not guardrails._has_only_iorails_flows()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_has_only_iorails_flows_unsupported_tool_output_rails(self, mock_llmrails_class):
        """Check if we have tool_output rails we don't use iorails"""
        config = _make_iorails_config({**_IORAILS_BASE_RAILS, "tool_output": {"flows": ["check tool output"]}})
        guardrails = Guardrails(config=config)
        assert not guardrails._has_only_iorails_flows()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_has_only_iorails_flows_unsupported_tool_input_rails(self, mock_llmrails_class):
        """Check if we have tool_input rails we don't use iorails"""
        config = _make_iorails_config({**_IORAILS_BASE_RAILS, "tool_input": {"flows": ["check tool input"]}})
        guardrails = Guardrails(config=config)
        assert not guardrails._has_only_iorails_flows()

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_has_only_iorails_flows_with_topic_safety_input_rails(self, mock_llmrails_class):
        """Content safety + topic safety input rails are both supported by IORails."""
        config = RailsConfig.from_content(
            config={
                "models": [
                    {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
                    {
                        "type": "content_safety",
                        "engine": "nim",
                        "model": "nvidia/llama-3.1-nemoguard-8b-content-safety",
                    },
                    {"type": "topic_control", "engine": "nim", "model": "nvidia/llama-3.1-nemoguard-8b-topic-control"},
                ],
                "rails": {
                    "input": {
                        "flows": [
                            "content safety check input $model=content_safety",
                            "topic safety check input $model=topic_control",
                        ]
                    },
                    "output": {"flows": ["content safety check output $model=content_safety"]},
                },
                "prompts": [
                    *NEMOGUARDS_CONFIG["prompts"],
                    {"task": "topic_safety_check_input $model=topic_control", "content": "placeholder"},
                ],
            }
        )
        guardrails = Guardrails(config=config)
        assert guardrails._has_only_iorails_flows() is True

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_has_only_iorails_flows_unsupported_self_check_output_rails(self, mock_llmrails_class):
        """Check if we have input and output content safety **and also output self-check** we can't use IORails"""
        config = _make_iorails_config(
            rails={
                "input": {"flows": ["content safety check input $model=content_safety"]},
                "output": {
                    "flows": [
                        "content safety check output $model=content_safety",
                        "self check output",
                    ]
                },
            },
            extra_prompts=[{"task": "self_check_output", "content": "placeholder"}],
        )
        guardrails = Guardrails(config=config)
        assert guardrails._has_only_iorails_flows() is False


class TestStreamAsyncIORails:
    """Tests for stream_async when routed through IORails."""

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_delegates_to_iorails(self, mock_init, mock_start, mock_stop, _content_safety_rails_config):
        """stream_async delegates to IORails.stream_async with correct args."""

        async def mock_stream():
            yield "hello"
            yield " world"

        async with Guardrails(config=_content_safety_rails_config, use_iorails=True) as guardrails:
            assert isinstance(guardrails.rails_engine, IORails)
            guardrails.rails_engine.stream_async = MagicMock(return_value=mock_stream())

            chunks = []
            async for chunk in guardrails.stream_async(messages=[{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

            assert chunks == ["hello", " world"]
            guardrails.rails_engine.stream_async.assert_called_once_with(
                messages=[{"role": "user", "content": "hi"}],
                options=None,
                include_metadata=False,
            )

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_forwards_supported_kwargs(self, mock_init, mock_start, mock_stop, _content_safety_rails_config):
        """options and include_metadata are forwarded to IORails.stream_async."""

        async def mock_stream():
            yield "ok"

        opts = GenerationOptions(llm_params={"temperature": 0.5})

        async with Guardrails(config=_content_safety_rails_config, use_iorails=True) as guardrails:
            assert isinstance(guardrails.rails_engine, IORails)
            guardrails.rails_engine.stream_async = MagicMock(return_value=mock_stream())

            chunks = []
            async for chunk in guardrails.stream_async(
                messages=[{"role": "user", "content": "hi"}],
                options=opts,
                include_metadata=True,
            ):
                chunks.append(chunk)

            guardrails.rails_engine.stream_async.assert_called_once_with(
                messages=[{"role": "user", "content": "hi"}],
                options=opts,
                include_metadata=True,
            )

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_filters_unsupported_kwargs(self, mock_init, mock_start, mock_stop, _content_safety_rails_config):
        """LLMRails-only kwargs (state, generator, etc.) are not passed to IORails and a warning is logged."""

        async def mock_stream():
            yield "ok"

        async with Guardrails(config=_content_safety_rails_config, use_iorails=True) as guardrails:
            assert isinstance(guardrails.rails_engine, IORails)
            guardrails.rails_engine.stream_async = MagicMock(return_value=mock_stream())

            with patch("nemoguardrails.guardrails.guardrails.log") as mock_log:
                chunks = []
                async for chunk in guardrails.stream_async(
                    messages=[{"role": "user", "content": "hi"}],
                    state={"events": []},
                    generator=MagicMock(),
                    include_generation_metadata=True,
                ):
                    chunks.append(chunk)

                mock_log.warning.assert_called_once()
                assert "ignoring unsupported kwargs" in mock_log.warning.call_args[0][0]

            # Only the supported kwargs should be passed
            guardrails.rails_engine.stream_async.assert_called_once_with(
                messages=[{"role": "user", "content": "hi"}],
                options=None,
                include_metadata=False,
            )

    @pytest.mark.asyncio
    @patch.object(IORails, "stop", new_callable=AsyncMock)
    @patch.object(IORails, "start", new_callable=AsyncMock)
    @patch.object(IORails, "__init__", return_value=None)
    async def test_prompt_converted_to_messages(self, mock_init, mock_start, mock_stop, _content_safety_rails_config):
        """A string prompt is converted to messages before reaching IORails."""

        async def mock_stream():
            yield "ok"

        async with Guardrails(config=_content_safety_rails_config, use_iorails=True) as guardrails:
            assert isinstance(guardrails.rails_engine, IORails)
            guardrails.rails_engine.stream_async = MagicMock(return_value=mock_stream())

            chunks = []
            async for chunk in guardrails.stream_async(prompt="hello"):
                chunks.append(chunk)

            guardrails.rails_engine.stream_async.assert_called_once_with(
                messages=[{"role": "user", "content": "hello"}],
                options=None,
                include_metadata=False,
            )


class TestLLMRailsOnlyMethods:
    """Tests for methods that exist on LLMRails but not IORails.

    Under LLMRails, each method must delegate to the underlying instance.
    Under IORails, each must raise NotImplementedError.
    """

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_generate_events_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.generate_events.return_value = [{"type": "BotMessage"}]
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        events = [{"type": "UtteranceUserActionFinished", "final_transcript": "hi"}]
        result = guardrails.generate_events(events)

        mock_llmrails_instance.generate_events.assert_called_once_with(events)
        assert result == [{"type": "BotMessage"}]

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_generate_events_async_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.generate_events_async = AsyncMock(return_value=[{"type": "BotMessage"}])
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        events = [{"type": "UtteranceUserActionFinished", "final_transcript": "hi"}]
        result = await guardrails.generate_events_async(events)

        mock_llmrails_instance.generate_events_async.assert_called_once_with(events)
        assert result == [{"type": "BotMessage"}]

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_process_events_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.process_events.return_value = ([{"type": "BotMessage"}], {})
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        events = [{"type": "UtteranceUserActionFinished"}]
        result = guardrails.process_events(events, state={"foo": "bar"}, blocking=True)

        mock_llmrails_instance.process_events.assert_called_once_with(events, {"foo": "bar"}, True)
        assert result == ([{"type": "BotMessage"}], {})

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_process_events_async_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.process_events_async = AsyncMock(return_value=([{"type": "BotMessage"}], {}))
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        events = [{"type": "UtteranceUserActionFinished"}]
        result = await guardrails.process_events_async(events, state={"foo": "bar"}, blocking=True)

        mock_llmrails_instance.process_events_async.assert_called_once_with(events, {"foo": "bar"}, True)
        assert result == ([{"type": "BotMessage"}], {})

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_check_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        from nemoguardrails.rails.llm.options import RailsResult, RailStatus, RailType

        sentinel = RailsResult(status=RailStatus.PASSED, content="ok")
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.check.return_value = sentinel
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        messages = [{"role": "user", "content": "hi"}]
        result = guardrails.check(messages, rail_types=[RailType.INPUT])

        mock_llmrails_instance.check.assert_called_once_with(messages, rail_types=[RailType.INPUT])
        assert result is sentinel

    @pytest.mark.asyncio
    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    async def test_check_async_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        from nemoguardrails.rails.llm.options import RailsResult, RailStatus, RailType

        sentinel = RailsResult(status=RailStatus.PASSED, content="ok")
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.check_async = AsyncMock(return_value=sentinel)
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        messages = [{"role": "user", "content": "hi"}]
        result = await guardrails.check_async(messages, rail_types=[RailType.OUTPUT])

        mock_llmrails_instance.check_async.assert_called_once_with(messages, rail_types=[RailType.OUTPUT])
        assert result is sentinel

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_register_action_delegates_and_returns_self(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.register_action.return_value = mock_llmrails_instance
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

        def my_action():
            pass

        result = guardrails.register_action(my_action, name="my_action")

        mock_llmrails_instance.register_action.assert_called_once_with(my_action, "my_action")
        assert result is guardrails  # Returns Guardrails facade for chaining

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_register_action_param_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.register_action_param.return_value = mock_llmrails_instance
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        result = guardrails.register_action_param("my_param", 42)

        mock_llmrails_instance.register_action_param.assert_called_once_with("my_param", 42)
        assert result is guardrails

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_register_filter_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.register_filter.return_value = mock_llmrails_instance
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

        def my_filter(x):
            return x

        result = guardrails.register_filter(my_filter, name="my_filter")

        mock_llmrails_instance.register_filter.assert_called_once_with(my_filter, "my_filter")
        assert result is guardrails

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_register_output_parser_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.register_output_parser.return_value = mock_llmrails_instance
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

        def my_parser(s):
            return s

        result = guardrails.register_output_parser(my_parser, "my_parser")

        mock_llmrails_instance.register_output_parser.assert_called_once_with(my_parser, "my_parser")
        assert result is guardrails

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_register_prompt_context_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.register_prompt_context.return_value = mock_llmrails_instance
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        result = guardrails.register_prompt_context("user_name", "alice")

        mock_llmrails_instance.register_prompt_context.assert_called_once_with("user_name", "alice")
        assert result is guardrails

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_register_embedding_search_provider_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        from nemoguardrails.embeddings.index import EmbeddingsIndex

        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.register_embedding_search_provider.return_value = mock_llmrails_instance
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

        class FakeIndex(EmbeddingsIndex):
            pass

        result = guardrails.register_embedding_search_provider("fake", FakeIndex)

        mock_llmrails_instance.register_embedding_search_provider.assert_called_once_with("fake", FakeIndex)
        assert result is guardrails

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_register_embedding_provider_delegates(self, mock_llmrails_class, _nemoguards_rails_config):
        from nemoguardrails.embeddings.providers.base import EmbeddingModel

        mock_llmrails_instance = MagicMock()
        mock_llmrails_instance.register_embedding_provider.return_value = mock_llmrails_instance
        mock_llmrails_class.return_value = mock_llmrails_instance

        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)

        class FakeModel(EmbeddingModel):
            engine_name = "fake"
            model = "fake"

            async def encode_async(self, documents):
                return []

            def encode(self, documents):
                return []

        result = guardrails.register_embedding_provider(FakeModel, name="fake")

        mock_llmrails_instance.register_embedding_provider.assert_called_once_with(FakeModel, "fake")
        assert result is guardrails

    @pytest.mark.parametrize(
        "method_name,args,is_async",
        [
            ("generate_events", ([],), False),
            ("generate_events_async", ([],), True),
            ("process_events", ([],), False),
            ("process_events_async", ([],), True),
            ("check", ([{"role": "user", "content": "hi"}],), False),
            ("check_async", ([{"role": "user", "content": "hi"}],), True),
            ("register_action", (lambda: None,), False),
            ("register_action_param", ("p", 1), False),
            ("register_filter", (lambda x: x,), False),
            ("register_output_parser", (lambda x: x, "p"), False),
            ("register_prompt_context", ("n", "v"), False),
            ("register_embedding_search_provider", ("n", type("X", (), {})), False),
            ("register_embedding_provider", (type("X", (), {}),), False),
        ],
    )
    @pytest.mark.asyncio
    @patch.object(IORails, "__init__", return_value=None)
    async def test_iorails_raises_not_implemented(
        self, mock_iorails_init, _content_safety_rails_config, method_name, args, is_async
    ):
        """Every LLMRails-only method must raise NotImplementedError under IORails."""
        guardrails = Guardrails(config=_content_safety_rails_config, use_iorails=True)
        assert isinstance(guardrails.rails_engine, IORails)

        method = getattr(guardrails, method_name)
        with pytest.raises(NotImplementedError, match=f"IORails doesn't support {method_name}"):
            if is_async:
                await method(*args)
            else:
                method(*args)


class TestGuardrailsPickle:
    """Tests for __getstate__ / __setstate__ pickle support on Guardrails."""

    @patch("nemoguardrails.guardrails.guardrails.LLMRails")
    def test_getstate_preserves_config_and_use_iorails(self, mock_llmrails_class, _nemoguards_rails_config):
        """__getstate__ preserves config, verbose, and use_iorails so the rebuilt
        instance lands on the same engine after a pickle round-trip."""
        mock_llmrails_class.return_value = MagicMock()
        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=False)
        state = guardrails.__getstate__()
        assert state == {"config": _nemoguards_rails_config, "verbose": False, "use_iorails": False}

    @patch.object(LLMRails, "__init__", return_value=None)
    def test_setstate_preserves_llmrails_on_iorails_compatible_config(
        self, mock_llmrails_init, _nemoguards_rails_config
    ):
        """Regression: a Guardrails(use_iorails=False) wrapper on an IORails-compatible
        config must rebuild as LLMRails, not silently switch to IORails. Without
        preserving use_iorails in pickle state, __setstate__ would default to True
        and route to IORails, making all LLMRails-only methods raise NotImplementedError.

        We patch LLMRails.__init__ (not the class itself) to keep class identity intact
        so isinstance() works against the real LLMRails.
        """
        guardrails = Guardrails.__new__(Guardrails)
        guardrails.__setstate__({"config": _nemoguards_rails_config, "use_iorails": False})

        assert guardrails.config is _nemoguards_rails_config
        assert isinstance(guardrails.rails_engine, LLMRails)
        mock_llmrails_init.assert_called_once_with(_nemoguards_rails_config, None, False)

    @patch.object(IORails, "__init__", return_value=None)
    def test_setstate_backwards_compat_old_pickle_without_use_iorails(
        self, mock_iorails_init, _nemoguards_rails_config
    ):
        """Older pickles (pre-fix) only serialized {"config": ...}. __setstate__ must
        still accept them, defaulting use_iorails to True."""
        guardrails = Guardrails.__new__(Guardrails)
        guardrails.__setstate__({"config": _nemoguards_rails_config})  # no use_iorails key

        assert guardrails.use_iorails_engine is True
        assert isinstance(guardrails.rails_engine, IORails)

    @patch.object(IORails, "__init__", return_value=None)
    def test_setstate_rebuilds_from_in_memory_config_iorails(self, mock_iorails_init, _nemoguards_rails_config):
        """__setstate__ uses the pickled config directly when config_path is unset
        (in-memory configs from RailsConfig.from_content). NEMOGUARDS_CONFIG is
        IORails-compatible and __setstate__ uses the default use_iorails=True, so
        the rebuilt wrapper lands on IORails."""
        guardrails = Guardrails.__new__(Guardrails)
        guardrails.__setstate__({"config": _nemoguards_rails_config})

        assert guardrails.config is _nemoguards_rails_config
        assert guardrails.verbose is False
        # __init__ runs and routes to IORails for this config
        assert isinstance(guardrails.rails_engine, IORails)
        mock_iorails_init.assert_called_once_with(_nemoguards_rails_config)

    @patch.object(LLMRails, "__init__", return_value=None)
    def test_setstate_rebuilds_from_in_memory_config_llmrails(self, mock_llmrails_init):
        """When the config has flows not supported by IORails, __setstate__ rebuilds
        the wrapper onto LLMRails (the fallback engine)."""
        llmrails_only_config = _make_iorails_config(
            rails={
                "input": {"flows": ["self check input"]},
                "output": {"flows": ["content safety check output $model=content_safety"]},
            },
            extra_prompts=[{"task": "self_check_input", "content": "placeholder"}],
        )

        guardrails = Guardrails.__new__(Guardrails)
        guardrails.__setstate__({"config": llmrails_only_config})

        assert guardrails.config is llmrails_only_config
        assert guardrails.verbose is False
        # 'self check input' is not in IORAILS_INPUT_FLOWS, so the wrapper falls back to LLMRails
        assert isinstance(guardrails.rails_engine, LLMRails)
        mock_llmrails_init.assert_called_once_with(llmrails_only_config, None, False)

    @patch.object(IORails, "__init__", return_value=None)
    @patch("nemoguardrails.guardrails.guardrails.RailsConfig.from_path")
    def test_setstate_reloads_from_path_when_config_path_set(
        self, mock_from_path, mock_iorails_init, _nemoguards_rails_config
    ):
        """When the pickled config has a config_path, __setstate__ reloads it
        from disk (picks up any on-disk changes since the pickle was written)
        rather than using the in-memory snapshot."""
        # Pickled config carries only the on-disk location.
        pickled_config = MagicMock()
        pickled_config.config_path = "/some/path/to/config"

        # Reload returns the fresh in-memory config.
        mock_from_path.return_value = _nemoguards_rails_config

        guardrails = Guardrails.__new__(Guardrails)
        guardrails.__setstate__({"config": pickled_config, "use_iorails": True})

        mock_from_path.assert_called_once_with("/some/path/to/config")
        assert guardrails.config is _nemoguards_rails_config
        assert isinstance(guardrails.rails_engine, IORails)

    @patch.object(IORails, "__init__", return_value=None)
    def test_pickle_preserves_iorails_round_trip(self, mock_iorails_init, _nemoguards_rails_config):
        """Full round-trip on the only permutation that produces IORails:
        Guardrails(use_iorails=True) without an llm on an IORails-compatible config.
        Verifies (1) the IORails branch of __getstate__ saves use_iorails=True,
        and (2) __setstate__ rebuilds onto IORails. This is the symmetric counterpart
        to test_pickle_preserves_llmrails_when_llm_was_passed."""
        guardrails = Guardrails(config=_nemoguards_rails_config, use_iorails=True)
        assert isinstance(guardrails.rails_engine, IORails)

        state = guardrails.__getstate__()
        assert state["use_iorails"] is True

        restored = Guardrails.__new__(Guardrails)
        restored.__setstate__(state)
        assert isinstance(restored.rails_engine, IORails)
        # Called twice: once during initial Guardrails(...), once during __setstate__
        assert mock_iorails_init.call_count == 2

    @patch.object(LLMRails, "__init__", return_value=None)
    def test_pickle_preserves_llmrails_when_llm_was_passed(
        self, mock_llmrails_init, _content_safety_rails_config, mock_llm
    ):
        """Regression (CodeRabbit P0): when an explicit LLM is passed, Guardrails uses
        LLMRails even with use_iorails=True and an IORails-compatible config (the llm
        argument forces LLMRails). Pickle drops the llm — so __getstate__ must save the
        *effective* engine choice (not the user kwarg), otherwise __setstate__ would
        rebuild with llm=None + use_iorails=True and silently switch to IORails."""
        # Initial wrapper: LLMRails despite use_iorails=True (because llm was passed)
        guardrails = Guardrails(config=_content_safety_rails_config, llm=mock_llm, use_iorails=True)
        assert isinstance(guardrails.rails_engine, LLMRails)

        # __getstate__ saves effective engine (False = LLMRails), not the user kwarg (True)
        state = guardrails.__getstate__()
        assert state["use_iorails"] is False

        # __setstate__ rebuilds onto LLMRails — engine choice survives the round-trip
        restored = Guardrails.__new__(Guardrails)
        restored.__setstate__(state)
        assert isinstance(restored.rails_engine, LLMRails)
        # Called twice: once during initial Guardrails(...), once during __setstate__
        assert mock_llmrails_init.call_count == 2

    @patch.object(IORails, "__init__", return_value=None)
    def test_getstate_preserves_verbose_true(self, mock_iorails_init, _nemoguards_rails_config):
        """__getstate__ captures verbose=True so a verbose Guardrails round-trips
        with logging configuration intact."""
        guardrails = Guardrails(config=_nemoguards_rails_config, verbose=True)
        state = guardrails.__getstate__()
        assert state["verbose"] is True

    @patch.object(IORails, "__init__", return_value=None)
    def test_setstate_restores_verbose_true(self, mock_iorails_init, _nemoguards_rails_config):
        """__setstate__ restores verbose=True so the rebuilt instance still has
        verbose logging active (rather than silently dropping back to False)."""
        guardrails = Guardrails.__new__(Guardrails)
        guardrails.__setstate__({"config": _nemoguards_rails_config, "verbose": True, "use_iorails": True})
        assert guardrails.verbose is True

    @patch.object(IORails, "__init__", return_value=None)
    def test_pickle_round_trip_preserves_verbose(self, mock_iorails_init, _nemoguards_rails_config):
        """Full round-trip: a Guardrails constructed with verbose=True must come
        back from __getstate__/__setstate__ with verbose=True. Regression for the
        bug where verbose was hardcoded to False on restore, silently obscuring
        debugging sessions for users who pickled a verbose wrapper."""
        guardrails = Guardrails(config=_nemoguards_rails_config, verbose=True)
        assert guardrails.verbose is True

        state = guardrails.__getstate__()
        restored = Guardrails.__new__(Guardrails)
        restored.__setstate__(state)
        assert restored.verbose is True

    @patch.object(IORails, "__init__", return_value=None)
    def test_setstate_backwards_compat_old_pickle_without_verbose(self, mock_iorails_init, _nemoguards_rails_config):
        """Older pickles (pre-fix) didn't serialize verbose. __setstate__ must
        still accept them, defaulting verbose to False."""
        guardrails = Guardrails.__new__(Guardrails)
        guardrails.__setstate__({"config": _nemoguards_rails_config, "use_iorails": True})
        assert guardrails.verbose is False
