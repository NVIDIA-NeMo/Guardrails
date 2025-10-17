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

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.cli.chat import (
    ChatState,
    extract_scene_text_content,
    parse_events_inputs,
    run_chat,
)

chat_module = sys.modules["nemoguardrails.cli.chat"]


class TestParseEventsInputs:
    def test_parse_simple_event(self):
        result = parse_events_inputs("UserAction")
        assert result == {"type": "UserAction"}

    def test_parse_event_with_params(self):
        result = parse_events_inputs('UserAction(name="test", value=123)')
        assert result == {"type": "UserAction", "name": "test", "value": 123}

    def test_parse_event_with_string_params(self):
        result = parse_events_inputs('UserAction(message="hello world")')
        assert result == {"type": "UserAction", "message": "hello world"}

    def test_parse_nested_event(self):
        result = parse_events_inputs("bot.UtteranceAction")
        assert result == {"type": "botUtteranceAction"}

    def test_parse_event_with_nested_params(self):
        result = parse_events_inputs('UserAction(data={"key": "value"})')
        assert result == {"type": "UserAction", "data": {"key": "value"}}

    def test_parse_event_with_list_params(self):
        result = parse_events_inputs("UserAction(items=[1, 2, 3])")
        assert result == {"type": "UserAction", "items": [1, 2, 3]}

    def test_parse_invalid_event(self):
        result = parse_events_inputs("Invalid.Event.Format.TooMany")
        assert result is None

    def test_parse_event_missing_equals(self):
        result = parse_events_inputs("UserAction(invalid_param)")
        assert result is None


class TestExtractSceneTextContent:
    def test_extract_empty_list(self):
        result = extract_scene_text_content([])
        assert result == ""

    def test_extract_single_text(self):
        content = [{"text": "Hello World"}]
        result = extract_scene_text_content(content)
        assert result == "\nHello World"

    def test_extract_multiple_texts(self):
        content = [{"text": "Line 1"}, {"text": "Line 2"}, {"text": "Line 3"}]
        result = extract_scene_text_content(content)
        assert result == "\nLine 1\nLine 2\nLine 3"

    def test_extract_mixed_content(self):
        content = [
            {"text": "Text 1"},
            {"image": "image.png"},
            {"text": "Text 2"},
            {"button": "Click Me"},
        ]
        result = extract_scene_text_content(content)
        assert result == "\nText 1\nText 2"

    def test_extract_no_text_content(self):
        content = [{"image": "image.png"}, {"button": "Click Me"}]
        result = extract_scene_text_content(content)
        assert result == ""


class TestChatState:
    def test_initial_state(self):
        chat_state = ChatState()
        assert chat_state.state is None
        assert chat_state.waiting_user_input is False
        assert chat_state.paused is False
        assert chat_state.running_timer_tasks == {}
        assert chat_state.input_events == []
        assert chat_state.output_events == []
        assert chat_state.output_state is None
        assert chat_state.events_counter == 0
        assert chat_state.first_time is False


class TestRunChat:
    def test_run_chat_v1_0(self):
        with patch.object(
            chat_module, "RailsConfig"
        ) as mock_rails_config, patch.object(
            chat_module, "LLMRails"
        ) as mock_llm_rails, patch(
            "asyncio.run"
        ) as mock_asyncio_run:
            mock_config = MagicMock()
            mock_config.colang_version = "1.0"
            mock_rails_config.from_path.return_value = mock_config

            run_chat(config_path="test_config")

            mock_rails_config.from_path.assert_called_once_with("test_config")
            mock_asyncio_run.assert_called_once()

    def test_run_chat_v2_x(self):
        with patch.object(
            chat_module, "RailsConfig"
        ) as mock_rails_config, patch.object(
            chat_module, "LLMRails"
        ) as mock_llm_rails, patch.object(
            chat_module, "get_or_create_event_loop"
        ) as mock_get_loop:
            mock_config = MagicMock()
            mock_config.colang_version = "2.x"
            mock_rails_config.from_path.return_value = mock_config

            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            run_chat(config_path="test_config")

            mock_rails_config.from_path.assert_called_once_with("test_config")
            mock_llm_rails.assert_called_once_with(mock_config, verbose=False)
            mock_loop.run_until_complete.assert_called_once()

    def test_run_chat_invalid_version(self):
        with patch.object(chat_module, "RailsConfig") as mock_rails_config:
            mock_config = MagicMock()
            mock_config.colang_version = "3.0"
            mock_rails_config.from_path.return_value = mock_config

            with pytest.raises(Exception, match="Invalid colang version"):
                run_chat(config_path="test_config")

    def test_run_chat_verbose_with_llm_calls(self):
        with patch.object(chat_module, "RailsConfig") as mock_rails_config, patch(
            "asyncio.run"
        ) as mock_asyncio_run, patch.object(chat_module, "console") as mock_console:
            mock_config = MagicMock()
            mock_config.colang_version = "1.0"
            mock_rails_config.from_path.return_value = mock_config

            run_chat(config_path="test_config", verbose=True, verbose_llm_calls=True)

            mock_console.print.assert_any_call(
                "NOTE: use the `--verbose-no-llm` option to exclude the LLM prompts "
                "and completions from the log.\n"
            )


class TestRunChatV1Async:
    @pytest.mark.asyncio
    async def test_run_chat_v1_no_config_no_server(self):
        from nemoguardrails.cli.chat import _run_chat_v1_0

        with pytest.raises(RuntimeError, match="At least one of"):
            await _run_chat_v1_0(config_path=None, server_url=None)

    @pytest.mark.asyncio
    @patch("builtins.input")
    @patch.object(chat_module, "LLMRails")
    @patch.object(chat_module, "RailsConfig")
    async def test_run_chat_v1_local_config(
        self, mock_rails_config, mock_llm_rails, mock_input
    ):
        from nemoguardrails.cli.chat import _run_chat_v1_0

        mock_config = MagicMock()
        mock_config.streaming_supported = False
        mock_rails_config.from_path.return_value = mock_config

        mock_rails = AsyncMock()
        mock_rails.generate_async = AsyncMock(
            return_value={"role": "assistant", "content": "Hello!"}
        )
        mock_rails.main_llm_supports_streaming = False
        mock_llm_rails.return_value = mock_rails

        mock_input.side_effect = ["test message", KeyboardInterrupt()]

        try:
            await _run_chat_v1_0(config_path="test_config")
        except KeyboardInterrupt:
            pass

        mock_rails.generate_async.assert_called_once()

    @pytest.mark.asyncio
    @patch("builtins.input")
    @patch.object(chat_module, "console")
    @patch.object(chat_module, "LLMRails")
    @patch.object(chat_module, "RailsConfig")
    async def test_run_chat_v1_streaming_not_supported(
        self, mock_rails_config, mock_llm_rails, mock_console, mock_input
    ):
        from nemoguardrails.cli.chat import _run_chat_v1_0

        mock_config = MagicMock()
        mock_config.streaming_supported = False
        mock_rails_config.from_path.return_value = mock_config

        mock_rails = AsyncMock()
        mock_llm_rails.return_value = mock_rails

        mock_input.side_effect = [KeyboardInterrupt()]

        try:
            await _run_chat_v1_0(config_path="test_config", streaming=True)
        except KeyboardInterrupt:
            pass

        mock_console.print.assert_any_call(
            "WARNING: The config `test_config` does not support streaming. "
            "Falling back to normal mode."
        )

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    @patch("builtins.input")
    async def test_run_chat_v1_server_mode(self, mock_input, mock_client_session):
        from nemoguardrails.cli.chat import _run_chat_v1_0

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.headers = {}
        mock_response.json = AsyncMock(
            return_value={
                "messages": [{"role": "assistant", "content": "Server response"}]
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_post_context = AsyncMock()
        mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_context.__aexit__ = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_context)

        mock_client_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session.return_value.__aexit__ = AsyncMock()

        mock_input.side_effect = ["test message", KeyboardInterrupt()]

        try:
            await _run_chat_v1_0(
                server_url="http://localhost:8000", config_id="test_id"
            )
        except KeyboardInterrupt:
            pass

        assert mock_session.post.called
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:8000/v1/chat/completions"
        assert "config_id" in call_args[1]["json"]
        assert call_args[1]["json"]["config_id"] == "test_id"
        assert call_args[1]["json"]["stream"] is False

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    @patch("builtins.input")
    async def test_run_chat_v1_server_streaming(self, mock_input, mock_client_session):
        from nemoguardrails.cli.chat import _run_chat_v1_0

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.headers = {"Transfer-Encoding": "chunked"}

        async def mock_iter_any():
            yield b"Stream "
            yield b"response"

        mock_response.content.iter_any = mock_iter_any
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_post_context = AsyncMock()
        mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_context.__aexit__ = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_context)

        mock_client_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session.return_value.__aexit__ = AsyncMock()

        mock_input.side_effect = ["test message", KeyboardInterrupt()]

        try:
            await _run_chat_v1_0(
                server_url="http://localhost:8000", config_id="test_id", streaming=True
            )
        except KeyboardInterrupt:
            pass

        assert mock_session.post.called
