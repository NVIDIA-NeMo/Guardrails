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

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemoguardrails.cli import app
from nemoguardrails.cli.providers import (
    _get_provider_completions,
    _list_providers,
    find_providers,
    select_provider,
    select_provider_type,
    select_provider_with_type,
)

runner = CliRunner()


class TestListProviders:
    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.get_chat_provider_names")
    @patch("nemoguardrails.cli.providers.get_llm_provider_names")
    def test_list_providers(
        self, mock_llm_providers, mock_chat_providers, mock_console
    ):
        mock_llm_providers.return_value = ["llm_provider_1", "llm_provider_2"]
        mock_chat_providers.return_value = ["chat_provider_1", "chat_provider_2"]

        _list_providers()

        assert mock_console.print.call_count >= 4
        mock_console.print.assert_any_call("\n[bold]Text Completion Providers:[/]")
        mock_console.print.assert_any_call("  • llm_provider_1")
        mock_console.print.assert_any_call("  • llm_provider_2")
        mock_console.print.assert_any_call("\n[bold]Chat Completion Providers:[/]")
        mock_console.print.assert_any_call("  • chat_provider_1")
        mock_console.print.assert_any_call("  • chat_provider_2")


class TestGetProviderCompletions:
    @patch("nemoguardrails.cli.providers.get_llm_provider_names")
    def test_get_text_completion_providers(self, mock_llm_providers):
        mock_llm_providers.return_value = ["provider_b", "provider_a"]

        result = _get_provider_completions("text completion")

        assert result == ["provider_a", "provider_b"]
        mock_llm_providers.assert_called_once()

    @patch("nemoguardrails.cli.providers.get_chat_provider_names")
    def test_get_chat_completion_providers(self, mock_chat_providers):
        mock_chat_providers.return_value = ["chat_b", "chat_a"]

        result = _get_provider_completions("chat completion")

        assert result == ["chat_a", "chat_b"]
        mock_chat_providers.assert_called_once()

    def test_get_providers_invalid_type(self):
        result = _get_provider_completions("invalid_type")
        assert result == []

    def test_get_providers_no_type(self):
        result = _get_provider_completions(None)
        assert result == []


class TestSelectProviderType:
    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    def test_select_provider_type_exact_match(self, mock_prompt_session, mock_console):
        mock_session = MagicMock()
        mock_session.prompt.return_value = "chat completion"
        mock_prompt_session.return_value = mock_session

        result = select_provider_type()

        assert result == "chat completion"
        mock_session.prompt.assert_called_once()

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    def test_select_provider_type_fuzzy_match(self, mock_prompt_session, mock_console):
        mock_session = MagicMock()
        mock_session.prompt.return_value = "chat"
        mock_prompt_session.return_value = mock_session

        result = select_provider_type()

        assert result == "chat completion"

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    def test_select_provider_type_empty_input(self, mock_prompt_session, mock_console):
        mock_session = MagicMock()
        mock_session.prompt.return_value = ""
        mock_prompt_session.return_value = mock_session

        result = select_provider_type()

        assert result is None

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    def test_select_provider_type_ambiguous_match(
        self, mock_prompt_session, mock_console
    ):
        mock_session = MagicMock()
        mock_session.prompt.return_value = "completion"
        mock_prompt_session.return_value = mock_session

        result = select_provider_type()

        assert result is None

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    def test_select_provider_type_keyboard_interrupt(
        self, mock_prompt_session, mock_console
    ):
        mock_session = MagicMock()
        mock_session.prompt.side_effect = KeyboardInterrupt()
        mock_prompt_session.return_value = mock_session

        result = select_provider_type()

        assert result is None

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    def test_select_provider_type_eof_error(self, mock_prompt_session, mock_console):
        mock_session = MagicMock()
        mock_session.prompt.side_effect = EOFError()
        mock_prompt_session.return_value = mock_session

        result = select_provider_type()

        assert result is None


class TestSelectProvider:
    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    @patch("nemoguardrails.cli.providers._get_provider_completions")
    def test_select_provider_exact_match(
        self, mock_get_completions, mock_prompt_session, mock_console
    ):
        mock_get_completions.return_value = ["openai", "anthropic", "azure"]
        mock_session = MagicMock()
        mock_session.prompt.return_value = "openai"
        mock_prompt_session.return_value = mock_session

        result = select_provider("chat completion")

        assert result == "openai"

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    @patch("nemoguardrails.cli.providers._get_provider_completions")
    def test_select_provider_fuzzy_match(
        self, mock_get_completions, mock_prompt_session, mock_console
    ):
        mock_get_completions.return_value = ["openai", "anthropic", "azure"]
        mock_session = MagicMock()
        mock_session.prompt.return_value = "open"
        mock_prompt_session.return_value = mock_session

        result = select_provider("chat completion")

        assert result == "openai"

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    @patch("nemoguardrails.cli.providers._get_provider_completions")
    def test_select_provider_empty_input(
        self, mock_get_completions, mock_prompt_session, mock_console
    ):
        mock_get_completions.return_value = ["openai", "anthropic"]
        mock_session = MagicMock()
        mock_session.prompt.return_value = ""
        mock_prompt_session.return_value = mock_session

        result = select_provider("chat completion")

        assert result is None

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    @patch("nemoguardrails.cli.providers._get_provider_completions")
    def test_select_provider_no_match(
        self, mock_get_completions, mock_prompt_session, mock_console
    ):
        mock_get_completions.return_value = ["openai", "anthropic"]
        mock_session = MagicMock()
        mock_session.prompt.return_value = "invalid_provider"
        mock_prompt_session.return_value = mock_session

        result = select_provider("chat completion")

        assert result is None

    @patch("nemoguardrails.cli.providers.console")
    @patch("nemoguardrails.cli.providers.PromptSession")
    @patch("nemoguardrails.cli.providers._get_provider_completions")
    def test_select_provider_keyboard_interrupt(
        self, mock_get_completions, mock_prompt_session, mock_console
    ):
        mock_get_completions.return_value = ["openai"]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = KeyboardInterrupt()
        mock_prompt_session.return_value = mock_session

        result = select_provider("chat completion")

        assert result is None


class TestSelectProviderWithType:
    @patch("nemoguardrails.cli.providers.select_provider")
    @patch("nemoguardrails.cli.providers.select_provider_type")
    def test_select_provider_with_type_success(
        self, mock_select_type, mock_select_provider
    ):
        mock_select_type.return_value = "chat completion"
        mock_select_provider.return_value = "openai"

        result = select_provider_with_type()

        assert result == ("chat completion", "openai")
        mock_select_type.assert_called_once()
        mock_select_provider.assert_called_once_with("chat completion")

    @patch("nemoguardrails.cli.providers.select_provider_type")
    def test_select_provider_with_type_no_type(self, mock_select_type):
        mock_select_type.return_value = None

        result = select_provider_with_type()

        assert result is None

    @patch("nemoguardrails.cli.providers.select_provider")
    @patch("nemoguardrails.cli.providers.select_provider_type")
    def test_select_provider_with_type_no_provider(
        self, mock_select_type, mock_select_provider
    ):
        mock_select_type.return_value = "chat completion"
        mock_select_provider.return_value = None

        result = select_provider_with_type()

        assert result is None


class TestFindProvidersCommand:
    @patch("nemoguardrails.cli.providers._list_providers")
    def test_find_providers_list_only_as_function(self, mock_list_providers):
        find_providers(list_only=True)
        mock_list_providers.assert_called_once()

    @patch("nemoguardrails.cli.providers.select_provider_with_type")
    @patch("typer.echo")
    def test_find_providers_interactive_success(self, mock_echo, mock_select):
        mock_select.return_value = ("chat completion", "openai")
        find_providers(list_only=False)
        mock_echo.assert_called_with("\nSelected chat completion provider: openai")

    @patch("nemoguardrails.cli.providers.select_provider_with_type")
    @patch("typer.echo")
    def test_find_providers_interactive_no_selection(self, mock_echo, mock_select):
        mock_select.return_value = None
        find_providers(list_only=False)
        mock_echo.assert_called_with("No provider selected.")

    def test_find_providers_cli_list(self):
        with patch("nemoguardrails.cli._list_providers") as mock_list:
            result = runner.invoke(app, ["find-providers", "--list"])
            assert result.exit_code == 0
            mock_list.assert_called_once()

    def test_find_providers_cli_interactive(self):
        with patch("nemoguardrails.cli.select_provider_with_type") as mock_select:
            mock_select.return_value = ("text completion", "llama")
            result = runner.invoke(app, ["find-providers"])
            assert result.exit_code == 0
            assert "Selected text completion provider: llama" in result.stdout
