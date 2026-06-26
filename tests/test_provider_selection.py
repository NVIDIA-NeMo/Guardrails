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

from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.cli.providers import (
    _get_provider_completions,
    _list_providers,
    find_providers,
    select_provider,
)


def test_list_providers(capsys):
    """Test listing all providers."""
    _list_providers()
    captured = capsys.readouterr()
    assert "Chat Completion Providers:" in captured.out


def test_get_provider_completions():
    """Test getting provider completions."""
    chat_providers = _get_provider_completions()
    assert isinstance(chat_providers, list)
    assert len(chat_providers) > 0


def test_providers_list_only(capsys):
    """Test providers command with list only option."""

    find_providers(list_only=True)
    captured = capsys.readouterr()
    assert "Chat Completion Providers:" in captured.out


@patch("nemoguardrails.cli.providers.PromptSession")
@pytest.mark.skip(reason="Skipping test temporarily breaks in python 3.9 and 3.10")
def test_select_provider(mock_session):
    """Test selecting a chat-completion provider."""
    session_instance = MagicMock()
    mock_session.return_value = session_instance

    # empty input
    session_instance.prompt.return_value = ""
    assert select_provider() is None

    # keyboard interrupt
    session_instance.prompt.side_effect = KeyboardInterrupt
    assert select_provider() is None
