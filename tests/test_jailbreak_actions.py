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

from unittest import mock

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.llm.taskmanager import LLMTaskManager


class TestJailbreakDetectionActions:
    """Test suite for jailbreak detection actions with comprehensive coverage of PR changes."""

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_with_nim_base_url(self, monkeypatch):
        """Test jailbreak_detection_model action with nim_base_url config."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_nim_request = mock.AsyncMock(return_value=True)
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.actions.jailbreak_nim_request",
            mock_nim_request,
        )

        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection:
                  nim_base_url: "http://localhost:8000/v1"
                  nim_server_endpoint: "classify"
                  api_key_env_var: "TEST_API_KEY"
            """,
        )

        monkeypatch.setenv("TEST_API_KEY", "test_token_123")

        llm_task_manager = LLMTaskManager(config=config)
        context = {"user_message": "test prompt"}

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is True

        mock_nim_request.assert_called_once_with(
            prompt="test prompt",
            nim_url="http://localhost:8000/v1",
            nim_auth_token="test_token_123",
            nim_classification_path="classify",
        )

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_api_key_not_set(self, monkeypatch, caplog):
        """Test warning when api_key_env_var is configured but environment variable is not set."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_nim_request = mock.AsyncMock(return_value=False)
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.actions.jailbreak_nim_request",
            mock_nim_request,
        )

        # create config with api_key_env_var but don't set the environment variable
        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection:
                  nim_base_url: "http://localhost:8000/v1"
                  api_key_env_var: "MISSING_API_KEY"
            """,
        )

        # ensure env var is not set
        monkeypatch.delenv("MISSING_API_KEY", raising=False)

        llm_task_manager = LLMTaskManager(config=config)
        context = {"user_message": "test prompt"}

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is False

        # verify warning was logged
        assert (
            "api_key_env var at MISSING_API_KEY but the environment variable was not set"
            in caplog.text
        )

        # verify nim request was called with None token
        mock_nim_request.assert_called_once_with(
            prompt="test prompt",
            nim_url="http://localhost:8000/v1",
            nim_auth_token=None,
            nim_classification_path="classify",
        )

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_no_api_key_env_var(self, monkeypatch):
        """Test that None token is used when api_key_env_var is not configured."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_nim_request = mock.AsyncMock(return_value=False)
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.actions.jailbreak_nim_request",
            mock_nim_request,
        )

        # create config without api_key_env_var
        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection:
                  nim_base_url: "http://localhost:8000/v1"
            """,
        )

        llm_task_manager = LLMTaskManager(config=config)
        context = {"user_message": "test prompt"}

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is False

        mock_nim_request.assert_called_once_with(
            prompt="test prompt",
            nim_url="http://localhost:8000/v1",
            nim_auth_token=None,
            nim_classification_path="classify",
        )

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_local_runtime_error(
        self, monkeypatch, caplog
    ):
        """Test RuntimeError handling when local model is not available."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_check_jailbreak = mock.MagicMock(
            side_effect=RuntimeError("No classifier available")
        )
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.model_based.checks.check_jailbreak",
            mock_check_jailbreak,
        )

        # create config with no endpoints (forces local mode)
        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection: {}
            """,
        )

        llm_task_manager = LLMTaskManager(config=config)
        context = {"user_message": "test prompt"}

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is False

        assert "Jailbreak detection model not available" in caplog.text
        assert "No classifier available" in caplog.text

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_local_import_error(
        self, monkeypatch, caplog
    ):
        """Test ImportError handling when dependencies are missing."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        # mock check_jailbreak to raise ImportError
        mock_check_jailbreak = mock.MagicMock(
            side_effect=ImportError("No module named 'sklearn'")
        )
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.model_based.checks.check_jailbreak",
            mock_check_jailbreak,
        )

        # create config with no endpoints (forces local mode)
        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection: {}
            """,
        )

        llm_task_manager = LLMTaskManager(config=config)
        context = {"user_message": "test prompt"}

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is False

        assert "Failed to import required dependencies for local model" in caplog.text
        assert (
            "Install scikit-learn and torch, or use NIM-based approach" in caplog.text
        )

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_local_success(self, monkeypatch, caplog):
        """Test successful local model execution."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_check_jailbreak = mock.MagicMock(
            return_value={"jailbreak": True, "score": 0.95}
        )
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.model_based.checks.check_jailbreak",
            mock_check_jailbreak,
        )

        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection: {}
            """,
        )

        llm_task_manager = LLMTaskManager(config=config)
        context = {"user_message": "malicious prompt"}

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is True

        assert "Local model jailbreak detection result" in caplog.text
        mock_check_jailbreak.assert_called_once_with(prompt="malicious prompt")

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_empty_context(self, monkeypatch):
        """Test handling of empty context."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_nim_request = mock.AsyncMock(return_value=False)
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.actions.jailbreak_nim_request",
            mock_nim_request,
        )

        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection:
                  nim_base_url: "http://localhost:8000/v1"
            """,
        )

        llm_task_manager = LLMTaskManager(config=config)

        result = await jailbreak_detection_model(llm_task_manager, None)
        assert result is False

        mock_nim_request.assert_called_once_with(
            prompt="",
            nim_url="http://localhost:8000/v1",
            nim_auth_token=None,
            nim_classification_path="classify",
        )

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_context_without_user_message(
        self, monkeypatch
    ):
        """Test handling of context without user_message key."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_nim_request = mock.AsyncMock(return_value=False)
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.actions.jailbreak_nim_request",
            mock_nim_request,
        )

        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection:
                  nim_base_url: "http://localhost:8000/v1"
            """,
        )

        llm_task_manager = LLMTaskManager(config=config)
        context = {"other_key": "other_value"}  # No user_message key

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is False

        mock_nim_request.assert_called_once_with(
            prompt="",
            nim_url="http://localhost:8000/v1",
            nim_auth_token=None,
            nim_classification_path="classify",
        )

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_legacy_server_endpoint(self, monkeypatch):
        """Test fallback to legacy server_endpoint when nim_base_url is not set."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_model_request = mock.AsyncMock(return_value=True)
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.actions.jailbreak_detection_model_request",
            mock_model_request,
        )

        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection:
                  server_endpoint: "http://legacy-server:1337/model"
            """,
        )

        llm_task_manager = LLMTaskManager(config=config)
        context = {"user_message": "test prompt"}

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is True

        mock_model_request.assert_called_once_with(
            prompt="test prompt", api_url="http://legacy-server:1337/model"
        )

    @pytest.mark.asyncio
    async def test_jailbreak_detection_model_none_response_handling(
        self, monkeypatch, caplog
    ):
        """Test handling when external service returns None."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        mock_nim_request = mock.AsyncMock(return_value=None)
        monkeypatch.setattr(
            "nemoguardrails.library.jailbreak_detection.actions.jailbreak_nim_request",
            mock_nim_request,
        )

        config = RailsConfig.from_content(
            """
            define user express greeting
              "hello"
            """,
            """
            rails:
              config:
                jailbreak_detection:
                  nim_base_url: "http://localhost:8000/v1"
            """,
        )

        llm_task_manager = LLMTaskManager(config=config)
        context = {"user_message": "test prompt"}

        result = await jailbreak_detection_model(llm_task_manager, context)
        assert result is False

        assert "Jailbreak endpoint not set up properly" in caplog.text
