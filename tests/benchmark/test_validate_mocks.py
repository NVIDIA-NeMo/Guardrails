#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Tests for validate_mocks.py script.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from nemoguardrails.benchmark.validate_mocks import (
    check_endpoint,
    check_rails_endpoint,
    main,
)


class TestCheckEndpoint:
    """Tests for check_endpoint function."""

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_success(self, mock_get):
        """Test successful health and model checks."""
        # Mock health check response
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}

        # Mock models check response
        models_response = MagicMock()
        models_response.status_code = 200
        models_response.json.return_value = {
            "data": [
                {"id": "meta/llama-3.3-70b-instruct"},
                {"id": "other-model"},
            ]
        }

        mock_get.side_effect = [health_response, models_response]

        success, summary = check_endpoint(8000, "meta/llama-3.3-70b-instruct")

        assert success
        assert "PASSED" in summary
        assert "8000" in summary
        assert mock_get.call_count == 2

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_health_check_failed_status(self, mock_get):
        """Test health check with non-200 status code."""
        health_response = MagicMock()
        health_response.status_code = 404

        mock_get.return_value = health_response

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_health_check_unhealthy_status(self, mock_get):
        """Test health check with unhealthy status."""
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "unhealthy"}

        models_response = MagicMock()
        models_response.status_code = 200
        models_response.json.return_value = {"data": [{"id": "test-model"}]}

        mock_get.side_effect = [health_response, models_response]

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_health_check_json_decode_error(self, mock_get):
        """Test health check with invalid JSON."""
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "", 0
        )

        mock_get.return_value = health_response

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_health_connection_error(self, mock_get):
        """Test health check with connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary
        assert "Connection Error" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_health_timeout(self, mock_get):
        """Test health check with timeout."""
        mock_get.side_effect = requests.exceptions.Timeout()

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary
        assert "Connection Timeout" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_model_check_failed_status(self, mock_get):
        """Test model check with non-200 status code."""
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}

        models_response = MagicMock()
        models_response.status_code = 404

        mock_get.side_effect = [health_response, models_response]

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_model_not_found(self, mock_get):
        """Test model check when expected model is not in the list."""
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}

        models_response = MagicMock()
        models_response.status_code = 200
        models_response.json.return_value = {
            "data": [
                {"id": "other-model-1"},
                {"id": "other-model-2"},
            ]
        }

        mock_get.side_effect = [health_response, models_response]

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_model_check_json_decode_error(self, mock_get):
        """Test model check with invalid JSON."""
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}

        models_response = MagicMock()
        models_response.status_code = 200
        models_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "", 0
        )

        mock_get.side_effect = [health_response, models_response]

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_model_check_unexpected_json_structure(self, mock_get):
        """Test model check with unexpected JSON structure."""
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}

        models_response = MagicMock()
        models_response.status_code = 200
        # Return invalid structure that will cause AttributeError
        models_response.json.return_value = "invalid"

        mock_get.side_effect = [health_response, models_response]

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_model_check_connection_error(self, mock_get):
        """Test model check with connection error."""
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}

        mock_get.side_effect = [health_response, requests.exceptions.ConnectionError()]

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_endpoint_model_check_timeout(self, mock_get):
        """Test model check with timeout."""
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}

        mock_get.side_effect = [health_response, requests.exceptions.Timeout()]

        success, summary = check_endpoint(8000, "test-model")

        assert not success
        assert "FAILED" in summary


class TestCheckRailsEndpoint:
    """Tests for check_rails_endpoint function."""

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_rails_endpoint_success(self, mock_get):
        """Test successful rails config check."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {"id": "config1", "name": "Config 1"},
            {"id": "config2", "name": "Config 2"},
        ]

        mock_get.return_value = response

        success, summary = check_rails_endpoint(9000)

        assert success
        assert "PASSED" in summary
        assert "9000" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_rails_endpoint_non_200_status(self, mock_get):
        """Test rails config check with non-200 status."""
        response = MagicMock()
        response.status_code = 404
        response.json.return_value = []

        mock_get.return_value = response

        success, summary = check_rails_endpoint(9000)

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_rails_endpoint_empty_list(self, mock_get):
        """Test rails config check with empty list response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = []

        mock_get.return_value = response

        success, summary = check_rails_endpoint(9000)

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_rails_endpoint_not_a_list(self, mock_get):
        """Test rails config check with non-list response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"error": "invalid"}

        mock_get.return_value = response

        success, summary = check_rails_endpoint(9000)

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_rails_endpoint_json_decode_error(self, mock_get):
        """Test rails config check with invalid JSON."""
        response = MagicMock()
        response.status_code = 200
        response.text = "invalid json"
        response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "", 0
        )

        mock_get.return_value = response

        success, summary = check_rails_endpoint(9000)

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_rails_endpoint_connection_error(self, mock_get):
        """Test rails config check with connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        success, summary = check_rails_endpoint(9000)

        assert not success
        assert "FAILED" in summary

    @patch("nemoguardrails.benchmark.validate_mocks.requests.get")
    def test_check_rails_endpoint_timeout(self, mock_get):
        """Test rails config check with timeout."""
        mock_get.side_effect = requests.exceptions.Timeout()

        success, summary = check_rails_endpoint(9000)

        assert not success
        assert "FAILED" in summary


class TestMain:
    """Tests for main function."""

    @patch("nemoguardrails.benchmark.validate_mocks.check_rails_endpoint")
    @patch("nemoguardrails.benchmark.validate_mocks.check_endpoint")
    def test_main_all_passed(self, mock_check_endpoint, mock_check_rails_endpoint):
        """Test main function when all checks pass."""
        mock_check_endpoint.side_effect = [
            (True, "Port 8000 (meta/llama-3.3-70b-instruct): PASSED"),
            (
                True,
                "Port 8001 (nvidia/llama-3.1-nemoguard-8b-content-safety): PASSED",
            ),
        ]
        mock_check_rails_endpoint.return_value = (
            True,
            "Port 9000 (Rails Config): PASSED",
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        assert mock_check_endpoint.call_count == 2
        assert mock_check_rails_endpoint.call_count == 1

    @patch("nemoguardrails.benchmark.validate_mocks.check_rails_endpoint")
    @patch("nemoguardrails.benchmark.validate_mocks.check_endpoint")
    def test_main_one_failed(self, mock_check_endpoint, mock_check_rails_endpoint):
        """Test main function when one check fails."""
        mock_check_endpoint.side_effect = [
            (False, "Port 8000 (meta/llama-3.3-70b-instruct): FAILED"),
            (
                True,
                "Port 8001 (nvidia/llama-3.1-nemoguard-8b-content-safety): PASSED",
            ),
        ]
        mock_check_rails_endpoint.return_value = (
            True,
            "Port 9000 (Rails Config): PASSED",
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("nemoguardrails.benchmark.validate_mocks.check_rails_endpoint")
    @patch("nemoguardrails.benchmark.validate_mocks.check_endpoint")
    def test_main_all_failed(self, mock_check_endpoint, mock_check_rails_endpoint):
        """Test main function when all checks fail."""
        mock_check_endpoint.side_effect = [
            (False, "Port 8000 (meta/llama-3.3-70b-instruct): FAILED"),
            (
                False,
                "Port 8001 (nvidia/llama-3.1-nemoguard-8b-content-safety): FAILED",
            ),
        ]
        mock_check_rails_endpoint.return_value = (
            False,
            "Port 9000 (Rails Config): FAILED",
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("nemoguardrails.benchmark.validate_mocks.check_rails_endpoint")
    @patch("nemoguardrails.benchmark.validate_mocks.check_endpoint")
    def test_main_rails_failed(self, mock_check_endpoint, mock_check_rails_endpoint):
        """Test main function when only rails check fails."""
        mock_check_endpoint.side_effect = [
            (True, "Port 8000 (meta/llama-3.3-70b-instruct): PASSED"),
            (
                True,
                "Port 8001 (nvidia/llama-3.1-nemoguard-8b-content-safety): PASSED",
            ),
        ]
        mock_check_rails_endpoint.return_value = (
            False,
            "Port 9000 (Rails Config): FAILED",
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


class TestScriptExecution:
    """Test the script can be executed directly."""

    def test_script_execution(self):
        """Test that the script can be executed via __main__."""
        # Import the module to trigger __main__ block if run directly
        import importlib

        import nemoguardrails.benchmark.validate_mocks as vm_module

        # Reload to ensure __main__ block is checked
        importlib.reload(vm_module)

        # If we're not running as __main__, the main function shouldn't be called
        # This test verifies the module can be imported without issues
        assert callable(vm_module.check_endpoint)
        assert callable(vm_module.check_rails_endpoint)
        assert callable(vm_module.main)

    @patch("nemoguardrails.benchmark.validate_mocks.check_rails_endpoint")
    @patch("nemoguardrails.benchmark.validate_mocks.check_endpoint")
    def test_script_can_run_as_main(
        self, mock_check_endpoint, mock_check_rails_endpoint
    ):
        """Test that the script can run when invoked as main."""
        mock_check_endpoint.side_effect = [
            (True, "Port 8000: PASSED"),
            (True, "Port 8001: PASSED"),
        ]
        mock_check_rails_endpoint.return_value = (True, "Port 9000: PASSED")

        # This simulates running the script directly
        script_path = "/Users/tgasser/projects/nemo_guardrails/nemoguardrails/benchmark/validate_mocks.py"
        with pytest.raises(SystemExit) as exc_info:
            with open(script_path, encoding="utf-8") as f:
                # Using exec to simulate script execution is necessary for testing __main__ block
                exec(
                    compile(f.read(), "validate_mocks.py", "exec"),
                    {"__name__": "__main__"},
                )  # noqa: S102

        assert exc_info.value.code == 0
