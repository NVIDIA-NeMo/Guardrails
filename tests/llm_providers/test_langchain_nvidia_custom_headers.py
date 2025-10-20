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

"""
Tests for the custom headers functionality in the ChatNVIDIA patch.

This module contains tests for the custom headers feature that allows users to pass
custom HTTP headers when making requests to NVIDIA AI endpoints.
"""

from unittest.mock import Mock, patch

import pytest

pytest.importorskip("langchain_nvidia_ai_endpoints")

from nemoguardrails.llm.providers._langchain_nvidia_ai_endpoints_patch import ChatNVIDIA


class TestChatNVIDIACustomHeadersInitialization:
    """Tests for ChatNVIDIA initialization with custom headers."""

    def test_init_without_custom_headers(self):
        """Test that ChatNVIDIA can be initialized without custom headers."""
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", base_url="http://localhost:8000/v1"
        )

        assert llm.model == "meta/llama-3.1-8b-instruct"
        assert llm.custom_headers is None
        assert llm.streaming is False

    def test_init_with_custom_headers(self):
        """Test that ChatNVIDIA can be initialized with custom headers."""
        custom_headers = {
            "X-Custom-Auth": "bearer-token",
            "X-Request-ID": "12345",
        }

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        assert llm.model == "meta/llama-3.1-8b-instruct"
        assert llm.custom_headers == custom_headers
        assert llm.streaming is False

    def test_init_with_empty_custom_headers(self):
        """Test that ChatNVIDIA handles empty custom headers dict."""
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers={},
            base_url="http://localhost:8000/v1",
        )

        assert llm.custom_headers == {}

    def test_custom_headers_field_is_optional(self):
        """Test that custom_headers field is optional and defaults to None."""
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", base_url="http://localhost:8000/v1"
        )

        assert hasattr(llm, "custom_headers")
        assert llm.custom_headers is None


class TestChatNVIDIACustomHeadersWrapping:
    """Tests for the method wrapping functionality."""

    def test_client_methods_wrapped_when_headers_present(self):
        """Test that _client methods are wrapped when custom headers are provided."""
        custom_headers = {"X-Test": "value"}

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        assert callable(llm._client.get_req)
        assert callable(llm._client.get_req_stream)
        assert llm._client.get_req.__name__ == "wrapped_get_req"
        assert llm._client.get_req_stream.__name__ == "wrapped_get_req_stream"

    def test_client_methods_not_wrapped_when_no_headers(self):
        """Test that _client methods are not wrapped when custom headers are None."""
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", base_url="http://localhost:8000/v1"
        )

        assert callable(llm._client.get_req)
        assert callable(llm._client.get_req_stream)
        assert llm._client.get_req.__name__ == "get_req"
        assert llm._client.get_req_stream.__name__ == "get_req_stream"

    def test_client_methods_not_wrapped_when_empty_dict(self):
        """Test that _client methods are not wrapped when custom headers is empty dict."""
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers={},
            base_url="http://localhost:8000/v1",
        )

        assert llm._client.get_req.__name__ == "get_req"
        assert llm._client.get_req_stream.__name__ == "get_req_stream"


class TestChatNVIDIACustomHeadersInRequests:
    """Tests for custom headers being included in HTTP requests."""

    def test_custom_headers_sent_in_invoke_request(self):
        """Test that custom headers are included in invoke() requests."""
        custom_headers = {
            "X-Custom-Auth": "test-token",
            "X-Request-ID": "12345",
        }

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "test response"}}]
        }

        captured_headers = {}

        def mock_post(*args, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})
            return mock_response

        with patch("requests.Session.post", side_effect=mock_post):
            llm.invoke("test message")

            assert "X-Custom-Auth" in captured_headers
            assert captured_headers["X-Custom-Auth"] == "test-token"
            assert "X-Request-ID" in captured_headers
            assert captured_headers["X-Request-ID"] == "12345"

    def test_custom_headers_merged_with_default_headers(self):
        """Test that custom headers are merged with default headers."""
        custom_headers = {"X-Custom-Header": "custom-value"}

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "test"}}]
        }

        captured_headers = {}

        def mock_post(*args, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})
            return mock_response

        with patch("requests.Session.post", side_effect=mock_post):
            llm.invoke("test")

            assert "X-Custom-Header" in captured_headers
            assert "Accept" in captured_headers
            assert "User-Agent" in captured_headers

    def test_multiple_custom_headers_sent(self):
        """Test that multiple custom headers are all sent correctly."""
        custom_headers = {
            "X-Header-1": "value1",
            "X-Header-2": "value2",
            "X-Header-3": "value3",
        }

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "test"}}]
        }

        captured_headers = {}

        def mock_post(*args, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})
            return mock_response

        with patch("requests.Session.post", side_effect=mock_post):
            llm.invoke("test")

            for key, value in custom_headers.items():
                assert key in captured_headers
                assert captured_headers[key] == value


class TestChatNVIDIACustomHeadersWithStreaming:
    """Tests for custom headers with streaming requests."""

    def test_custom_headers_sent_in_streaming_request(self):
        """Test that custom headers are included in streaming requests."""
        custom_headers = {
            "X-Stream-ID": "stream-123",
            "X-Custom-Auth": "stream-token",
        }

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            streaming=True,
            base_url="http://localhost:8000/v1",
        )

        captured_headers = {}

        def mock_post(*args, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})

            mock_response = Mock()
            mock_response.status_code = 200

            def mock_iter_lines():
                yield b'data: {"choices": [{"delta": {"content": "test"}, "finish_reason": "stop"}]}'

            mock_response.iter_lines = mock_iter_lines
            return mock_response

        with patch("requests.Session.post", side_effect=mock_post):
            list(llm.stream("test message"))

            assert "X-Stream-ID" in captured_headers
            assert captured_headers["X-Stream-ID"] == "stream-123"
            assert "X-Custom-Auth" in captured_headers
            assert captured_headers["X-Custom-Auth"] == "stream-token"


class TestChatNVIDIACustomHeadersPydanticCompatibility:
    """Tests for Pydantic compatibility with custom headers."""

    def test_model_dump_includes_custom_headers(self):
        """Test that model_dump() includes custom headers."""
        custom_headers = {"X-Test": "value"}

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        dumped = llm.model_dump()
        assert "custom_headers" in dumped
        assert dumped["custom_headers"] == custom_headers

    def test_custom_headers_type_validation(self):
        """Test that custom headers must be a dict of strings."""
        custom_headers = {"X-Test": "value"}

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        assert isinstance(llm.custom_headers, dict)
        assert all(isinstance(k, str) for k in llm.custom_headers.keys())
        assert all(isinstance(v, str) for v in llm.custom_headers.values())

    def test_dict_conversion_works(self):
        """Test that dict() conversion works with custom headers."""
        custom_headers = {"X-Test": "value"}

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        as_dict = dict(llm)
        assert "custom_headers" in as_dict


class TestChatNVIDIACustomHeadersEdgeCases:
    """Tests for edge cases and error handling."""

    def test_custom_headers_with_special_characters(self):
        """Test that headers with special characters work correctly."""
        custom_headers = {
            "X-Special-Chars": "value-with-dashes",
            "X-Numbers-123": "456",
        }

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        assert llm.custom_headers == custom_headers

    def test_custom_headers_immutability(self):
        """Test that modifying the original dict doesn't affect the LLM instance."""
        original_headers = {"X-Test": "original"}

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=original_headers.copy(),
            base_url="http://localhost:8000/v1",
        )

        original_headers["X-Test"] = "modified"
        original_headers["X-New"] = "new"

        assert llm.custom_headers["X-Test"] == "original"
        assert "X-New" not in llm.custom_headers

    def test_custom_headers_with_streaming_field(self):
        """Test that custom headers work together with streaming field."""
        custom_headers = {"X-Test": "value"}

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            streaming=True,
            base_url="http://localhost:8000/v1",
        )

        assert llm.custom_headers == custom_headers
        assert llm.streaming is True

    def test_custom_headers_preserved_across_multiple_calls(self):
        """Test that custom headers are preserved across multiple invoke calls."""
        custom_headers = {"X-Persistent": "value"}

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers=custom_headers,
            base_url="http://localhost:8000/v1",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "test"}}]
        }

        call_count = 0
        captured_headers_list = []

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_headers_list.append(kwargs.get("headers", {}))
            return mock_response

        with patch("requests.Session.post", side_effect=mock_post):
            llm.invoke("test 1")
            llm.invoke("test 2")
            llm.invoke("test 3")

            assert call_count == 3

            for headers in captured_headers_list:
                assert "X-Persistent" in headers
                assert headers["X-Persistent"] == "value"


class TestChatNVIDIACustomHeadersVersionCompatibility:
    """Tests for version compatibility checks."""

    def test_current_version_supports_extra_headers(self):
        """Test that the current installed version supports extra_headers parameter."""
        import inspect

        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", base_url="http://localhost:8000/v1"
        )

        sig = inspect.signature(llm._client.get_req)
        assert "extra_headers" in sig.parameters, (
            "Current version should support extra_headers. "
            "This test failing means you have an incompatible version installed."
        )

        sig_stream = inspect.signature(llm._client.get_req_stream)
        assert "extra_headers" in sig_stream.parameters, (
            "Current version should support extra_headers in get_req_stream. "
            "This test failing means you have an incompatible version installed."
        )

    def test_version_check_logic_with_missing_method(self):
        """Test that hasattr check works for detecting missing get_req method."""
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", base_url="http://localhost:8000/v1"
        )

        mock_client = Mock(spec=[])
        has_get_req = hasattr(mock_client, "get_req")

        assert not has_get_req, "Mock without get_req should return False for hasattr"

    def test_version_check_logic_with_missing_parameter(self):
        """Test that inspect.signature can detect missing extra_headers parameter."""
        import inspect

        def mock_get_req(payload={}):
            pass

        sig = inspect.signature(mock_get_req)
        has_extra_headers = "extra_headers" in sig.parameters

        assert (
            not has_extra_headers
        ), "Mock function without extra_headers should be detectable"

    def test_no_error_when_custom_headers_none(self):
        """Test that version checks are skipped when custom_headers is None."""
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", base_url="http://localhost:8000/v1"
        )
        assert llm.custom_headers is None

    def test_no_error_when_custom_headers_empty(self):
        """Test that version checks are skipped when custom_headers is empty dict."""
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            custom_headers={},
            base_url="http://localhost:8000/v1",
        )
        assert llm.custom_headers == {}
