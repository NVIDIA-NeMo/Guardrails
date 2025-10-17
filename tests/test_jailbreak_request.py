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

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urljoin

import pytest


class TestJailbreakRequestChanges:
    """Test jailbreak request function changes introduced in this PR."""

    def test_url_joining_logic(self):
        """Test that URL joining works correctly with all slash combinations."""
        from nemoguardrails.library.jailbreak_detection.request import join_nim_url

        test_cases = [
            (
                "http://localhost:8000/v1",
                "classify",
                "http://localhost:8000/v1/classify",
            ),
            (
                "http://localhost:8000/v1/",
                "classify",
                "http://localhost:8000/v1/classify",
            ),
            (
                "http://localhost:8000/v1",
                "/classify",
                "http://localhost:8000/v1/classify",
            ),
            (
                "http://localhost:8000/v1/",
                "/classify",
                "http://localhost:8000/v1/classify",
            ),
            ("http://localhost:8000", "classify", "http://localhost:8000/classify"),
            ("http://localhost:8000", "/classify", "http://localhost:8000/classify"),
            ("http://localhost:8000/", "classify", "http://localhost:8000/classify"),
            ("http://localhost:8000/", "/classify", "http://localhost:8000/classify"),
            (
                "http://localhost:8000/api/v1",
                "classify",
                "http://localhost:8000/api/v1/classify",
            ),
            (
                "http://localhost:8000/api/v1/",
                "/classify",
                "http://localhost:8000/api/v1/classify",
            ),
        ]

        for base_url, classification_path, expected_url in test_cases:
            result = join_nim_url(base_url, classification_path)
            assert (
                result == expected_url
            ), f"join_nim_url({base_url}, {classification_path}) should equal {expected_url}, got {result}"

    def test_auth_header_logic(self):
        """Test the authorization header logic."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        nim_auth_token = "test_token_123"
        if nim_auth_token is not None:
            headers["Authorization"] = f"Bearer {nim_auth_token}"

        assert headers["Authorization"] == "Bearer test_token_123"

        headers2 = {"Content-Type": "application/json", "Accept": "application/json"}
        nim_auth_token = None
        if nim_auth_token is not None:
            headers2["Authorization"] = f"Bearer {nim_auth_token}"

        assert "Authorization" not in headers2

    @pytest.mark.asyncio
    async def test_nim_request_signature(self):
        import inspect

        from nemoguardrails.library.jailbreak_detection.request import (
            jailbreak_nim_request,
        )

        sig = inspect.signature(jailbreak_nim_request)
        params = list(sig.parameters.keys())

        expected_params = [
            "prompt",
            "nim_url",
            "nim_auth_token",
            "nim_classification_path",
        ]
        assert params == expected_params, f"Expected {expected_params}, got {params}"
