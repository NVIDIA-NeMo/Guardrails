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
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.llm.providers.providers import _llm_providers
from nemoguardrails.llm.providers.trtllm.llm import TRTLLM


def test_trtllm_provider_registered():
    """Test that the TRTLLM provider is registered in the _llm_providers dictionary."""
    assert "trt_llm" in _llm_providers
    assert _llm_providers["trt_llm"] == TRTLLM


@pytest.mark.asyncio
async def test_trtllm_provider_has_acall():
    """Test that the TRTLLM provider has the _acall method."""
    assert hasattr(TRTLLM, "_acall")
    assert callable(getattr(TRTLLM, "_acall"))


@pytest.mark.asyncio
async def test_trtllm_provider_acall_implementation():
    """Test the implementation of the _acall method in the TRTLLM provider."""
    # Create a mock instance of TRTLLM
    with patch("nemoguardrails.llm.providers.trtllm.llm.TRTLLM") as mock_trtllm:
        # Configure the mock to return a specific value from _call
        mock_instance = MagicMock()
        mock_instance._call.return_value = "Mock TRTLLM response"

        # Add the _acall method to the mock instance
        async def mock_acall(*args, **kwargs):
            return await asyncio.to_thread(mock_instance._call, *args, **kwargs)

        mock_instance._acall = mock_acall
        mock_trtllm.return_value = mock_instance

        # Create an instance and call _acall
        trtllm_instance = mock_trtllm()
        result = await trtllm_instance._acall("test prompt")

        # Verify the result
        assert result == "Mock TRTLLM response"
        mock_instance._call.assert_called_once_with("test prompt")
