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

"""E2E tests for llm_params functionality with real LLM providers."""

import os
import tempfile
from pathlib import Path

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.rails.llm.config import RailsConfig

LIVE_TEST_MODE = os.environ.get("LIVE_TEST_MODE") or os.environ.get("TEST_LIVE_MODE")


@pytest.fixture
def openai_config_content():
    """Create OpenAI config for testing."""
    return """
    models:
      - type: main
        engine: openai
        model: gpt-4o
    """


@pytest.fixture
def nim_config_content():
    """Create NIM config for testing."""
    return """
    models:
      - type: main
        engine: nim
        model: meta/llama-3.1-70b-instruct
        api_base: https://integrate.api.nvidia.com/v1
    """


@pytest.fixture
def openai_config_path(openai_config_content):
    """Create temporary OpenAI config file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yml"
        config_path.write_text(openai_config_content)
        yield str(temp_dir)


@pytest.fixture
def nim_config_path(nim_config_content):
    """Create temporary NIM config file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yml"
        config_path.write_text(nim_config_content)
        yield str(temp_dir)


@pytest.mark.skipif(
    not LIVE_TEST_MODE,
    reason="This test requires LIVE_TEST_MODE or TEST_LIVE_MODE environment variable to be set for live testing",
)
class TestLLMParamsOpenAI:
    """End-to-end tests for llm_params with OpenAI gpt-4o."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_openai_llm_params_temperature(self, openai_config_path):
        """Test that temperature parameter affects response variability with OpenAI."""
        config = RailsConfig.from_path(openai_config_path)
        rails = LLMRails(config, verbose=False)

        prompt = (
            "Say exactly 'Hello World' and nothing else and try to use a random word as a name, e.g Hello NVIDIAN!."
        )

        response1 = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.0}},
        )

        response2 = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.0}},
        )

        assert response1.response is not None
        assert response2.response is not None
        assert response1.response == response2.response

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_openai_llm_params_max_tokens(self, openai_config_path):
        """Test that max_tokens parameter limits response length with OpenAI."""
        config = RailsConfig.from_path(openai_config_path)
        rails = LLMRails(config, verbose=False)

        max_tokens_short = 10
        max_tokens_long = 100
        prompt = "Please describe the benefits of exercise. Be detailed."

        response_short = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"max_tokens": max_tokens_short}},
        )

        response_long = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"max_tokens": max_tokens_long}},
        )

        assert response_short.response is not None
        assert response_long.response is not None

        short_content = response_short.response[-1]["content"]
        long_content = response_long.response[-1]["content"]

        assert len(short_content) < len(long_content)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_openai_llm_params_combined(self, openai_config_path):
        """Test multiple parameters (temperature + max_tokens) with OpenAI."""
        config = RailsConfig.from_path(openai_config_path)
        rails = LLMRails(config, verbose=False)

        prompt = "Explain Python in exactly 5 words."

        response = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.1, "max_tokens": 10}},
        )

        assert response.response is not None
        content = response.response[-1]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_openai_llm_params_direct_llm_call(self, openai_config_path):
        """Test llm_params directly with llm_call function and OpenAI."""
        config = RailsConfig.from_path(openai_config_path)
        rails = LLMRails(config, verbose=False)

        llm = rails.llm
        prompt = "Say 'test' and nothing else."

        response = await llm_call(llm, prompt, llm_params={"temperature": 0.0, "max_tokens": 5})

        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_openai_llm_params_streaming(self, openai_config_path):
        """Test llm_params work with streaming responses from OpenAI."""
        config = RailsConfig.from_path(openai_config_path)
        config.streaming = True
        rails = LLMRails(config, verbose=False)

        prompt = "Count from 1 to 3."

        response = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.0, "max_tokens": 20}},
        )

        assert response.response is not None
        content = response.response[-1]["content"]
        assert "1" in content


@pytest.mark.skipif(
    not LIVE_TEST_MODE,
    reason="This test requires LIVE_TEST_MODE or TEST_LIVE_MODE environment variable to be set for live testing",
)
class TestLLMParamsNIM:
    """End-to-end tests for llm_params with NVIDIA NIM."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("NVIDIA_API_KEY"),
        reason="NVIDIA API key not available for e2e testing",
    )
    async def test_nim_llm_params_temperature(self, nim_config_path):
        """Test that temperature parameter works with NIM models."""
        config = RailsConfig.from_path(nim_config_path)
        rails = LLMRails(config, verbose=False)

        prompt = "Say exactly 'Hello World' and nothing else."

        response = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.1}},
        )

        assert response.response is not None
        content = response.response[-1]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("NVIDIA_API_KEY"),
        reason="NVIDIA API key not available for e2e testing",
    )
    async def test_nim_llm_params_max_tokens(self, nim_config_path):
        """Test that max_tokens parameter works with NIM models."""
        config = RailsConfig.from_path(nim_config_path)
        rails = LLMRails(config, verbose=False)

        prompt = "Write a short story."

        response = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"max_tokens": 15}},
        )

        assert response.response is not None
        content = response.response[-1]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("NVIDIA_API_KEY"),
        reason="NVIDIA API key not available for e2e testing",
    )
    async def test_nim_llm_params_combined(self, nim_config_path):
        """Test multiple parameters with NIM models."""
        config = RailsConfig.from_path(nim_config_path)
        rails = LLMRails(config, verbose=False)

        prompt = "Explain AI in simple terms."

        response = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.3, "max_tokens": 30}},
        )

        assert response.response is not None
        content = response.response[-1]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("NVIDIA_API_KEY"),
        reason="NVIDIA API key not available for e2e testing",
    )
    async def test_nim_llm_params_direct_llm_call(self, nim_config_path):
        """Test llm_params directly with llm_call function and NIM."""
        config = RailsConfig.from_path(nim_config_path)
        rails = LLMRails(config, verbose=False)

        llm = rails.llm
        prompt = "Say 'test'."

        response = await llm_call(llm, prompt, llm_params={"temperature": 0.2, "max_tokens": 10})

        assert response is not None
        assert len(response) > 0


@pytest.mark.skipif(
    not LIVE_TEST_MODE,
    reason="This test requires LIVE_TEST_MODE or TEST_LIVE_MODE environment variable to be set for live testing",
)
class TestLLMParamsIntegration:
    """Integration tests for llm_params functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_llm_params_isolation(self, openai_config_path):
        """Test that llm_params don't contaminate between calls."""
        config = RailsConfig.from_path(openai_config_path)
        rails = LLMRails(config, verbose=False)

        prompt = "Say 'test'."

        response1 = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.0}},
        )

        response2 = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.9}},
        )

        response3 = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.0}},
        )

        assert response1.response is not None
        assert response2.response is not None
        assert response3.response is not None

        content1 = response1.response[-1]["content"]
        content3 = response3.response[-1]["content"]

        assert content1 == content3

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_llm_params_with_rails(self, openai_config_path):
        """Test that llm_params work with output rails."""
        config = RailsConfig.from_path(openai_config_path)
        rails = LLMRails(config, verbose=False)

        prompt = "Hello there!"

        response = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.0, "max_tokens": 50}},
        )

        assert response.response is not None
        assert len(response.response) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_llm_params_override_defaults(self, openai_config_path):
        """Test that explicit llm_params override any defaults."""
        config = RailsConfig.from_path(openai_config_path)
        rails = LLMRails(config, verbose=False)

        prompt = "Generate a response."

        response_no_params = await rails.generate_async(messages=[{"role": "user", "content": prompt}])

        response_with_params = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": {"temperature": 0.0}},
        )

        # Handle different response formats (dict vs GenerationResponse)
        if hasattr(response_no_params, "response"):
            assert response_no_params.response is not None
        else:
            assert response_no_params is not None

        if hasattr(response_with_params, "response"):
            assert response_with_params.response is not None
        else:
            assert response_with_params is not None

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_openai_unsupported_params_error_handling(self, openai_config_path):
        """Test that unsupported parameters are properly handled."""
        # update config to use o1-mini which doesn't support temperature
        config_content = """
        models:
          - type: main
            engine: openai
            model: o1-mini
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(config_content)
            config = RailsConfig.from_path(temp_dir)
            rails = LLMRails(config, verbose=False)

            # this should fail with a clear error message
            with pytest.raises(Exception) as exc_info:
                await rails.generate_async(
                    messages=[{"role": "user", "content": "Say hello"}],
                    options={"llm_params": {"temperature": 0.5}},
                )

            error_message = str(exc_info.value)
            assert "temperature" in error_message.lower()
            assert "unsupported" in error_message.lower() or "not support" in error_message.lower()
