# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""E2E tests for LLM isolation in streaming scenarios."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.config import RailsConfig

TEST_LIVE_MODE = os.environ.get("TEST_LIVE_MODE")


@pytest.fixture
def test_config_content():
    """Create a test configuration for isolated testing."""
    return """
    models:
      - type: main
        engine: openai
        model: gpt-4o-mini

    rails:
      output:
        parallel: True
        streaming:
          enabled: True
          stream_first: False
        flows:
          - self check output

    streaming: True

    prompts:
      - task: self_check_output
        content: |
          Your task is to check if the bot message below complies with the company policy.
          Company policy for the bot:
          - messages should not contain harmful content

          Bot message: {{ bot_message }}

          Question: Should the message be blocked (Yes or No)?
          Answer:
    """


@pytest.fixture
def test_config_path(test_config_content):
    """Create a temporary config file for testing.

    This is intentional to use from_path
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yml"
        config_path.write_text(test_config_content)
        yield str(temp_dir)


@pytest.mark.skipif(
    not TEST_LIVE_MODE,
    reason="This test requires TEST_LIVE_MODE environment variable to be set for live testing",
)
class TestLLMIsolationE2E:
    """End-to-end tests for LLM isolation functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available for e2e testing",
    )
    async def test_parameter_isolation_in_streaming_no_contamination(
        self, test_config_path
    ):
        """Test that parameter modifications in actions don't contaminate main LLM.

        This is the main test that verifies the fix for the max_tokens contamination bug.
        """

        config = RailsConfig.from_path(test_config_path)
        rails = LLMRails(config, verbose=False)

        # track LLM state before and after streaming calls
        llm_states = []

        async def capture_llm_state(iteration: int, when: str):
            """Capture current LLM state for analysis."""
            state = {
                "iteration": iteration,
                "when": when,
                "max_tokens_attr": getattr(rails.llm, "max_tokens", None),
                "model_kwargs": getattr(rails.llm, "model_kwargs", {}).copy(),
                "max_tokens_in_kwargs": getattr(rails.llm, "model_kwargs", {}).get(
                    "max_tokens", "NOT_SET"
                ),
            }
            llm_states.append(state)
            return state

        # perform multiple streaming iterations
        responses = []
        for i in range(3):
            await capture_llm_state(i + 1, "before")

            # perform streaming call that triggers output rails
            response = ""
            try:
                async for chunk in rails.stream_async(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Write exactly 20 words about Python programming in iteration {i + 1}",
                        }
                    ]
                ):
                    response += chunk
            except Exception as e:
                response = f"Error: {str(e)}"

            responses.append(response.strip())

            # capture state after streaming call
            await capture_llm_state(i + 1, "after")

        # analyze results for parameter contamination
        contamination_detected = False
        contaminated_states = []
        for state in llm_states:
            # check if max_tokens=3 (from self_check_output) contaminated main LLM
            if state["max_tokens_attr"] == 3 or state["max_tokens_in_kwargs"] == 3:
                contamination_detected = True
                contaminated_states.append(state)

        # analyze response quality (truncation indicates contamination)
        truncated_responses = []
        for i, response in enumerate(responses):
            if response and not response.startswith("Error:"):
                word_count = len(response.split())
                if word_count < 10:
                    truncated_responses.append(
                        {
                            "iteration": i + 1,
                            "word_count": word_count,
                            "response": response,
                        }
                    )

        assert (
            not contamination_detected
        ), f"Parameter contamination detected in LLM states: {contaminated_states}"

        assert len(truncated_responses) == 0, (
            f"Found {len(truncated_responses)} truncated responses: {truncated_responses}. "
            f"This indicates parameter contamination."
        )

        # verify we got reasonable responses
        valid_responses = [r for r in responses if r and not r.startswith("Error:")]
        assert (
            len(valid_responses) >= 2
        ), f"Too many API errors, can't verify isolation. Responses: {responses}"

    @pytest.mark.asyncio
    async def test_isolated_llm_registration_during_initialization(
        self, test_config_path
    ):
        """Test that isolated LLMs are properly registered during initialization."""

        config = RailsConfig.from_path(test_config_path)
        rails = LLMRails(config, verbose=False)

        registered_params = rails.runtime.registered_action_params

        assert "llm" in registered_params, "Main LLM not registered"

        isolated_llm_params = [
            key
            for key in registered_params.keys()
            if key.endswith("_llm") and key != "llm"
        ]

        assert (
            len(isolated_llm_params) > 0
        ), f"No isolated LLMs were created. Registered params: {list(registered_params.keys())}"

        # verify isolated LLMs are different instances from main LLM
        main_llm = registered_params["llm"]
        for param_name in isolated_llm_params:
            isolated_llm = registered_params[param_name]
            assert (
                isolated_llm is not main_llm
            ), f"Isolated LLM '{param_name}' is the same instance as main LLM"

            # verify model_kwargs are isolated (different dict instances)
            if hasattr(isolated_llm, "model_kwargs") and hasattr(
                main_llm, "model_kwargs"
            ):
                assert (
                    isolated_llm.model_kwargs is not main_llm.model_kwargs
                ), f"Isolated LLM '{param_name}' shares model_kwargs dict with main LLM"

    @pytest.mark.asyncio
    async def test_concurrent_action_execution_with_different_parameters(
        self, test_config_path
    ):
        """Test that concurrent actions with different parameters don't interfere."""

        config = RailsConfig.from_path(test_config_path)
        rails = LLMRails(config, verbose=False)

        # create mock actions that would use different LLM parameters
        original_llm_state = {
            "max_tokens": getattr(rails.llm, "max_tokens", None),
            "temperature": getattr(rails.llm, "temperature", None),
            "model_kwargs": getattr(rails.llm, "model_kwargs", {}).copy(),
        }

        async def simulate_concurrent_actions():
            """Simulate multiple actions running concurrently."""
            # this simulates what happens during parallel rails when multiple
            # output rails run concurrently

            tasks = []

            # simulate different actions that would modify LLM parameters
            for i in range(3):
                task = asyncio.create_task(
                    self._simulate_action_with_llm_params(
                        rails, f"action_{i}", i * 10 + 3
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        await simulate_concurrent_actions()

        # verify main LLM state is unchanged
        final_llm_state = {
            "max_tokens": getattr(rails.llm, "max_tokens", None),
            "temperature": getattr(rails.llm, "temperature", None),
            "model_kwargs": getattr(rails.llm, "model_kwargs", {}).copy(),
        }

        assert original_llm_state == final_llm_state, (
            f"Main LLM state changed after concurrent actions. "
            f"Original: {original_llm_state}, Final: {final_llm_state}"
        )

    async def _simulate_action_with_llm_params(
        self, rails, action_name: str, max_tokens: int
    ):
        """Simulate action that uses llm_params context manager."""
        from nemoguardrails.llm.params import llm_params

        action_llm_param = f"{action_name}_llm"
        if action_llm_param in rails.runtime.registered_action_params:
            action_llm = rails.runtime.registered_action_params[action_llm_param]
        else:
            action_llm = rails.llm  # fallback to main LLM

        async with llm_params(action_llm, max_tokens=max_tokens, temperature=0.1):
            await asyncio.sleep(0.01)

            return {
                "action": action_name,
                "llm_id": id(action_llm),
                "max_tokens": getattr(action_llm, "max_tokens", None),
                "model_kwargs": getattr(action_llm, "model_kwargs", {}).copy(),
            }

    def test_shallow_copy_preserves_important_attributes(self, test_config_path):
        """Test that shallow copy preserves HTTP clients and other important attributes."""

        config = RailsConfig.from_path(test_config_path)
        rails = LLMRails(config, verbose=False)

        isolated_llm_params = [
            key
            for key in rails.runtime.registered_action_params.keys()
            if key.endswith("_llm") and key != "llm"
        ]

        if not isolated_llm_params:
            pytest.skip("No isolated LLMs found for testing")

        main_llm = rails.runtime.registered_action_params["llm"]
        isolated_llm = rails.runtime.registered_action_params[isolated_llm_params[0]]

        if hasattr(main_llm, "client"):
            assert hasattr(
                isolated_llm, "client"
            ), "HTTP client not preserved in isolated LLM"
            assert (
                isolated_llm.client is main_llm.client
            ), "HTTP client should be shared (shallow copy)"

        if hasattr(main_llm, "api_key"):
            assert hasattr(
                isolated_llm, "api_key"
            ), "API key not preserved in isolated LLM"
            assert (
                isolated_llm.api_key == main_llm.api_key
            ), "API key should be preserved"

        # model_kwargs should be isolated (deep copy of this specific dict)
        if hasattr(main_llm, "model_kwargs") and hasattr(isolated_llm, "model_kwargs"):
            assert (
                isolated_llm.model_kwargs is not main_llm.model_kwargs
            ), "model_kwargs should be isolated between LLM instances"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("iterations", [1, 3, 5])
    async def test_parameter_isolation_multiple_iterations(
        self, test_config_path, iterations
    ):
        """Test parameter isolation across different numbers of iterations."""

        config = RailsConfig.from_path(test_config_path)
        rails = LLMRails(config, verbose=False)

        responses = []
        contamination_detected = False

        for i in range(iterations):
            # LLM state before call
            _pre_state = {
                "max_tokens": getattr(rails.llm, "max_tokens", None),
                "model_kwargs_max_tokens": getattr(rails.llm, "model_kwargs", {}).get(
                    "max_tokens", "NOT_SET"
                ),
            }

            try:
                # simulate the streaming call without actually calling API
                # just trigger the initialization and check state
                response = f"Mock response for iteration {i + 1}"
                responses.append(response)
            except Exception as e:
                responses.append(f"Error: {str(e)}")

            # check LLM state after call
            post_state = {
                "max_tokens": getattr(rails.llm, "max_tokens", None),
                "model_kwargs_max_tokens": getattr(rails.llm, "model_kwargs", {}).get(
                    "max_tokens", "NOT_SET"
                ),
            }

            # check for contamination
            if (
                post_state["max_tokens"] == 3
                or post_state["model_kwargs_max_tokens"] == 3
            ):
                contamination_detected = True
                break

        assert (
            not contamination_detected
        ), f"Parameter contamination detected after {iterations} iterations"

        assert (
            len(responses) == iterations
        ), f"Expected {iterations} responses, got {len(responses)}"


@pytest.mark.skipif(
    not TEST_LIVE_MODE,
    reason="This test requires TEST_LIVE_MODE environment variable to be set for live testing",
)
class TestLLMIsolationErrorHandling:
    """Test error handling and edge cases in LLM isolation."""

    def test_initialization_with_no_actions(self, test_config_path):
        """Test LLM isolation when no actions are loaded."""

        minimal_config_content = """
        models:
          - type: main
            engine: openai
            model: gpt-4o-mini
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(minimal_config_content)

            # should not crash even with no actions
            config = RailsConfig.from_path(str(temp_dir))
            rails = LLMRails(config, verbose=False)

            # should have main LLM registered
            assert "llm" in rails.runtime.registered_action_params

    def test_initialization_with_specialized_llms_only(self):
        """Test that specialized LLMs from config are preserved."""

        config_content = """
        models:
          - type: main
            engine: openai
            model: gpt-4o-mini
          - type: content_safety
            engine: openai
            model: gpt-3.5-turbo
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(config_content)

            config = RailsConfig.from_path(str(temp_dir))
            rails = LLMRails(config, verbose=False)

            assert "llm" in rails.runtime.registered_action_params
            assert "content_safety_llm" in rails.runtime.registered_action_params

            main_llm = rails.runtime.registered_action_params["llm"]
            content_safety_llm = rails.runtime.registered_action_params[
                "content_safety_llm"
            ]
            assert main_llm is not content_safety_llm


async def run_parameter_contamination_test():
    """Manual test runner for debugging."""
    test_instance = TestLLMIsolationE2E()

    test_config = """
    models:
      - type: main
        engine: openai
        model: gpt-4o-mini

    rails:
      output:
        parallel: True
        streaming:
          enabled: True
          stream_first: False
        flows:
          - self check output

    streaming: True
    """

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yml"
        config_path.write_text(test_config)

        await test_instance.test_parameter_isolation_in_streaming_no_contamination(
            temp_dir
        )


if __name__ == "__main__":
    asyncio.run(run_parameter_contamination_test())
