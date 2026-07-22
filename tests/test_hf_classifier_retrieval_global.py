# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import AsyncMock, MagicMock

from nemoguardrails import LLMRails, RailsConfig


class MockHfClassifierResponse:
    """Mock response from HF classifier"""

    def __init__(self, is_transform=False, transform_text=None, is_blocked=False):
        self.is_transform = is_transform
        self.transform_text = transform_text or {}
        self.is_blocked = is_blocked


class TestHfClassifierRetrievalGlobal(unittest.TestCase):
    """Test cases for HF classifier retrieval global variable propagation"""

    def test_hf_classifier_v1_retrieval_transform_propagates_global(self):
        """Test that transformed relevant_chunks propagates to global context in v1"""

        # Create mock action that returns transform response
        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={
                "relevant_chunks": ["new_chunk_1", "new_chunk_2"],
                "response": "Transformed response"
            }
        )

        # Create config with the v1 flow
        config = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_retrieval
                $user_input = "test query"
                $classifier = "test_classifier"
                execute hf classifier check retrieval
                $test_output = $relevant_chunks
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        # Initialize LLMRails
        app = LLMRails(config)

        # Override the action with mock
        app.register_action("hf_classifier_check_retrieval", mock_action)

        # Run the flow
        result = app.generate(messages=[{"role": "user", "content": "Hello"}])

        # Verify that $relevant_chunks was propagated to global context
        self.assertEqual(app.context.get("relevant_chunks"), ["new_chunk_1", "new_chunk_2"])
        self.assertEqual(app.context.get("test_output"), ["new_chunk_1", "new_chunk_2"])

    def test_hf_classifier_v1_retrieval_transform_propagates_to_caller_flow(self):
        """Test that transformed relevant_chunks is accessible in caller flow for v1"""

        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={
                "relevant_chunks": ["caller_chunk_1", "caller_chunk_2"],
                "response": "Transformed response"
            }
        )

        config = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow parent_flow
                $user_input = "test query"
                $classifier = "test_classifier"
                execute hf classifier check retrieval
                $parent_output = $relevant_chunks

            define flow test_parent
                execute parent_flow
                $test_result = $parent_output
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        app = LLMRails(config)
        app.register_action("hf_classifier_check_retrieval", mock_action)

        result = app.generate(messages=[{"role": "user", "content": "Hello"}])

        # Verify propagation to parent flow and beyond
        self.assertEqual(app.context.get("relevant_chunks"), ["caller_chunk_1", "caller_chunk_2"])
        self.assertEqual(app.context.get("parent_output"), ["caller_chunk_1", "caller_chunk_2"])
        self.assertEqual(app.context.get("test_result"), ["caller_chunk_1", "caller_chunk_2"])

    def test_hf_classifier_v1_retrieval_no_transform_doesnt_modify_chunks(self):
        """Test that when is_transform is False, relevant_chunks is not modified in v1"""

        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=False,
            is_blocked=False
        )

        config = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_no_transform
                $user_input = "test query"
                $original_chunks = ["original_chunk_1", "original_chunk_2"]
                $relevant_chunks = $original_chunks
                $classifier = "test_classifier"
                execute hf classifier check retrieval
                $test_output = $relevant_chunks
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        app = LLMRails(config)
        app.register_action("hf_classifier_check_retrieval", mock_action)

        result = app.generate(messages=[{"role": "user", "content": "Hello"}])

        # Verify relevant_chunks remains unchanged
        self.assertEqual(app.context.get("relevant_chunks"), ["original_chunk_1", "original_chunk_2"])
        self.assertEqual(app.context.get("test_output"), ["original_chunk_1", "original_chunk_2"])

    def test_hf_classifier_v1_retrieval_with_empty_transform_text(self):
        """Test transform with empty transform_text in v1"""

        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={}
        )

        config = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_empty_transform
                $user_input = "test query"
                $original_chunks = ["original"]
                $relevant_chunks = $original_chunks
                $classifier = "test_classifier"
                execute hf classifier check retrieval
                $test_output = $relevant_chunks
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        app = LLMRails(config)
        app.register_action("hf_classifier_check_retrieval", mock_action)

        result = app.generate(messages=[{"role": "user", "content": "Hello"}])

        # Should set relevant_chunks to whatever transform_text returns
        # In this case, it should be an empty dict or the value from transform_text
        self.assertEqual(app.context.get("relevant_chunks"), {})

    def test_hf_classifier_v1_retrieval_blocked_with_exceptions(self):
        """Test blocked flow with exceptions enabled in v1"""

        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=False,
            is_blocked=True
        )

        config = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_blocked
                $user_input = "test query"
                $original_chunks = ["original"]
                $relevant_chunks = $original_chunks
                $classifier = "test_classifier"
                $config.enable_rails_exceptions = True
                execute hf classifier check input
                $test_output = $relevant_chunks
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        app = LLMRails(config)
        app.register_action("hf_classifier_check_input", mock_action)

        # Should create event and stop, but relevant_chunks should remain unchanged
        result = app.generate(messages=[{"role": "user", "content": "Hello"}])

        # relevant_chunks should still be the original value since the flow stopped
        self.assertEqual(app.context.get("relevant_chunks"), ["original"])

    def test_hf_classifier_v1_multiple_calls_keep_transformed_value(self):
        """Test that transformed relevant_chunks persists across multiple flow calls"""

        mock_action = AsyncMock()
        # First call returns transform
        mock_action.side_effect = [
            MockHfClassifierResponse(
                is_transform=True,
                transform_text={
                    "relevant_chunks": ["first_transform_chunk"],
                    "response": "First transform"
                }
            ),
            MockHfClassifierResponse(
                is_transform=True,
                transform_text={
                    "relevant_chunks": ["second_transform_chunk"],
                    "response": "Second transform"
                }
            )
        ]

        config = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_multiple
                $user_input = "test query"
                $classifier = "test_classifier"

                # First call
                $original_chunks = ["original"]
                $relevant_chunks = $original_chunks
                execute hf classifier check retrieval
                $first_output = $relevant_chunks

                # Second call
                execute hf classifier check retrieval
                $second_output = $relevant_chunks
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        app = LLMRails(config)
        app.register_action("hf_classifier_check_retrieval", mock_action)

        result = app.generate(messages=[{"role": "user", "content": "Hello"}])

        # Verify the transformations persist correctly
        self.assertEqual(app.context.get("relevant_chunks"), ["second_transform_chunk"])
        self.assertEqual(app.context.get("first_output"), ["first_transform_chunk"])
        self.assertEqual(app.context.get("second_output"), ["second_transform_chunk"])

    def test_original_hf_classifier_retrieval_transform_propagates_global(self):
        """Test that transformed relevant_chunks propagates to global context in original flow"""

        # Create mock action that returns transform response
        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={
                "relevant_chunks": ["original_chunk_1", "original_chunk_2"],
                "response": "Original transformed response"
            }
        )

        # Create config with the original flow
        config = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_retrieval
                $user_input = "test query"
                $classifier = "test_classifier"
                execute hf classifier check retrieval $classifier
                $test_output = $relevant_chunks
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        # Initialize LLMRails
        app = LLMRails(config)

        # Override the action with mock
        app.register_action("HfClassifierCheckRetrievalAction", mock_action)

        # Run the flow
        result = app.generate(messages=[{"role": "user", "content": "Hello"}])

        # Verify that $relevant_chunks was propagated to global context
        self.assertEqual(app.context.get("relevant_chunks"), ["original_chunk_1", "original_chunk_2"])
        self.assertEqual(app.context.get("test_output"), ["original_chunk_1", "original_chunk_2"])

    def test_both_versions_with_classifier_parameter(self):
        """Test both versions with classifier parameter"""

        # Test v1 version
        mock_action_v1 = AsyncMock()
        mock_action_v1.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={
                "relevant_chunks": ["v1_test_chunk"],
                "response": "V1 test response"
            }
        )

        config_v1 = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_v1
                $user_input = "test query"
                $classifier = "test_classifier"
                execute hf classifier check retrieval
                $result_v1 = $relevant_chunks
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        app_v1 = LLMRails(config_v1)
        app_v1.register_action("hf_classifier_check_retrieval", mock_action_v1)
        result_v1 = app_v1.generate(messages=[{"role": "user", "content": "Hello"}])

        # Test original version
        mock_action_original = AsyncMock()
        mock_action_original.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={
                "relevant_chunks": ["original_test_chunk"],
                "response": "Original test response"
            }
        )

        config_original = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_original
                $user_input = "test query"
                $classifier = "test_classifier"
                execute hf classifier check retrieval $classifier
                $result_original = $relevant_chunks
            """,
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """
        )

        app_original = LLMRails(config_original)
        app_original.register_action("HfClassifierCheckRetrievalAction", mock_action_original)
        result_original = app_original.generate(messages=[{"role": "user", "content": "Hello"}])

        # Verify the fix works for both versions
        self.assertEqual(app_v1.context.get("relevant_chunks"), ["v1_test_chunk"])
        self.assertEqual(app_v1.context.get("result_v1"), ["v1_test_chunk"])

        self.assertEqual(app_original.context.get("relevant_chunks"), ["original_test_chunk"])
        self.assertEqual(app_original.context.get("result_original"), ["original_test_chunk"])


if __name__ == "__main__":
    unittest.main()