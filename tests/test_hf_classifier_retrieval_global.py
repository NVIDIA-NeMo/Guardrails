import unittest
from unittest.mock import AsyncMock

from nemoguardrails import LLMRails, RailsConfig


class MockHfClassifierResponse:
    def __init__(self, is_transform=False, transform_text=None, is_blocked=False):
        self.is_transform = is_transform
        self.transform_text = transform_text or {}
        self.is_blocked = is_blocked


class TestHfClassifierRetrievalGlobal(unittest.TestCase):

    def test_v1_retrieval_transform_propagates_global(self):
        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={
                "relevant_chunks": ["new_chunk_1", "new_chunk_2"],
                "response": "Transformed response"
            }
        )

        config = RailsConfig.from_content(
            colang_content="""
            define user express greeting
                "Hello"

            define flow test_retrieval
                $user_input = "test query"
                $classifier = "test_classifier"
                $relevant_chunks = execute hf classifier check retrieval
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
        app.generate(messages=[{"role": "user", "content": "Hello"}])

        self.assertEqual(app.context.get("relevant_chunks"), ["new_chunk_1", "new_chunk_2"])
        self.assertEqual(app.context.get("test_output"), ["new_chunk_1", "new_chunk_2"])

    def test_original_retrieval_transform_propagates_global(self):
        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={
                "relevant_chunks": ["original_chunk_1", "original_chunk_2"],
                "response": "Original transformed response"
            }
        )

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

        app = LLMRails(config)
        app.register_action("HfClassifierCheckRetrievalAction", mock_action)
        app.generate(messages=[{"role": "user", "content": "Hello"}])

        self.assertEqual(app.context.get("relevant_chunks"), ["original_chunk_1", "original_chunk_2"])
        self.assertEqual(app.context.get("test_output"), ["original_chunk_1", "original_chunk_2"])

    def test_no_transform_doesnt_modify_chunks(self):
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
        app.generate(messages=[{"role": "user", "content": "Hello"}])

        self.assertEqual(app.context.get("relevant_chunks"), ["original_chunk_1", "original_chunk_2"])
        self.assertEqual(app.context.get("test_output"), ["original_chunk_1", "original_chunk_2"])

    def test_v1_retrieval_with_empty_transform_text(self):
        mock_action = AsyncMock()
        mock_action.return_value = MockHfClassifierResponse(
            is_transform=True,
            transform_text={"relevant_chunks": []}
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
                $relevant_chunks = execute hf classifier check retrieval
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
        app.generate(messages=[{"role": "user", "content": "Hello"}])

        self.assertEqual(app.context.get("relevant_chunks"), [])
        self.assertEqual(app.context.get("test_output"), [])

    def test_v1_retrieval_blocked_flow(self):
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
        app.generate(messages=[{"role": "user", "content": "Hello"}])

        self.assertEqual(app.context.get("relevant_chunks"), ["original"])


if __name__ == "__main__":
    unittest.main()