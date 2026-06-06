# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for domain hallucination detection actions."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.library.domain_hallucination import actions


@dataclass
class MockLLMResponse:
    """Mock LLM response."""

    content: str


class TestSelfCheckDomainHallucination:
    """Test self_check_domain_hallucination rail action."""

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Test action with empty response."""
        result = await actions.self_check_domain_hallucination(
            llm_task_manager=None,
            context={},
            llm=None,
            config=None,
        )
        # Empty response should return True (safe)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_entities_in_response(self):
        """Test action with response containing no URLs/domains."""
        mock_llm_task_manager = MagicMock()
        mock_config = MagicMock()

        result = await actions.self_check_domain_hallucination(
            llm_task_manager=mock_llm_task_manager,
            context={
                "bot_message": "Machine learning is a branch of AI.",
                "user_message": "What is AI?",
            },
            llm=None,
            config=mock_config,
        )
        # No entities, should return True (safe)
        assert result is True

    @pytest.mark.asyncio
    async def test_context_message_extraction(self):
        """Test extraction of messages from different context keys."""
        mock_llm_task_manager = MagicMock()
        mock_config = MagicMock()

        # Mock the Layer 1 check to track if it was called
        with patch(
            "nemoguardrails.library.domain_hallucination.layer1_check.layer1_check_domain_hallucination"
        ) as mock_layer1:
            mock_layer1.return_value = {
                "is_hallucinated": False,
                "status": "clean",
            }

            # Test with bot_message
            await actions.self_check_domain_hallucination(
                llm_task_manager=mock_llm_task_manager,
                context={
                    "bot_message": "See https://github.com",
                },
                llm=None,
                config=mock_config,
            )
            assert mock_layer1.called

    @pytest.mark.asyncio
    async def test_response_mapping_safe(self):
        """Test output mapping for safe responses."""
        # When is_hallucinated=False, should return True (not blocked)
        with patch(
            "nemoguardrails.library.domain_hallucination.layer1_check.layer1_check_domain_hallucination"
        ) as mock_layer1:
            mock_layer1.return_value = {
                "is_hallucinated": False,
                "status": "clean",
            }

            mock_llm_task_manager = MagicMock()
            mock_config = MagicMock()

            result = await actions.self_check_domain_hallucination(
                llm_task_manager=mock_llm_task_manager,
                context={
                    "bot_message": "Visit https://pytorch.org",
                },
                llm=None,
                config=mock_config,
            )
            # Safe response: return True
            assert result is True

    @pytest.mark.asyncio
    async def test_response_mapping_hallucinated(self):
        """Test output mapping for hallucinated responses."""
        # When is_hallucinated=True, should return False (blocked)
        with patch(
            "nemoguardrails.library.domain_hallucination.layer1_check.layer1_check_domain_hallucination"
        ) as mock_layer1:
            mock_layer1.return_value = {
                "is_hallucinated": True,
                "status": "suspicious",
            }

            mock_llm_task_manager = MagicMock()
            mock_config = MagicMock()

            result = await actions.self_check_domain_hallucination(
                llm_task_manager=mock_llm_task_manager,
                context={
                    "bot_message": "Visit https://pytorch-fake.org",
                },
                llm=None,
                config=mock_config,
            )
            # Hallucinated response: return False (block)
            assert result is False

    @pytest.mark.asyncio
    async def test_kwargs_handling(self):
        """Test that extra kwargs are ignored."""
        with patch(
            "nemoguardrails.library.domain_hallucination.layer1_check.layer1_check_domain_hallucination"
        ) as mock_layer1:
            mock_layer1.return_value = {
                "is_hallucinated": False,
                "status": "clean",
            }

            mock_llm_task_manager = MagicMock()
            mock_config = MagicMock()

            # Should not raise with extra kwargs
            result = await actions.self_check_domain_hallucination(
                llm_task_manager=mock_llm_task_manager,
                context={"bot_message": "Safe text"},
                llm=None,
                config=mock_config,
                extra_param="should_be_ignored",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_alternative_context_keys(self):
        """Test alternative context key names."""
        with patch(
            "nemoguardrails.library.domain_hallucination.layer1_check.layer1_check_domain_hallucination"
        ) as mock_layer1:
            mock_layer1.return_value = {
                "is_hallucinated": False,
                "status": "clean",
            }

            mock_llm_task_manager = MagicMock()
            mock_config = MagicMock()

            # Test with last_bot_message
            await actions.self_check_domain_hallucination(
                llm_task_manager=mock_llm_task_manager,
                context={
                    "last_bot_message": "See https://pytorch.org",
                    "last_user_message": "Tell me about PyTorch",
                },
                llm=None,
                config=mock_config,
            )
            assert mock_layer1.called

            # Test with assistant_output
            await actions.self_check_domain_hallucination(
                llm_task_manager=mock_llm_task_manager,
                context={
                    "assistant_output": "Visit https://github.com",
                    "user_input": "What is GitHub?",
                },
                llm=None,
                config=mock_config,
            )
            assert mock_layer1.called


class TestActionSignature:
    """Test rail action signature and decorators."""

    def test_action_is_callable(self):
        """Test that self_check_domain_hallucination is callable."""
        assert callable(actions.self_check_domain_hallucination)

    def test_action_is_async(self):
        """Test that action is async."""
        import asyncio

        assert asyncio.iscoroutinefunction(actions.self_check_domain_hallucination)


class TestActionIntegration:
    """Integration tests with Layer 1 check."""

    @pytest.mark.asyncio
    async def test_action_calls_layer1_with_correct_args(self):
        """Test that action calls Layer 1 with correct arguments."""
        with patch(
            "nemoguardrails.library.domain_hallucination.layer1_check.layer1_check_domain_hallucination"
        ) as mock_layer1:
            mock_layer1.return_value = {
                "is_hallucinated": False,
                "status": "clean",
                "entities": {"has_entities": False},
            }

            mock_llm_task_manager = MagicMock()
            mock_config = MagicMock()
            mock_llm = MagicMock()

            bot_msg = "Check https://pytorch.org"
            user_msg = "Tell me about PyTorch"

            await actions.self_check_domain_hallucination(
                llm_task_manager=mock_llm_task_manager,
                context={
                    "bot_message": bot_msg,
                    "user_message": user_msg,
                },
                llm=mock_llm,
                config=mock_config,
            )

            # Verify Layer 1 was called with correct arguments
            assert mock_layer1.called
            call_args = mock_layer1.call_args
            assert call_args.kwargs["bot_response"] == bot_msg
            assert call_args.kwargs["user_message"] == user_msg
            assert call_args.kwargs["llm_task_manager"] == mock_llm_task_manager
            assert call_args.kwargs["config"] == mock_config

    @pytest.mark.asyncio
    async def test_logging_on_results(self):
        """Test that action logs Layer 1 results."""
        with patch(
            "nemoguardrails.library.domain_hallucination.layer1_check.layer1_check_domain_hallucination"
        ) as mock_layer1:
            mock_layer1.return_value = {
                "is_hallucinated": True,
                "status": "suspicious",
            }

            with patch("nemoguardrails.library.domain_hallucination.actions.logger") as mock_logger:
                mock_llm_task_manager = MagicMock()
                mock_config = MagicMock()

                await actions.self_check_domain_hallucination(
                    llm_task_manager=mock_llm_task_manager,
                    context={"bot_message": "Visit https://fake.com"},
                    llm=None,
                    config=mock_config,
                )

                # Should log the result
                assert mock_logger.info.called


class TestActionErrorHandling:
    """Test action error handling."""

    @pytest.mark.asyncio
    async def test_missing_context(self):
        """Test action with missing context."""
        result = await actions.self_check_domain_hallucination(
            llm_task_manager=MagicMock(),
            context=None,  # Missing context
            llm=None,
            config=MagicMock(),
        )
        # Should handle gracefully
        assert result is True

    @pytest.mark.asyncio
    async def test_empty_context(self):
        """Test action with empty context."""
        result = await actions.self_check_domain_hallucination(
            llm_task_manager=MagicMock(),
            context={},  # Empty context
            llm=None,
            config=MagicMock(),
        )
        # Should handle gracefully
        assert result is True


class TestFullIntegration:
    """Full integration tests from action to Layer 1."""

    @pytest.mark.asyncio
    async def test_action_with_real_layer1_path(self):
        """Test full action flow without LLM (fast path)."""
        # This tests the full path: action -> layer1_check -> extract_entities
        mock_llm_task_manager = MagicMock()
        mock_config = MagicMock()

        result = await actions.self_check_domain_hallucination(
            llm_task_manager=mock_llm_task_manager,
            context={"bot_message": "Talk about AI"},
            llm=None,
            config=mock_config,
        )

        # No entities, should return True (safe)
        assert result is True

    @pytest.mark.asyncio
    async def test_action_all_context_variants(self):
        """Test action with all possible context key variants."""
        with patch(
            "nemoguardrails.library.domain_hallucination.layer1_check.layer1_check_domain_hallucination"
        ) as mock_layer1:
            mock_layer1.return_value = {
                "is_hallucinated": False,
                "status": "clean",
            }

            mock_task_manager = MagicMock()
            mock_config = MagicMock()

            # Test with assistant_output and user_input
            result1 = await actions.self_check_domain_hallucination(
                llm_task_manager=mock_task_manager,
                context={
                    "assistant_output": "See https://pytorch.org",
                    "user_input": "Query",
                },
                llm=None,
                config=mock_config,
            )
            assert result1 is True

            # Test priority: bot_message > assistant_output
            result2 = await actions.self_check_domain_hallucination(
                llm_task_manager=mock_task_manager,
                context={
                    "bot_message": "Primary",
                    "assistant_output": "Secondary",
                    "last_bot_message": "Tertiary",
                },
                llm=None,
                config=mock_config,
            )

            # Verify bot_message was used (check call args)
            calls = mock_layer1.call_args_list
            assert calls[-1].kwargs["bot_response"] == "Primary"
