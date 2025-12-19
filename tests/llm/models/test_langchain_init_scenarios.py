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
Comprehensive tests for model initialization scenarios.

This module tests all possible paths through the initialization chain:

    INITIALIZATION ORDER (chat mode):
    ┌─────────────────────────────────────────────────────────────────┐
    │  #1  _handle_model_special_cases     [chat, text]               │
    │  #2  _init_chat_completion_model     [chat only]                │
    │  #3  _init_community_chat_models     [chat only]                │
    │  #4  _init_text_completion_model     [text, chat]               │
    └─────────────────────────────────────────────────────────────────┘

    INITIALIZATION ORDER (text mode):
    ┌─────────────────────────────────────────────────────────────────┐
    │  #1  _handle_model_special_cases     [chat, text]               │
    │  #4  _init_text_completion_model     [text, chat]               │
    │      (steps #2 and #3 are skipped - chat only)                  │
    └─────────────────────────────────────────────────────────────────┘

EXCEPTION PRIORITY RULES:
    1. ImportError (first one seen) - helps users know which package to install
    2. Last exception (if no ImportError) - later initializers are more specific
    3. Generic error (if no exceptions) - "Failed to initialize model..."

OUTCOME TYPES:
    Success  - Returns valid model, chain stops
    None     - Returns None, chain continues
    Error    - Raises exception, caught & stored, chain continues
    Skipped  - Mode not supported, skipped entirely
"""

from dataclasses import dataclass
from typing import Callable, Optional, Type
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.llm.models.langchain_initializer import (
    _PROVIDER_INITIALIZERS,
    _SPECIAL_MODEL_INITIALIZERS,
    ModelInitializationError,
    init_langchain_model,
)
from nemoguardrails.llm.providers.providers import (
    _chat_providers,
    _llm_providers,
    register_chat_provider,
    register_llm_provider,
)


@dataclass
class MockProvider:
    """Factory for creating mock provider classes with configurable behavior."""

    behavior: str
    error_type: Optional[Type[Exception]] = None
    error_msg: str = ""

    def create_class(self):
        """
        Create a provider class with the specified behavior.

        Behaviors:
        - "success": Provider initializes successfully
        - "error": Provider raises error_type with error_msg during __init__
        """
        behavior = self.behavior
        error_type = self.error_type
        error_msg = self.error_msg

        class _Provider:
            model_fields = {"model": None}

            def __init__(self, **kwargs):
                if behavior == "success":
                    self.model = kwargs.get("model")
                elif behavior == "error":
                    raise error_type(error_msg)

            async def _acall(self, *args, **kwargs):
                return "response"

        return _Provider


class ProviderRegistry:
    """
    Helper to register providers and automatically clean them up after tests.

    Usage:
        with registry fixture:
            registry.register_chat("name", provider_class)
            # provider is automatically removed after test
    """

    def __init__(self):
        self._originals = {}

    def register_chat(self, name: str, provider_cls):
        self._originals[("chat", name)] = _chat_providers.get(name)
        if provider_cls is None:
            _chat_providers[name] = None
        else:
            register_chat_provider(name, provider_cls)

    def register_llm(self, name: str, provider_cls):
        self._originals[("llm", name)] = _llm_providers.get(name)
        register_llm_provider(name, provider_cls)

    def register_special(self, pattern: str, handler: Callable):
        self._originals[("special", pattern)] = _SPECIAL_MODEL_INITIALIZERS.get(pattern)
        _SPECIAL_MODEL_INITIALIZERS[pattern] = handler

    def cleanup(self):
        for (ptype, name), original in self._originals.items():
            registry = {
                "chat": _chat_providers,
                "llm": _llm_providers,
                "special": _SPECIAL_MODEL_INITIALIZERS,
            }[ptype]

            if original is not None:
                registry[name] = original
            elif name in registry:
                del registry[name]


@pytest.fixture
def registry():
    reg = ProviderRegistry()
    yield reg
    reg.cleanup()


class TestSuccessScenarios:
    """
    Tests where initialization succeeds at various points in the chain.

    Each test verifies that when an initializer succeeds, the chain stops
    and returns the model without trying subsequent initializers.
    """

    @pytest.mark.parametrize(
        ("scenario", "model_name", "provider", "mode", "setup_fn"),
        [
            pytest.param(
                "special_case_success",
                "gpt-3.5-turbo-instruct",
                "openai",
                "chat",
                lambda r: r.register_llm("openai", MockProvider("success").create_class()),
                id="special_case_gpt35_instruct",
            ),
            pytest.param(
                "chat_completion_success",
                "test-model",
                "openai",
                "chat",
                lambda r: None,
                id="chat_completion_via_langchain",
            ),
            pytest.param(
                "community_chat_success",
                "test-model",
                "_test_community",
                "chat",
                lambda r: r.register_chat("_test_community", MockProvider("success").create_class()),
                id="community_chat_provider",
            ),
            pytest.param(
                "text_completion_success",
                "test-model",
                "_test_text",
                "text",
                lambda r: r.register_llm("_test_text", MockProvider("success").create_class()),
                id="text_completion_provider",
            ),
            pytest.param(
                "text_as_chat_fallback",
                "test-model",
                "_test_text_fallback",
                "chat",
                lambda r: r.register_llm("_test_text_fallback", MockProvider("success").create_class()),
                id="text_completion_as_chat_fallback",
            ),
        ],
    )
    def test_success_scenarios(self, registry, scenario, model_name, provider, mode, setup_fn):
        """
        Verify successful initialization at each point in the chain.

        When any initializer succeeds, the chain stops immediately and the
        model is returned. Subsequent initializers are not attempted.
        """
        setup_fn(registry)

        if scenario == "chat_completion_success":
            mock_model = MagicMock()
            with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model", return_value=mock_model):
                result = init_langchain_model(model_name, provider, mode, {})
                assert result == mock_model
        else:
            result = init_langchain_model(model_name, provider, mode, {})
            assert result is not None


class TestSingleErrorScenarios:
    """
    Tests where exactly one initializer raises an error.

    These verify that meaningful errors are preserved when later
    initializers return None (the core fix of PR #1516).

    WHY THIS MATTERS:
    Before PR #1516, when a provider wasn't found, the code raised
    RuntimeError("Could not find provider 'X'"). This masked meaningful
    errors from earlier initializers (e.g., "Invalid API key").

    After PR #1516, "provider not found" returns None instead, allowing
    meaningful errors to be preserved.
    """

    SINGLE_ERROR_CASES = [
        pytest.param(
            "chat",
            "_test_err",
            {"chat": ("error", ValueError, "Invalid API key")},
            "Invalid API key",
            id="chat_error_preserved",
        ),
        pytest.param(
            "chat",
            "_test_err",
            {"community": ("error", ValueError, "Rate limit exceeded")},
            "Rate limit exceeded",
            id="community_error_preserved",
        ),
        pytest.param(
            "text",
            "_test_err",
            {"llm": ("error", ValueError, "Invalid config")},
            "Invalid config",
            id="text_error_preserved",
        ),
        pytest.param(
            "chat",
            "_test_err",
            {"community": ("error", ImportError, "Missing package X")},
            "Missing package X",
            id="import_error_preserved",
        ),
    ]

    @pytest.mark.parametrize(("mode", "provider", "error_config", "expected_msg"), SINGLE_ERROR_CASES)
    def test_single_error_preserved(self, registry, mode, provider, error_config, expected_msg):
        """
        When one initializer raises an error and others return None,
        the error should be preserved in the final exception.

        This is the core behavior PR #1516 fixes: "provider not found"
        RuntimeErrors now return None instead of masking meaningful errors.
        """
        for init_type, (behavior, exc_type, msg) in error_config.items():
            provider_cls = MockProvider(behavior, exc_type, msg).create_class()
            if init_type == "chat":
                with patch(
                    "nemoguardrails.llm.models.langchain_initializer.init_chat_model",
                    side_effect=exc_type(msg),
                ):
                    with pytest.raises(ModelInitializationError) as exc_info:
                        init_langchain_model("test-model", provider, mode, {})
                    assert expected_msg in str(exc_info.value)
                return
            elif init_type == "community":
                registry.register_chat(provider, provider_cls)
            elif init_type == "llm":
                registry.register_llm(provider, provider_cls)

        with pytest.raises(ModelInitializationError) as exc_info:
            init_langchain_model("test-model", provider, mode, {})

        assert expected_msg in str(exc_info.value)


class TestMultipleErrorPriority:
    """
    Tests for exception priority when multiple initializers fail.

    PRIORITY RULES:
    1. ImportError (first seen) - always wins, helps with package installation
    2. Last exception - for non-ImportError, later errors take precedence

    WHY LAST EXCEPTION WINS:
    Later initializers in the chain are more specific fallbacks. For example:
    - #2 (chat_completion): General langchain initialization
    - #3 (community_chat): Specific community provider
    If both fail, the community error is likely more relevant.
    """

    @pytest.mark.parametrize(
        ("errors", "expected_winner", "reason"),
        [
            pytest.param(
                [("chat", ValueError, "Error A"), ("community", ValueError, "Error B")],
                "Error B",
                "Last ValueError wins (community is after chat)",
                id="valueerror_last_wins",
            ),
            pytest.param(
                [("chat", RuntimeError, "Error A"), ("community", ValueError, "Error B")],
                "Error B",
                "Last exception wins regardless of type",
                id="different_types_last_wins",
            ),
            pytest.param(
                [("chat", ImportError, "Import A"), ("community", ValueError, "Error B")],
                "Import A",
                "ImportError always wins over other exceptions",
                id="import_beats_value",
            ),
            pytest.param(
                [("chat", ValueError, "Error A"), ("community", ImportError, "Import B")],
                "Import B",
                "ImportError wins even if it comes later",
                id="later_import_still_wins",
            ),
            pytest.param(
                [("chat", ImportError, "Import A"), ("community", ImportError, "Import B")],
                "Import A",
                "First ImportError wins when multiple occur",
                id="first_import_wins",
            ),
        ],
    )
    def test_exception_priority(self, registry, errors, expected_winner, reason):
        """
        Verify exception priority rules are correctly applied.

        The system tracks:
        - first_import_error: First ImportError seen (never overwritten)
        - last_exception: Most recent exception (always overwritten)

        Final error uses first_import_error if set, else last_exception.
        """
        provider = "_test_priority"

        for init_type, exc_type, msg in errors:
            if init_type == "chat":
                chat_exc = (exc_type, msg)
            elif init_type == "community":
                registry.register_chat(provider, MockProvider("error", exc_type, msg).create_class())

        with patch(
            "nemoguardrails.llm.models.langchain_initializer.init_chat_model",
            side_effect=chat_exc[0](chat_exc[1]),
        ):
            with pytest.raises(ModelInitializationError) as exc_info:
                init_langchain_model("test-model", provider, "chat", {})

        assert expected_winner in str(exc_info.value), f"Failed: {reason}"


class TestErrorRecovery:
    """
    Tests where early errors are recovered by later successful initialization.

    KEY INSIGHT:
    When ANY initializer succeeds, all previous errors are discarded
    and the model is returned successfully. This allows the system to
    gracefully fall back through multiple initialization methods.
    """

    @pytest.mark.parametrize(
        ("failing_initializers", "succeeding_initializer"),
        [
            pytest.param(["special"], "chat", id="special_fails_chat_succeeds"),
            pytest.param(["special", "chat"], "community", id="special_chat_fail_community_succeeds"),
            pytest.param(["special", "chat", "community"], "text", id="all_fail_except_text"),
        ],
    )
    def test_later_success_recovers_from_errors(self, registry, failing_initializers, succeeding_initializer):
        """
        Errors from earlier initializers don't matter if a later one succeeds.
        The chain continues until success or all options exhausted.
        """
        provider = "_test_recovery"
        mock_model = MagicMock()

        if "special" in failing_initializers:

            def special_fails(*args, **kwargs):
                raise ValueError("Special failed")

            registry.register_special("test-recovery", special_fails)

        chat_behavior = mock_model if succeeding_initializer == "chat" else ValueError("Chat failed")
        community_cls = (
            MockProvider("success").create_class()
            if succeeding_initializer == "community"
            else MockProvider("error", ValueError, "Community failed").create_class()
        )
        text_cls = MockProvider("success").create_class() if succeeding_initializer == "text" else None

        registry.register_chat(provider, community_cls)
        if text_cls:
            registry.register_llm(provider, text_cls)

        with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model") as mock_chat:
            if isinstance(chat_behavior, MagicMock):
                mock_chat.return_value = chat_behavior
            else:
                mock_chat.side_effect = chat_behavior

            result = init_langchain_model("test-recovery-model", provider, "chat", {})
            assert result is not None


class TestSpecialCaseHandling:
    """
    Tests for special case handlers (gpt-3.5-turbo-instruct, nvidia).

    Special cases are tried FIRST and can override normal initialization.

    BUG FIX (PR #1516 + our fix):
    After PR #1516, special case handlers can return None when provider
    not found. Our fix ensures _handle_model_special_cases properly handles
    None returns without raising TypeError.
    """

    def test_gpt35_instruct_nonexistent_provider_no_typeerror(self, registry):
        """
        BUG FIX TEST: gpt-3.5-turbo-instruct with nonexistent provider.

        BEFORE FIX: _handle_model_special_cases raised TypeError("invalid type")
                    when _init_gpt35_turbo_instruct returned None.

        AFTER FIX: Returns None, chain continues, meaningful error preserved.

        Flow:
        1. _handle_model_special_cases -> _init_gpt35_turbo_instruct
        2. _init_text_completion_model -> provider not found -> returns None
        3. _init_gpt35_turbo_instruct returns None
        4. [BEFORE] isinstance(None, BaseLLM) fails -> TypeError
           [AFTER] result is None -> return None
        5. Chain continues to _init_chat_completion_model
        6. Meaningful error from langchain is surfaced
        """
        with pytest.raises(ModelInitializationError) as exc_info:
            init_langchain_model("gpt-3.5-turbo-instruct", "nonexistent_xyz", "chat", {})

        error_msg = str(exc_info.value)
        assert "invalid type" not in error_msg.lower(), "TypeError should not leak to user"
        assert "nonexistent_xyz" in error_msg

    def test_nvidia_provider_import_error(self, registry):
        """
        NVIDIA provider surfaces ImportError when package missing.

        nvidia_ai_endpoints is a provider-specific special case that
        requires langchain_nvidia_ai_endpoints package.
        """

        def nvidia_import_error(*args, **kwargs):
            raise ImportError("langchain_nvidia_ai_endpoints not installed")

        registry.register_special("nvidia_ai_endpoints", None)
        _PROVIDER_INITIALIZERS["nvidia_ai_endpoints"] = nvidia_import_error

        try:
            with pytest.raises(ModelInitializationError) as exc_info:
                init_langchain_model("test-model", "nvidia_ai_endpoints", "chat", {})

            assert "langchain_nvidia_ai_endpoints" in str(exc_info.value)
        finally:
            from nemoguardrails.llm.models.langchain_initializer import _init_nvidia_model

            _PROVIDER_INITIALIZERS["nvidia_ai_endpoints"] = _init_nvidia_model


class TestModeFiltering:
    """
    Tests that initializers are correctly filtered by mode.

    MODE FILTERING BEHAVIOR:
    - Chat mode: tries all 4 initializers (#1, #2, #3, #4)
    - Text mode: skips #2 (chat_completion) and #3 (community_chat)

    WHY: Chat completion and community chat are explicitly marked as
    supporting only "chat" mode in their ModelInitializer definitions.
    """

    def test_text_mode_skips_chat_initializers(self, registry):
        """
        In text mode, chat-only initializers (#2, #3) are skipped.

        This test verifies that even if a chat provider would raise an error,
        it's not attempted in text mode.
        """
        provider = "_test_text_mode"

        registry.register_chat(provider, MockProvider("error", ValueError, "SHOULD NOT SEE").create_class())
        registry.register_llm(provider, MockProvider("success").create_class())

        result = init_langchain_model("test-model", provider, "text", {})

        assert result is not None
        assert result.model == "test-model"

    def test_chat_mode_tries_all_initializers(self, registry):
        """
        In chat mode, all initializers are tried in order.

        Text completion (#4) is tried last as a fallback because it
        supports both "text" and "chat" modes.
        """
        provider = "_test_chat_mode"

        registry.register_llm(provider, MockProvider("success").create_class())

        with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model", return_value=None):
            result = init_langchain_model("test-model", provider, "chat", {})

        assert result is not None


class TestEdgeCases:
    """
    Edge cases and boundary conditions.

    These test unusual or malformed inputs to ensure graceful handling.
    """

    def test_none_provider_handled(self, registry):
        """
        Provider registered as None doesn't crash.

        This tests defensive programming - while registering None as a
        provider should never happen in practice, the code should handle
        it gracefully (fail with appropriate error, not crash).
        """
        provider = "_test_none"
        registry.register_chat(provider, None)
        _chat_providers[provider] = None

        with pytest.raises((ModelInitializationError, TypeError, AttributeError)):
            init_langchain_model("test-model", provider, "chat", {})

    def test_empty_model_name_rejected(self):
        """
        Empty model name raises clear error.

        Model name is required - fail early with clear message.
        """
        with pytest.raises(ModelInitializationError, match="Model name is required"):
            init_langchain_model("", "openai", "chat", {})

    def test_invalid_mode_rejected(self):
        """
        Invalid mode raises clear error.

        Only "chat" and "text" modes are supported.
        """
        with pytest.raises(ValueError, match="Unsupported mode"):
            init_langchain_model("test-model", "openai", "invalid", {})

    def test_all_return_none_generic_error(self, registry):
        """
        When all initializers return None, generic error is raised.

        This happens when:
        - No special case matches
        - All provider lookups return "not found"
        - No actual initialization is attempted

        The generic error tells the user initialization failed but doesn't
        have specific details since nothing actually tried and failed.
        """
        with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model", return_value=None):
            with pytest.raises(ModelInitializationError) as exc_info:
                init_langchain_model("test-model", "nonexistent_xyz", "chat", {})

            error_msg = str(exc_info.value)
            assert "Failed to initialize model" in error_msg
            assert "nonexistent_xyz" in error_msg


class TestE2EIntegration:
    """
    End-to-end tests through RailsConfig/LLMRails.

    These verify the full user-facing flow from config to error message,
    ensuring errors are properly propagated through the entire stack.

    FLOW: YAML config -> RailsConfig -> LLMRails -> init_llm_model ->
          init_langchain_model -> provider initialization
    """

    def test_e2e_meaningful_error_from_config(self, registry):
        """
        Full flow: RailsConfig -> LLMRails -> meaningful error.

        When a provider is found but initialization fails (e.g., invalid
        API key), the error should bubble up through the entire stack
        with the meaningful message intact.
        """
        from nemoguardrails import LLMRails, RailsConfig

        provider = "_e2e_test"
        registry.register_chat(provider, MockProvider("error", ValueError, "Invalid API key: sk-xxx").create_class())

        config = RailsConfig.from_content(
            config={"models": [{"type": "main", "engine": provider, "model": "test-model"}]}
        )

        with pytest.raises(ModelInitializationError) as exc_info:
            LLMRails(config=config)

        assert "Invalid API key" in str(exc_info.value)

    def test_e2e_successful_initialization(self, registry):
        """
        Full flow: RailsConfig -> LLMRails -> success.

        When provider is found and initializes successfully, the full
        stack should complete without errors.
        """
        from nemoguardrails import LLMRails, RailsConfig

        provider = "_e2e_success"
        registry.register_llm(provider, MockProvider("success").create_class())

        config = RailsConfig.from_content(
            config={"models": [{"type": "main", "engine": provider, "model": "test-model", "mode": "text"}]}
        )

        rails = LLMRails(config=config)
        assert rails.llm is not None


class TestMultipleErrorScenarios:
    """
    Tests for scenarios where multiple initializers raise exceptions.

    WHAT HAPPENS WITH MULTIPLE ERRORS:
    Each initializer that fails has its exception caught and stored.
    The final error message uses the exception with highest priority
    according to these rules:

    1. first_import_error (if any ImportError was seen)
       WHY: ImportErrors indicate missing packages, which is actionable
            for users ("pip install X")

    2. last_exception (if no ImportError)
       WHY: Later initializers are more specific fallbacks, so their
            errors are likely more relevant

    3. Generic message (if no exceptions at all)
       WHY: All initializers returned None (provider not found)
    """

    def test_all_initializers_raise_valueerror_last_one_wins(self, registry):
        """
        When all initializers raise ValueError, the LAST one wins.

        Flow: Special(Val) -> Chat(Val) -> Community(Val) -> Text(None)
        Expected: Community's ValueError (last non-None raiser)

        WHY: Community chat is the most specific initializer that ran,
        so its error is most relevant.
        """
        from nemoguardrails.llm.models.langchain_initializer import (
            _SPECIAL_MODEL_INITIALIZERS,
            ModelInitializationError,
            init_langchain_model,
        )

        def special_fails(*args, **kwargs):
            raise ValueError("Special case error")

        original_special = _SPECIAL_MODEL_INITIALIZERS.get("test-multi-error")

        with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model") as mock_chat:
            mock_chat.side_effect = ValueError("Chat completion error")

            with patch(
                "nemoguardrails.llm.models.langchain_initializer._get_chat_completion_provider"
            ) as mock_community:
                mock_provider = MagicMock()
                mock_provider.model_fields = {"model": None}
                mock_provider.side_effect = ValueError("Community chat error - SHOULD WIN")
                mock_community.return_value = mock_provider

                try:
                    _SPECIAL_MODEL_INITIALIZERS["test-multi-error"] = special_fails

                    with pytest.raises(ModelInitializationError) as exc_info:
                        init_langchain_model("test-multi-error-model", "fake_provider", "chat", {})

                    assert "Community chat error" in str(exc_info.value) or "Chat completion error" in str(
                        exc_info.value
                    )

                finally:
                    if original_special:
                        _SPECIAL_MODEL_INITIALIZERS["test-multi-error"] = original_special
                    elif "test-multi-error" in _SPECIAL_MODEL_INITIALIZERS:
                        del _SPECIAL_MODEL_INITIALIZERS["test-multi-error"]

    def test_importerror_from_chat_prioritized_over_valueerror_from_community(self, registry):
        """
        ImportError from chat completion is prioritized over ValueError from community.

        Flow: Special(None) -> Chat(ImportError) -> Community(ValueError) -> Text(None)
        Expected: Chat's ImportError (ImportError always wins)

        WHY: ImportError tells users which package to install. This is more
        actionable than a ValueError about configuration.
        """
        from nemoguardrails.llm.models.langchain_initializer import (
            ModelInitializationError,
            init_langchain_model,
        )
        from nemoguardrails.llm.providers.providers import (
            _chat_providers,
            register_chat_provider,
        )

        class ValueErrorProvider:
            model_fields = {"model": None}

            def __init__(self, **kwargs):
                raise ValueError("Community ValueError - should NOT win")

        test_provider = "_test_import_vs_value"
        original = _chat_providers.get(test_provider)

        try:
            register_chat_provider(test_provider, ValueErrorProvider)

            with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model") as mock_chat:
                mock_chat.side_effect = ImportError("Missing langchain_partner package - SHOULD WIN")

                with pytest.raises(ModelInitializationError) as exc_info:
                    init_langchain_model("test-model", test_provider, "chat", {})

                assert "langchain_partner" in str(exc_info.value)
                assert "should NOT win" not in str(exc_info.value).lower()

        finally:
            if original:
                _chat_providers[test_provider] = original
            elif test_provider in _chat_providers:
                del _chat_providers[test_provider]

    def test_first_importerror_wins_over_later_importerror(self, registry):
        """
        When multiple ImportErrors occur, the FIRST one wins.

        Flow: Special(None) -> Chat(ImportError#1) -> Community(ImportError#2) -> Text(None)
        Expected: Chat's ImportError (first ImportError)

        WHY: The first missing package encountered is the most direct
        blocker. Install that first, then retry.
        """
        from nemoguardrails.llm.models.langchain_initializer import (
            ModelInitializationError,
            init_langchain_model,
        )
        from nemoguardrails.llm.providers.providers import (
            _chat_providers,
            register_chat_provider,
        )

        class SecondImportErrorProvider:
            model_fields = {"model": None}

            def __init__(self, **kwargs):
                raise ImportError("Second ImportError - should NOT win")

        test_provider = "_test_first_import"
        original = _chat_providers.get(test_provider)

        try:
            register_chat_provider(test_provider, SecondImportErrorProvider)

            with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model") as mock_chat:
                mock_chat.side_effect = ImportError("First ImportError - SHOULD WIN")

                with pytest.raises(ModelInitializationError) as exc_info:
                    init_langchain_model("test-model", test_provider, "chat", {})

                assert "First ImportError" in str(exc_info.value)
                assert "Second ImportError" not in str(exc_info.value)

        finally:
            if original:
                _chat_providers[test_provider] = original
            elif test_provider in _chat_providers:
                del _chat_providers[test_provider]

    def test_special_case_error_masked_by_later_successful_init(self, registry):
        """
        When special case fails but later initializer succeeds, no error.

        Flow: Special(ValueError) -> Chat(Success)
        Expected: Success (chat model returned)

        WHY: The fallback system is working as designed. Special case
        failed, but a more general initializer succeeded.
        """
        from nemoguardrails.llm.models.langchain_initializer import (
            _SPECIAL_MODEL_INITIALIZERS,
            init_langchain_model,
        )

        def special_fails(*args, **kwargs):
            raise ValueError("Special case failed")

        original = _SPECIAL_MODEL_INITIALIZERS.get("test-recovery")

        try:
            _SPECIAL_MODEL_INITIALIZERS["test-recovery"] = special_fails

            mock_model = MagicMock()
            with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model") as mock_chat:
                mock_chat.return_value = mock_model

                result = init_langchain_model("test-recovery-model", "openai", "chat", {})
                assert result == mock_model

        finally:
            if original:
                _SPECIAL_MODEL_INITIALIZERS["test-recovery"] = original
            elif "test-recovery" in _SPECIAL_MODEL_INITIALIZERS:
                del _SPECIAL_MODEL_INITIALIZERS["test-recovery"]

    def test_chat_and_community_both_fail_community_wins(self, registry):
        """
        When chat and community both fail with ValueError, community (later) wins.

        Flow: Special(None) -> Chat(ValueError#1) -> Community(ValueError#2) -> Text(None)
        Expected: Community's ValueError (last exception)

        WHY: Community chat initializer is more specific than general
        chat completion, so its error is likely more relevant.
        """
        from nemoguardrails.llm.models.langchain_initializer import (
            ModelInitializationError,
            init_langchain_model,
        )
        from nemoguardrails.llm.providers.providers import register_chat_provider

        class CommunityFailProvider:
            model_fields = {"model": None}

            def __init__(self, **kwargs):
                raise ValueError("Community error: rate limit exceeded - SHOULD WIN")

        test_provider = "_test_chat_community_fail"
        original = _chat_providers.get(test_provider)

        try:
            register_chat_provider(test_provider, CommunityFailProvider)

            with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model") as mock_chat:
                mock_chat.side_effect = ValueError("Chat error: invalid model - should NOT win")

                with pytest.raises(ModelInitializationError) as exc_info:
                    init_langchain_model("test-model", test_provider, "chat", {})

                assert "rate limit exceeded" in str(exc_info.value)

        finally:
            if original:
                _chat_providers[test_provider] = original
            elif test_provider in _chat_providers:
                del _chat_providers[test_provider]

    def test_text_mode_special_fails_text_completion_fails(self, registry):
        """
        In text mode, when both special and text completion fail.

        Flow (text mode): Special(ValueError#1) -> Text(ValueError#2)
        Expected: Text's ValueError (last exception)

        WHY: In text mode, only 2 initializers run (special + text).
        Text completion is the more general initializer, but since it's
        the last one tried, its error takes precedence.
        """
        from nemoguardrails.llm.models.langchain_initializer import (
            _SPECIAL_MODEL_INITIALIZERS,
            ModelInitializationError,
            init_langchain_model,
        )
        from nemoguardrails.llm.providers.providers import register_llm_provider

        def special_fails(*args, **kwargs):
            raise ValueError("Special error - should NOT win")

        class TextFailProvider:
            model_fields = {"model": None}

            def __init__(self, **kwargs):
                raise ValueError("Text completion error - SHOULD WIN")

            async def _acall(self, *args, **kwargs):
                pass

        test_provider = "_test_text_mode_multi"
        original_special = _SPECIAL_MODEL_INITIALIZERS.get("test-text-multi")
        original_llm = _llm_providers.get(test_provider)

        try:
            _SPECIAL_MODEL_INITIALIZERS["test-text-multi"] = special_fails
            register_llm_provider(test_provider, TextFailProvider)

            with pytest.raises(ModelInitializationError) as exc_info:
                init_langchain_model("test-text-multi-model", test_provider, "text", {})

            assert "Text completion error" in str(exc_info.value)

        finally:
            if original_special:
                _SPECIAL_MODEL_INITIALIZERS["test-text-multi"] = original_special
            elif "test-text-multi" in _SPECIAL_MODEL_INITIALIZERS:
                del _SPECIAL_MODEL_INITIALIZERS["test-text-multi"]

            if original_llm:
                _llm_providers[test_provider] = original_llm
            elif test_provider in _llm_providers:
                del _llm_providers[test_provider]

    def test_runtimeerror_vs_valueerror_last_wins(self, registry):
        """
        RuntimeError and ValueError both caught by Exception handler, last wins.

        Flow: Special(None) -> Chat(RuntimeError) -> Community(ValueError) -> Text(None)
        Expected: Community's ValueError (last exception)

        WHY: Both RuntimeError and ValueError are caught by the same
        Exception handler. No special priority between them, so last wins.
        """
        from nemoguardrails.llm.models.langchain_initializer import (
            ModelInitializationError,
            init_langchain_model,
        )
        from nemoguardrails.llm.providers.providers import register_chat_provider

        class ValueErrorProvider:
            model_fields = {"model": None}

            def __init__(self, **kwargs):
                raise ValueError("ValueError from community - SHOULD WIN")

        test_provider = "_test_runtime_vs_value"
        original = _chat_providers.get(test_provider)

        try:
            register_chat_provider(test_provider, ValueErrorProvider)

            with patch("nemoguardrails.llm.models.langchain_initializer.init_chat_model") as mock_chat:
                mock_chat.side_effect = RuntimeError("RuntimeError from chat - should NOT win")

                with pytest.raises(ModelInitializationError) as exc_info:
                    init_langchain_model("test-model", test_provider, "chat", {})

                assert "ValueError from community" in str(exc_info.value)

        finally:
            if original:
                _chat_providers[test_provider] = original
            elif test_provider in _chat_providers:
                del _chat_providers[test_provider]
