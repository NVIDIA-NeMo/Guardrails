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

import warnings
from unittest.mock import MagicMock

import pytest

from nemoguardrails.llm.frameworks import (
    _areset_frameworks,
    _reset_frameworks,
    get_default_framework,
    get_framework,
    register_framework,
    set_default_framework,
)
from nemoguardrails.llm.providers import (
    get_chat_provider_names,
    get_llm_provider_names,
    get_provider_names,
    register_chat_provider,
    register_llm_provider,
    register_provider,
)
from nemoguardrails.types import LLMModel


@pytest.fixture(autouse=True)
def clean_registry():
    _reset_frameworks()
    yield
    _reset_frameworks()


class FakeFramework:
    def create_model(self, model_name, provider_name, model_kwargs=None):
        return MagicMock(spec=LLMModel)

    def register_provider(self, name, provider_cls):
        return None

    def get_provider_names(self):
        return []

    async def reset(self):
        return


class TestRegistry:
    def test_register_and_get_framework(self):
        fw = FakeFramework()
        register_framework("fake", fw)
        assert get_framework("fake") is fw

    def test_register_duplicate_raises_valueerror(self):
        register_framework("dup", FakeFramework())
        with pytest.raises(ValueError, match="already registered"):
            register_framework("dup", FakeFramework())

    def test_get_unregistered_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown framework"):
            get_framework("nonexistent")

    def test_langchain_lazy_auto_registration(self):
        fw = get_framework("langchain")
        from nemoguardrails.integrations.langchain.llm_adapter import LangChainFramework

        assert isinstance(fw, LangChainFramework)

    def test_set_and_get_default_framework(self):
        register_framework("custom", FakeFramework())
        set_default_framework("custom")
        assert get_default_framework() == "custom"

    def test_set_default_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown framework"):
            set_default_framework("nonexistent")

    def test_default_is_default(self):
        assert get_default_framework() == "default"

    def test_default_from_env_var(self, monkeypatch):
        monkeypatch.setenv("NEMOGUARDRAILS_LLM_FRAMEWORK", "litellm")
        _reset_frameworks()
        assert get_default_framework() == "litellm"

    def test_reset_clears_registry(self):
        register_framework("temp", FakeFramework())
        _reset_frameworks()
        with pytest.raises(KeyError):
            get_framework("temp")


class _ResetSpyFramework:
    def __init__(self):
        self.reset_count = 0

    def create_model(self, model_name, provider_name, model_kwargs=None):
        return MagicMock(spec=LLMModel)

    def register_provider(self, name, provider_cls):
        return None

    def get_provider_names(self):
        return []

    async def reset(self):
        self.reset_count += 1


class TestAresetFrameworks:
    @pytest.mark.asyncio
    async def test_async_reset_calls_framework_reset(self):
        spy = _ResetSpyFramework()
        register_framework("spy_running_loop", spy)
        await _areset_frameworks()
        assert spy.reset_count == 1

    @pytest.mark.asyncio
    async def test_async_reset_clears_pool(self):
        spy = _ResetSpyFramework()
        register_framework("spy_pool", spy)
        await _areset_frameworks()
        with pytest.raises(KeyError):
            get_framework("spy_pool")

    @pytest.mark.asyncio
    async def test_sync_wrapper_raises_in_running_loop(self):
        with pytest.raises(RuntimeError, match="asyncio.run"):
            _reset_frameworks()


class _FrameworkWithoutReset:
    """Used to validate that `_areset_frameworks` is defensive about a
    missing `reset` even though `register_framework` would now reject it."""

    def create_model(self, model_name, provider_name, model_kwargs=None):
        return MagicMock(spec=LLMModel)

    def register_provider(self, name, provider_cls):
        return None

    def get_provider_names(self):
        return []


class _RaisingFramework:
    def __init__(self, exc=None):
        self._exc = exc or RuntimeError("boom")

    def create_model(self, model_name, provider_name, model_kwargs=None):
        return MagicMock(spec=LLMModel)

    def register_provider(self, name, provider_cls):
        return None

    def get_provider_names(self):
        return []

    async def reset(self):
        raise self._exc


class TestResetFrameworksErrorIsolation:
    def test_failing_framework_does_not_block_subsequent_resets(self):
        bad = _RaisingFramework()
        good = _ResetSpyFramework()
        register_framework("bad", bad)
        register_framework("good", good)
        _reset_frameworks()
        assert good.reset_count == 1

    def test_failing_framework_still_clears_registry(self):
        bad = _RaisingFramework()
        register_framework("bad_only", bad)
        _reset_frameworks()
        with pytest.raises(KeyError):
            get_framework("bad_only")


class TestResetFrameworksMissingResetMethod:
    """`register_framework` now rejects frameworks without an async `reset`,
    but `_areset_frameworks` keeps a defensive guard for any framework that
    arrives in the registry through other paths (test injection, future
    refactors). These tests bypass `register_framework` to exercise that
    safety net directly."""

    def test_framework_without_reset_does_not_crash(self):
        from nemoguardrails.llm.frameworks import registry as _r

        _r._frameworks["no_reset"] = _FrameworkWithoutReset()
        _reset_frameworks()
        with pytest.raises(KeyError):
            get_framework("no_reset")

    def test_framework_without_reset_resets_default(self, monkeypatch):
        from nemoguardrails.llm.frameworks import registry as _r

        monkeypatch.setenv("NEMOGUARDRAILS_LLM_FRAMEWORK", "default")
        _r._frameworks["no_reset_2"] = _FrameworkWithoutReset()
        _reset_frameworks()
        assert get_default_framework() == "default"


class TestRegisterFrameworkValidation:
    """`register_framework` rejects two authoring mistakes at registration
    time: a sync `reset`, and an object that does not match the
    `LLMFramework` protocol."""

    def test_sync_reset_raises_typeerror(self):
        class BadFramework:
            def create_model(self, model_name, provider_name, model_kwargs=None):
                return MagicMock(spec=LLMModel)

            def register_provider(self, name, provider_cls):
                return None

            def get_provider_names(self):
                return []

            def reset(self):  # missing async
                return

        with pytest.raises(TypeError, match="must be an async coroutine function"):
            register_framework("bad_sync_reset", BadFramework())

    def test_missing_protocol_methods_raises_typeerror(self):
        class NotAFramework:
            pass

        with pytest.raises(TypeError, match="does not implement LLMFramework"):
            register_framework("not_a_framework", NotAFramework())

    def test_partial_protocol_raises_typeerror(self):
        class Partial:
            def create_model(self, model_name, provider_name, model_kwargs=None):
                return MagicMock(spec=LLMModel)

            async def reset(self):
                return

            # missing register_provider, get_provider_names

        with pytest.raises(TypeError, match="does not implement LLMFramework"):
            register_framework("partial", Partial())


class FakeChatProvider:
    pass


class FakeLLMProvider:
    async def _acall(self, prompt, stop=None, **kwargs):
        return "fake"


@pytest.fixture(autouse=False)
def clean_providers():
    from nemoguardrails.integrations.langchain.providers import providers as _p

    chat_backup = dict(_p._chat_providers)
    llm_backup = dict(_p._llm_providers)
    yield
    _p._chat_providers.clear()
    _p._chat_providers.update(chat_backup)
    _p._llm_providers.clear()
    _p._llm_providers.update(llm_backup)


@pytest.mark.usefixtures("clean_providers", "langchain_framework")
class TestProviderRegistration:
    def test_register_provider_appears_in_get_provider_names(self):
        register_provider("test_provider", FakeChatProvider)
        assert "test_provider" in get_provider_names()

    def test_register_chat_provider_appears_in_chat_names(self):
        register_chat_provider("test_chat", FakeChatProvider)
        assert "test_chat" in get_chat_provider_names()

    def test_register_llm_provider_appears_in_llm_names(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            register_llm_provider("test_llm", FakeLLMProvider)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert "test_llm" in get_llm_provider_names()

    def test_chat_and_llm_provider_names_are_different_subsets(self):
        register_chat_provider("only_chat_test", FakeChatProvider)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            register_llm_provider("only_llm_test", FakeLLMProvider)

        chat_names = get_chat_provider_names()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            llm_names = get_llm_provider_names()

        assert "only_chat_test" in chat_names
        assert "only_chat_test" not in llm_names
        assert "only_llm_test" in llm_names
        assert "only_llm_test" not in chat_names

    def test_get_provider_names_returns_both(self):
        register_chat_provider("both_chat", FakeChatProvider)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            register_llm_provider("both_llm", FakeLLMProvider)

        all_names = get_provider_names()
        assert "both_chat" in all_names
        assert "both_llm" in all_names

    def test_register_llm_provider_emits_deprecation(self):
        with pytest.warns(DeprecationWarning, match="removed in 0.23.0"):
            register_llm_provider("dep_test", FakeLLMProvider)

    def test_get_llm_provider_names_emits_deprecation(self):
        with pytest.warns(DeprecationWarning, match="removed in 0.23.0"):
            get_llm_provider_names()
