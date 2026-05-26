# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for deprecated public-API aliases and Guardrails-facade proxies.

Covers the new property/setter surface introduced when LLMRails internal
attributes were reclassified as private:

* Deprecated read-only aliases on LLMRails (kb, embedding_search_providers,
  default_embedding_{model,engine,params}, llm_generation_actions).
* Deprecated read/write alias on LLMRails for explain_info.
* First-class passthrough_fn property/setter on LLMRails (no warning).
* Guardrails-facade proxies for explain_info (deprecated), passthrough_fn,
  and events_history_cache — including the IORails-engine raise paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails import Guardrails
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG


@pytest.fixture
def _content_safety_config():
    """A config IORails can handle — used for both engine-routing fixtures below."""
    return RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)


@pytest.fixture
def llmrails_guardrails(_content_safety_config):
    """Guardrails wrapping an uninitialized LLMRails (use_iorails=False).

    Patches LLMRails.__init__ to a no-op so we get a Guardrails whose
    ``rails_engine`` is a real LLMRails instance with no attributes set —
    tests then poke the specific underscored attributes they need.
    """
    with patch.object(LLMRails, "__init__", return_value=None):
        yield Guardrails(config=_content_safety_config, use_iorails=False)


@pytest.fixture
def iorails_guardrails(_content_safety_config):
    """Guardrails wrapping an uninitialized IORails (use_iorails=True).

    Patches IORails.__init__ so the facade routes to IORails without
    actually starting any work queues, telemetry, or HTTP clients.
    """
    with patch.object(IORails, "__init__", return_value=None):
        yield Guardrails(config=_content_safety_config, use_iorails=True)


class TestLLMRailsDeprecatedAliases:
    """Deprecated property aliases on LLMRails.

    Each public name is a thin getter (and, for default_embedding_*, also a
    setter) that warns and forwards to the underscored attribute. Tests
    verify both the warning and the forwarded value/assignment.
    """

    @pytest.mark.parametrize(
        "public_name, private_name",
        [
            ("kb", "_kb"),
            ("embedding_search_providers", "_embedding_search_providers"),
            ("default_embedding_model", "_default_embedding_model"),
            ("default_embedding_engine", "_default_embedding_engine"),
            ("default_embedding_params", "_default_embedding_params"),
            ("llm_generation_actions", "_llm_generation_actions"),
        ],
    )
    def test_read_emits_deprecation_warning(self, public_name, private_name):
        """Reading a deprecated alias emits DeprecationWarning and returns the underscored value."""
        # Bypass __init__: we're testing the @property descriptor itself,
        # not LLMRails setup (which would need an LLM, runtime, etc.).
        rails = LLMRails.__new__(LLMRails)
        sentinel = object()
        setattr(rails, private_name, sentinel)
        with pytest.warns(DeprecationWarning, match=rf"LLMRails\.{public_name}.*deprecated"):
            value = getattr(rails, public_name)
        assert value is sentinel

    @pytest.mark.parametrize(
        "public_name, private_name",
        [
            ("default_embedding_model", "_default_embedding_model"),
            ("default_embedding_engine", "_default_embedding_engine"),
            ("default_embedding_params", "_default_embedding_params"),
        ],
    )
    def test_write_emits_deprecation_warning(self, public_name, private_name):
        """Writing a deprecated default_embedding_* alias warns and forwards to the underscored attribute.

        These have setters (unlike kb / embedding_search_providers / llm_generation_actions)
        because they were previously plain instance attributes that downstream code may have
        written to. The setter preserves that write path during deprecation.
        """
        # Bypass __init__: testing only the @setter descriptor.
        rails = LLMRails.__new__(LLMRails)
        sentinel = object()
        with pytest.warns(DeprecationWarning, match=rf"Setting LLMRails\.{public_name}.*deprecated"):
            setattr(rails, public_name, sentinel)
        assert getattr(rails, private_name) is sentinel


class TestLLMRailsExplainInfoDeprecation:
    """explain_info has both a deprecated getter and a deprecated setter on LLMRails."""

    def test_read_emits_warning(self):
        """Reading rails.explain_info warns and returns rails._explain_info by reference."""
        # Bypass __init__: testing only the @property descriptor.
        rails = LLMRails.__new__(LLMRails)
        sentinel = object()
        rails._explain_info = sentinel
        with pytest.warns(DeprecationWarning, match=r"LLMRails\.explain_info.*deprecated"):
            value = rails.explain_info
        assert value is sentinel

    def test_write_emits_warning(self):
        """Writing rails.explain_info warns and forwards the assignment to rails._explain_info."""
        # Bypass __init__: testing only the @setter descriptor.
        rails = LLMRails.__new__(LLMRails)
        rails._explain_info = None
        sentinel = object()
        with pytest.warns(DeprecationWarning, match=r"LLMRails\.explain_info"):
            rails.explain_info = sentinel
        assert rails._explain_info is sentinel


class TestLLMRailsPassthroughFn:
    """First-class passthrough_fn API on LLMRails — no deprecation warning."""

    def test_read_forwards_to_actions(self):
        """rails.passthrough_fn returns _llm_generation_actions._passthrough_fn (no warning)."""
        # Bypass __init__: testing only the @property descriptor.
        rails = LLMRails.__new__(LLMRails)
        rails._llm_generation_actions = MagicMock()
        sentinel = object()
        rails._llm_generation_actions._passthrough_fn = sentinel
        assert rails.passthrough_fn is sentinel

    def test_write_forwards_to_actions(self):
        """rails.passthrough_fn = fn writes to _llm_generation_actions._passthrough_fn (no warning)."""
        # Bypass __init__: testing only the @setter descriptor.
        rails = LLMRails.__new__(LLMRails)
        rails._llm_generation_actions = MagicMock()
        sentinel = object()
        rails.passthrough_fn = sentinel
        assert rails._llm_generation_actions._passthrough_fn is sentinel


class TestGuardrailsFacadeExplainInfo:
    """Facade proxy for explain_info: deprecated getter+setter under LLMRails, raises under IORails."""

    def test_read_delegates_to_llmrails(self, llmrails_guardrails):
        """Reading guardrails.explain_info warns and returns the wrapped LLMRails's _explain_info."""
        sentinel = object()
        llmrails_guardrails.rails_engine._explain_info = sentinel
        with pytest.warns(DeprecationWarning, match=r"Guardrails\.explain_info.*deprecated"):
            value = llmrails_guardrails.explain_info
        assert value is sentinel

    def test_write_delegates_to_llmrails(self, llmrails_guardrails):
        """Writing guardrails.explain_info warns and writes to the wrapped LLMRails's _explain_info."""
        llmrails_guardrails.rails_engine._explain_info = None
        sentinel = object()
        with pytest.warns(DeprecationWarning, match=r"Setting Guardrails\.explain_info"):
            llmrails_guardrails.explain_info = sentinel
        assert llmrails_guardrails.rails_engine._explain_info is sentinel

    def test_read_raises_under_iorails(self, iorails_guardrails):
        """Reading guardrails.explain_info on an IORails-backed facade raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r"IORails doesn't support explain_info"):
            _ = iorails_guardrails.explain_info

    def test_write_raises_under_iorails(self, iorails_guardrails):
        """Writing guardrails.explain_info on an IORails-backed facade raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r"IORails doesn't support explain_info"):
            iorails_guardrails.explain_info = None


class TestGuardrailsFacadePassthroughFn:
    """Facade proxy for passthrough_fn: first-class getter+setter under LLMRails, raises under IORails."""

    def test_read_delegates_to_llmrails(self, llmrails_guardrails):
        """Reading guardrails.passthrough_fn returns the wrapped LLMRails's passthrough_fn (no warning)."""
        llmrails_guardrails.rails_engine._llm_generation_actions = MagicMock()
        sentinel = object()
        llmrails_guardrails.rails_engine._llm_generation_actions._passthrough_fn = sentinel
        assert llmrails_guardrails.passthrough_fn is sentinel

    def test_write_delegates_to_llmrails(self, llmrails_guardrails):
        """Writing guardrails.passthrough_fn writes through to LLMRails's _passthrough_fn storage."""
        llmrails_guardrails.rails_engine._llm_generation_actions = MagicMock()
        sentinel = object()
        llmrails_guardrails.passthrough_fn = sentinel
        assert llmrails_guardrails.rails_engine._llm_generation_actions._passthrough_fn is sentinel

    def test_read_raises_under_iorails(self, iorails_guardrails):
        """Reading guardrails.passthrough_fn on an IORails-backed facade raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r"IORails doesn't support passthrough_fn"):
            _ = iorails_guardrails.passthrough_fn

    def test_write_raises_under_iorails(self, iorails_guardrails):
        """Writing guardrails.passthrough_fn on an IORails-backed facade raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r"IORails doesn't support passthrough_fn"):
            iorails_guardrails.passthrough_fn = None


class TestGuardrailsFacadeEventsHistoryCache:
    """Facade proxy for events_history_cache: plain getter+setter under LLMRails, raises under IORails."""

    def test_read_delegates_to_llmrails(self, llmrails_guardrails):
        """Reading guardrails.events_history_cache returns the wrapped LLMRails's plain attribute."""
        sentinel = {"a": 1}
        llmrails_guardrails.rails_engine.events_history_cache = sentinel
        assert llmrails_guardrails.events_history_cache is sentinel

    def test_write_delegates_to_llmrails(self, llmrails_guardrails):
        """Writing guardrails.events_history_cache writes through to the LLMRails's plain attribute."""
        llmrails_guardrails.rails_engine.events_history_cache = {}
        sentinel = {"b": 2}
        llmrails_guardrails.events_history_cache = sentinel
        assert llmrails_guardrails.rails_engine.events_history_cache is sentinel

    def test_read_raises_under_iorails(self, iorails_guardrails):
        """Reading guardrails.events_history_cache on an IORails-backed facade raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r"IORails doesn't support events_history_cache"):
            _ = iorails_guardrails.events_history_cache

    def test_write_raises_under_iorails(self, iorails_guardrails):
        """Writing guardrails.events_history_cache on an IORails-backed facade raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r"IORails doesn't support events_history_cache"):
            iorails_guardrails.events_history_cache = {}
