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

import os
import sys
from types import ModuleType
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailDecision, RailOutcome
from nemoguardrails.library.zscaler_aiguard import actions as zscaler_actions
from tests.utils import TestChat

input_rail_config = RailsConfig.from_content(
    yaml_content="""
        models: []
        rails:
          input:
            flows:
              - zscaler aiguard moderation on input
    """
)

output_rail_config = RailsConfig.from_content(
    yaml_content="""
        models: []
        rails:
          output:
            flows:
              - zscaler aiguard moderation on output
    """
)

both_rails_config = RailsConfig.from_content(
    yaml_content="""
        models: []
        rails:
          input:
            flows:
              - zscaler aiguard moderation on input
          output:
            flows:
              - zscaler aiguard moderation on output
    """
)

exceptions_config = RailsConfig.from_content(
    yaml_content="""
        models: []
        enable_rails_exceptions: true
        rails:
          input:
            flows:
              - zscaler aiguard moderation on input
          output:
            flows:
              - zscaler aiguard moderation on output
    """
)


def _allow_result(**overrides) -> RailOutcome:
    """Build a mock AI Guard ALLOW result."""
    base = {
        "action": "ALLOW",
        "severity": "NONE",
        "policy_name": "TestPolicy",
        "transaction_id": "txn-test-001",
        "detectors": {},
        "blocking_detectors": [],
    }
    base.update(overrides)
    return RailOutcome.allow(metadata=base)


def _block_result(direction="IN", blocking_detectors=None, reason=None, **overrides) -> RailOutcome:
    """Build a mock AI Guard BLOCK result."""
    if blocking_detectors is None:
        blocking_detectors = ["toxicity"]
    detectors = {name: {"action": "BLOCK", "triggered": True, "severity": "CRITICAL"} for name in blocking_detectors}
    target = "user prompt" if direction == "IN" else "LLM response"
    message = (
        f"Zscaler AI Guard blocked the {target}. "
        f"Severity: CRITICAL. Policy: TestPolicy. "
        f"Detectors: {', '.join(blocking_detectors)}. "
        f"Transaction: txn-test-002."
    )
    base = {
        "action": "BLOCK",
        "severity": "CRITICAL",
        "policy_name": "TestPolicy",
        "transaction_id": "txn-test-002",
        "detectors": detectors,
        "blocking_detectors": blocking_detectors,
    }
    base.update(overrides)
    return RailOutcome.block(reason=reason or message, metadata=base)


@pytest.fixture(autouse=True)
def _reset_sdk_client(monkeypatch):
    """Reset the module-level SDK client between tests."""
    import nemoguardrails.library.zscaler_aiguard.actions as mod

    mod._sdk_client = None
    mod._sdk_client_cloud = None
    for name in ("AIGUARD_API_KEY", "AIGUARD_CLOUD", "AIGUARD_POLICY_ID"):
        monkeypatch.delenv(name, raising=False)
    yield
    mod._sdk_client = None
    mod._sdk_client_cloud = None


# ---------------------------------------------------------------------------
# Integration tests (TestChat-based)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zscaler_aiguard_input_allowed():
    """Clean user input should pass through to the LLM."""
    config = both_rails_config

    chat = TestChat(
        config,
        llm_completions=["Hi! How can I help you today?"],
    )

    async def mock_action(text: Optional[str] = None, direction: str = "IN", **kwargs):
        return _allow_result()

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "Hello"
    await chat.bot_async("Hi! How can I help you today?")


@pytest.mark.asyncio
async def test_zscaler_aiguard_input_blocked():
    """Input containing sensitive data should be blocked (default path, no exception)."""
    config = both_rails_config

    chat = TestChat(
        config,
        llm_completions=["I don't know the answer to that."],
    )

    async def mock_action(text: Optional[str] = None, direction: str = "IN", **kwargs):
        if text and "AKIAIOSFODNN7EXAMPLE" in text:
            return _block_result(direction="IN", blocking_detectors=["credentials", "secrets"])
        return _allow_result()

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "My AWS key is AKIAIOSFODNN7EXAMPLE"
    await chat.bot_async("I don't know the answer to that.")


@pytest.mark.asyncio
async def test_zscaler_aiguard_output_allowed():
    """Clean LLM output should pass through to the user."""
    config = both_rails_config

    chat = TestChat(
        config,
        llm_completions=["The capital of France is Paris."],
    )

    async def mock_action(text: Optional[str] = None, direction: str = "IN", **kwargs):
        return _allow_result()

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "What is the capital of France?"
    await chat.bot_async("The capital of France is Paris.")


@pytest.mark.asyncio
async def test_zscaler_aiguard_output_blocked():
    """LLM output containing PII should be blocked (default path, no exception)."""
    config = both_rails_config

    chat = TestChat(
        config,
        llm_completions=[
            "John's SSN is 123-45-6789",
            "I don't know the answer to that.",
        ],
    )

    async def mock_action(text: Optional[str] = None, direction: str = "IN", **kwargs):
        if direction == "OUT" and text and "SSN" in text:
            return _block_result(direction="OUT", blocking_detectors=["pii"])
        return _allow_result()

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "Tell me about John"
    await chat.bot_async("I don't know the answer to that.")


@pytest.mark.asyncio
async def test_zscaler_aiguard_api_error_blocks():
    """API failures should trigger fail-closed behavior (default path, no exception)."""
    config = both_rails_config

    chat = TestChat(
        config,
        llm_completions=["I don't know the answer to that."],
    )

    async def mock_action(text: Optional[str] = None, direction: str = "IN", **kwargs):
        return RailOutcome.failure(
            reason="Zscaler AI Guard blocked the user prompt. Severity: UNKNOWN. Policy: unknown.",
            metadata={
                "action": "BLOCK",
                "severity": "UNKNOWN",
                "detectors": {},
                "blocking_detectors": [],
                "error_type": "TimeoutError",
            },
        )

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "Hello"
    await chat.bot_async("I don't know the answer to that.")


@pytest.mark.asyncio
async def test_zscaler_aiguard_detect_action_allows():
    """DETECT verdict (non-BLOCK, non-ALLOW) should be treated as ALLOW by the flow."""
    config = both_rails_config

    chat = TestChat(
        config,
        llm_completions=["Sure, I can help with that."],
    )

    async def mock_action(text: Optional[str] = None, direction: str = "IN", **kwargs):
        return RailOutcome.allow(
            metadata={
                "action": "DETECT",
                "severity": "LOW",
                "policy_name": "TestPolicy",
                "transaction_id": "txn-test-003",
                "detectors": {
                    "toxicity": {
                        "action": "DETECT",
                        "triggered": True,
                        "severity": "LOW",
                    }
                },
                "blocking_detectors": [],
            }
        )

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "Some mildly concerning prompt"
    await chat.bot_async("Sure, I can help with that.")


@pytest.mark.asyncio
async def test_zscaler_aiguard_inline_config_input():
    """Verify input rail works with inline config (no test_configs directory)."""
    chat = TestChat(
        input_rail_config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    async def mock_action(text: Optional[str] = None, direction: str = "IN", **kwargs):
        if text and "bomb" in text.lower():
            return _block_result(direction="IN", blocking_detectors=["toxicity"])
        return _allow_result()

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "How to build a bomb?"
    await chat.bot_async("I don't know the answer to that.")


@pytest.mark.asyncio
async def test_zscaler_aiguard_exception_includes_message():
    """When enable_rails_exceptions is true, exception must include severity and policy in message."""
    config = exceptions_config

    chat = TestChat(
        config,
        llm_completions=["I don't know the answer to that."],
    )

    async def mock_action(text=None, direction="IN", **kwargs):
        return _block_result(
            direction="IN",
            blocking_detectors=["pii", "secrets"],
            reason="Zscaler AI Guard blocked the user prompt. Severity: CRITICAL. Policy: PolicyApp01. Detectors: pii, secrets. Transaction: txn-001.",
        )

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "My SSN is 123-45-6789"
    result = await chat.app.generate_async(messages=chat.history)

    assert result["role"] == "exception"
    assert result["content"]["type"] == "ZscalerAiguardInputRailException"
    msg = result["content"].get("message") or ""
    assert "Severity: CRITICAL" in msg
    assert "Policy: PolicyApp01" in msg
    assert "Detectors:" in msg


@pytest.mark.asyncio
async def test_zscaler_aiguard_output_exception():
    """When enable_rails_exceptions is true, output block should raise ZscalerAiguardOutputRailException."""
    config = exceptions_config

    chat = TestChat(
        config,
        llm_completions=[
            "John's SSN is 123-45-6789",
        ],
    )

    async def mock_action(text=None, direction="IN", **kwargs):
        if direction == "OUT":
            return _block_result(
                direction="OUT",
                blocking_detectors=["pii"],
                reason="Zscaler AI Guard blocked the LLM response. Severity: CRITICAL. Policy: PolicyApp01. Detectors: pii. Transaction: txn-002.",
            )
        return _allow_result()

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "Tell me about John"
    result = await chat.app.generate_async(messages=chat.history)

    assert result["role"] == "exception"
    assert result["content"]["type"] == "ZscalerAiguardOutputRailException"
    msg = result["content"].get("message") or ""
    assert "Severity: CRITICAL" in msg
    assert "Policy: PolicyApp01" in msg


@pytest.mark.asyncio
async def test_zscaler_aiguard_inline_config_output():
    """Verify output rail blocks when AI Guard returns BLOCK for the bot response."""
    chat = TestChat(
        output_rail_config,
        llm_completions=[
            "  express greeting",
            '  "Here is some content"',
        ],
    )

    async def mock_action(text: Optional[str] = None, direction: str = "IN", **kwargs):
        if direction == "OUT":
            return _block_result(direction="OUT", blocking_detectors=["secrets"])
        return _allow_result()

    chat.app.register_action(mock_action, "call_zscaler_aiguard_api")

    chat >> "Give me a secret key"
    await chat.bot_async("I don't know the answer to that.")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_zscaler_aiguard_get_attr_dict():
    """_get_attr should work with both dicts and objects."""
    from nemoguardrails.library.zscaler_aiguard.actions import _get_attr

    assert _get_attr({"key": "value"}, "key") == "value"
    assert _get_attr({"key": "value"}, "missing", "default") == "default"

    class SimpleObj:
        pass

    obj = SimpleObj()
    obj.key = "value"
    assert _get_attr(obj, "key") == "value"
    assert _get_attr(obj, "missing", "default") == "default"


@pytest.mark.unit
def test_zscaler_aiguard_build_block_message():
    """_build_block_message should produce a readable message with all context."""
    from nemoguardrails.library.zscaler_aiguard.actions import _build_block_message

    msg = _build_block_message(
        direction="IN",
        severity="CRITICAL",
        policy_name="PolicyApp01",
        blocking_detectors=["pii", "secrets"],
        transaction_id="txn-abc-123",
    )
    assert "Zscaler AI Guard blocked the user prompt." in msg
    assert "Severity: CRITICAL." in msg
    assert "Policy: PolicyApp01." in msg
    assert "Detectors: pii, secrets." in msg
    assert "Transaction: txn-abc-123." in msg

    msg_out = _build_block_message(
        direction="OUT",
        severity="HIGH",
        policy_name="OutputPolicy",
        blocking_detectors=["toxicity"],
    )
    assert "Zscaler AI Guard blocked the LLM response." in msg_out
    assert "Severity: HIGH." in msg_out
    assert "Policy: OutputPolicy." in msg_out
    assert "Detectors: toxicity." in msg_out
    assert "Transaction:" not in msg_out

    msg_no_detectors = _build_block_message(
        direction="IN",
        severity="UNKNOWN",
        policy_name="unknown",
        blocking_detectors=[],
    )
    assert "Detectors:" not in msg_no_detectors


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_empty_text():
    """Action should return ALLOW for empty text without calling the API."""
    from nemoguardrails.library.zscaler_aiguard.actions import (
        call_zscaler_aiguard_api,
    )

    result = await call_zscaler_aiguard_api(text=None, direction="IN")
    assert result.decision is RailDecision.ALLOW
    assert result.metadata["action"] == "ALLOW"
    assert result.reason is None

    result = await call_zscaler_aiguard_api(text="", direction="IN")
    assert result.decision is RailDecision.ALLOW
    assert result.metadata["action"] == "ALLOW"
    assert result.reason is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_scan_success():
    """Action should correctly parse an ALLOW result from the SDK."""
    with patch("nemoguardrails.library.zscaler_aiguard.actions._scan_sync") as mock_scan:
        mock_scan.return_value = {
            "action": "ALLOW",
            "severity": "NONE",
            "policyName": "DefaultPolicy",
            "transactionId": "txn-abc-123",
            "detectorResponses": {
                "toxicity": {
                    "action": "ALLOW",
                    "triggered": False,
                    "severity": "NONE",
                },
                "pii": {
                    "action": "ALLOW",
                    "triggered": False,
                    "severity": "NONE",
                },
            },
        }

        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        result = await call_zscaler_aiguard_api(text="Hello world", direction="IN")

        assert result.decision is RailDecision.ALLOW
        assert result.metadata["action"] == "ALLOW"
        assert result.metadata["policy_name"] == "DefaultPolicy"
        assert result.metadata["transaction_id"] == "txn-abc-123"
        assert "toxicity" in result.metadata["detectors"]
        assert "pii" in result.metadata["detectors"]
        assert result.metadata["blocking_detectors"] == []
        assert result.reason is None
        mock_scan.assert_called_once_with("Hello world", "IN", None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_scan_block():
    """Action should correctly parse a BLOCK result from the SDK."""
    with patch("nemoguardrails.library.zscaler_aiguard.actions._scan_sync") as mock_scan:
        mock_scan.return_value = {
            "action": "BLOCK",
            "severity": "CRITICAL",
            "policyName": "StrictPolicy",
            "transactionId": "txn-xyz-789",
            "detectorResponses": {
                "credentials": {
                    "action": "BLOCK",
                    "triggered": True,
                    "severity": "CRITICAL",
                },
                "toxicity": {
                    "action": "ALLOW",
                    "triggered": False,
                    "severity": "NONE",
                },
            },
        }

        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        result = await call_zscaler_aiguard_api(text="My key is AKIAIOSFODNN7EXAMPLE", direction="IN")

        assert result.decision is RailDecision.BLOCK
        assert result.metadata["action"] == "BLOCK"
        assert result.metadata["severity"] == "CRITICAL"
        assert result.metadata["blocking_detectors"] == ["credentials"]
        assert result.metadata["detectors"]["credentials"]["triggered"] is True
        assert result.metadata["detectors"]["toxicity"]["triggered"] is False
        assert "Severity: CRITICAL" in result.reason
        assert "Policy: StrictPolicy" in result.reason
        assert "Detectors: credentials" in result.reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_unknown_verdict_blocks():
    """An unrecognized vendor verdict should fail closed."""
    with patch("nemoguardrails.library.zscaler_aiguard.actions._scan_sync") as mock_scan:
        mock_scan.return_value = {
            "action": "REVIEW",
            "severity": "UNKNOWN",
            "policyName": "DefaultPolicy",
            "transactionId": "txn-review-001",
            "detectorResponses": {},
        }

        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        result = await call_zscaler_aiguard_api(text="Hello", direction="IN")

        assert result.decision is RailDecision.BLOCK
        assert result.metadata["action"] == "REVIEW"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_exception_fail_closed():
    """Action should return BLOCK when the SDK raises an exception (fail-closed)."""
    with patch("nemoguardrails.library.zscaler_aiguard.actions._scan_sync") as mock_scan:
        mock_scan.side_effect = RuntimeError("Connection refused")

        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        result = await call_zscaler_aiguard_api(text="Hello", direction="IN")

        assert result.decision is RailDecision.BLOCK
        assert result.failed is True
        assert result.metadata["action"] == "BLOCK"
        assert result.metadata["severity"] == "UNKNOWN"
        assert result.metadata["policy_name"] == "unknown"
        assert result.metadata["transaction_id"] is None
        assert result.metadata["error_type"] == "RuntimeError"
        assert result.reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_none_result_blocks():
    """Action should return BLOCK when the SDK returns None (fail-closed)."""
    with patch("nemoguardrails.library.zscaler_aiguard.actions._scan_sync") as mock_scan:
        mock_scan.return_value = None

        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        result = await call_zscaler_aiguard_api(text="Hello", direction="IN")

        assert result.decision is RailDecision.BLOCK
        assert result.metadata["action"] == "BLOCK"
        assert result.metadata["severity"] == "UNKNOWN"
        assert result.metadata["policy_name"] == "unknown"
        assert result.metadata["transaction_id"] is None
        assert result.reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_policy_id_param():
    """Action should pass policy_id to _scan_sync when provided."""
    with patch("nemoguardrails.library.zscaler_aiguard.actions._scan_sync") as mock_scan:
        mock_scan.return_value = {
            "action": "ALLOW",
            "severity": "NONE",
            "policyName": "CustomPolicy",
            "transactionId": "txn-policy-001",
            "detectorResponses": {},
        }

        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        result = await call_zscaler_aiguard_api(text="Test content", direction="IN", policy_id=900)

        assert result.decision is RailDecision.ALLOW
        mock_scan.assert_called_once_with("Test content", "IN", 900)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_policy_id_env():
    """Action should read AIGUARD_POLICY_ID from environment when no param given."""
    with (
        patch("nemoguardrails.library.zscaler_aiguard.actions._scan_sync") as mock_scan,
        patch.dict(os.environ, {"AIGUARD_POLICY_ID": "1234"}),
    ):
        mock_scan.return_value = {
            "action": "ALLOW",
            "severity": "NONE",
            "policyName": "EnvPolicy",
            "transactionId": "txn-env-001",
            "detectorResponses": {},
        }

        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        result = await call_zscaler_aiguard_api(text="Test content", direction="IN")

        assert result.decision is RailDecision.ALLOW
        mock_scan.assert_called_once_with("Test content", "IN", 1234)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_policy_id_param_overrides_env():
    """Explicit policy_id param should take precedence over env var."""
    with (
        patch("nemoguardrails.library.zscaler_aiguard.actions._scan_sync") as mock_scan,
        patch.dict(os.environ, {"AIGUARD_POLICY_ID": "1234"}),
    ):
        mock_scan.return_value = {
            "action": "ALLOW",
            "severity": "NONE",
            "policyName": "ParamPolicy",
            "transactionId": "txn-override-001",
            "detectorResponses": {},
        }

        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        result = await call_zscaler_aiguard_api(text="Test content", direction="IN", policy_id=5678)

        assert result.decision is RailDecision.ALLOW
        mock_scan.assert_called_once_with("Test content", "IN", 5678)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_aiguard_action_invalid_policy_id_env():
    """Invalid AIGUARD_POLICY_ID env var should raise ValueError (fail-closed)."""
    with patch.dict(os.environ, {"AIGUARD_POLICY_ID": "not-a-number"}):
        from nemoguardrails.library.zscaler_aiguard.actions import (
            call_zscaler_aiguard_api,
        )

        with pytest.raises(ValueError, match="AIGUARD_POLICY_ID"):
            await call_zscaler_aiguard_api(text="Test content", direction="IN")


@pytest.fixture
def sdk_helper(monkeypatch):
    helper = MagicMock(name="LegacyZGuardClientHelper")
    legacy = ModuleType("zscaler.zaiguard.legacy")
    setattr(legacy, "LegacyZGuardClientHelper", helper)
    monkeypatch.setitem(sys.modules, "zscaler", ModuleType("zscaler"))
    monkeypatch.setitem(sys.modules, "zscaler.zaiguard", ModuleType("zscaler.zaiguard"))
    monkeypatch.setitem(sys.modules, "zscaler.zaiguard.legacy", legacy)
    monkeypatch.setenv("AIGUARD_API_KEY", "test-api-key")
    return helper


@pytest.mark.unit
def test_zscaler_sdk_client_initialization_and_cache(sdk_helper, monkeypatch):
    client = zscaler_actions._get_sdk_client()

    sdk_helper.assert_called_once_with(cloud="us1")
    assert client is sdk_helper.return_value
    assert zscaler_actions._get_sdk_client() is client
    sdk_helper.assert_called_once()

    monkeypatch.setenv("AIGUARD_CLOUD", "eu1")
    zscaler_actions._get_sdk_client()

    assert sdk_helper.call_count == 2
    sdk_helper.assert_called_with(cloud="eu1")


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("policy_id", [None, 900])
@pytest.mark.parametrize("direction", ["IN", "OUT"])
async def test_zscaler_sdk_policy_calls(sdk_helper, policy_id, direction):
    detection = sdk_helper.return_value.policy_detection
    detection.resolve_and_execute_policy.return_value = ({"action": "ALLOW"}, None, None)
    detection.execute_policy.return_value = ({"action": "BLOCK"}, None, None)

    result = await zscaler_actions.call_zscaler_aiguard_api(
        text="Test content", direction=direction, policy_id=policy_id
    )

    if policy_id is None:
        detection.resolve_and_execute_policy.assert_called_once_with(content="Test content", direction=direction)
        detection.execute_policy.assert_not_called()
        assert result.decision is RailDecision.ALLOW
    else:
        detection.execute_policy.assert_called_once_with(
            content="Test content", direction=direction, policy_id=policy_id
        )
        detection.resolve_and_execute_policy.assert_not_called()
        assert result.decision is RailDecision.BLOCK
    assert result.failed is False


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("policy_id", [None, 900])
async def test_zscaler_sdk_error_tuple_returns_failure(sdk_helper, policy_id, caplog):
    detection = sdk_helper.return_value.policy_detection
    response = (None, None, "sensitive provider error detail")
    detection.resolve_and_execute_policy.return_value = response
    detection.execute_policy.return_value = response

    result = await zscaler_actions.call_zscaler_aiguard_api(text="Test content", policy_id=policy_id)

    assert result.is_blocked
    assert result.failed is True
    assert result.metadata["error_type"] == "RuntimeError"
    assert "sensitive provider error detail" not in repr(result)
    assert "sensitive provider error detail" not in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError, ConnectionError, ValueError])
async def test_zscaler_sdk_exceptions_return_failure(sdk_helper, error_type, caplog):
    sdk_helper.return_value.policy_detection.resolve_and_execute_policy.side_effect = error_type(
        "sensitive provider error detail"
    )

    result = await zscaler_actions.call_zscaler_aiguard_api(text="Test content")

    assert result.is_blocked
    assert result.failed is True
    assert result.metadata["error_type"] == error_type.__name__
    assert "sensitive provider error detail" not in repr(result)
    assert "sensitive provider error detail" not in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_sdk_missing_dependency_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "zscaler.zaiguard.legacy", None)

    with pytest.raises(ImportError, match="pip install zscaler-sdk-python"):
        await zscaler_actions.call_zscaler_aiguard_api(text="Test content")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zscaler_sdk_missing_api_key_raises(sdk_helper, monkeypatch):
    monkeypatch.delenv("AIGUARD_API_KEY")

    with pytest.raises(EnvironmentError, match="AIGUARD_API_KEY"):
        await zscaler_actions.call_zscaler_aiguard_api(text="Test content")

    sdk_helper.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scan_fails", [False, True])
async def test_zscaler_iorails_distinguishes_scan_failures_from_policy_blocks(sdk_helper, monkeypatch, scan_fails):
    from nemoguardrails.guardrails.iorails import INTERNAL_ERROR_MESSAGE, REFUSAL_MESSAGE, IORails
    from nemoguardrails.rails.llm.options import RailStatus

    monkeypatch.setattr(
        "nemoguardrails.guardrails.compiled_rail._is_installed", lambda name: name == "zscaler-sdk-python"
    )
    detection = sdk_helper.return_value.policy_detection.resolve_and_execute_policy
    if scan_fails:
        detection.return_value = (None, None, "sensitive provider error detail")
    else:
        detection.return_value = ({"action": "BLOCK"}, None, None)

    async with IORails(input_rail_config, _report_usage=False) as rails:
        result = await rails.check_async([{"role": "user", "content": "Test content"}])

    assert result.status == RailStatus.BLOCKED
    assert result.rail == "zscaler aiguard moderation on input"
    assert result.content == (INTERNAL_ERROR_MESSAGE if scan_fails else REFUSAL_MESSAGE)
    assert "sensitive provider error detail" not in repr(result)


@pytest.fixture
def colang2_config():
    return RailsConfig.from_content(
        yaml_content="""
            colang_version: "2.x"
            models: []
        """,
        colang_content="""
            import guardrails
            import nemoguardrails.library.zscaler_aiguard

            flow input rails $input_text
                zscaler aiguard moderation on input

            flow output rails $output_text
                zscaler aiguard moderation on output

            flow main
                user said "Hello"
                bot say "Ordinary response"
        """,
    )


@pytest.mark.unit
def test_zscaler_colang2_scans_actual_input_and_output(sdk_helper, colang2_config):
    scan = sdk_helper.return_value.policy_detection.resolve_and_execute_policy
    scan.return_value = ({"action": "ALLOW"}, None, None)
    chat = TestChat(colang2_config, llm_completions=[])

    chat >> "Hello"
    chat << "Ordinary response"

    assert scan.call_args_list == [
        call(content="Hello", direction="IN"),
        call(content="Ordinary response", direction="OUT"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("blocked_direction", ["IN", "OUT"])
def test_zscaler_colang2_block_returns_refusal(sdk_helper, colang2_config, blocked_direction):
    def scan(content, direction):
        return ({"action": "BLOCK" if direction == blocked_direction else "ALLOW"}, None, None)

    detection = sdk_helper.return_value.policy_detection.resolve_and_execute_policy
    detection.side_effect = scan
    chat = TestChat(colang2_config, llm_completions=[])

    chat >> "Hello"
    chat << "I'm sorry, I can't respond to that."

    expected_content = "Hello" if blocked_direction == "IN" else "Ordinary response"
    assert call(content=expected_content, direction=blocked_direction) in detection.call_args_list
