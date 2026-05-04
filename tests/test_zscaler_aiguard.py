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
from typing import Optional
from unittest.mock import patch

import pytest

from nemoguardrails import RailsConfig
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


def _allow_result(**overrides):
    """Build a mock AI Guard ALLOW result."""
    base = {
        "action": "ALLOW",
        "severity": "NONE",
        "policy_name": "TestPolicy",
        "transaction_id": "txn-test-001",
        "detectors": {},
        "blocking_detectors": [],
        "message": "",
    }
    base.update(overrides)
    return base


def _block_result(direction="IN", blocking_detectors=None, **overrides):
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
        "message": message,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _reset_sdk_client():
    """Reset the module-level SDK client between tests."""
    import nemoguardrails.library.zscaler_aiguard.actions as mod

    mod._sdk_client = None
    mod._sdk_client_cloud = None
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
        return {
            "action": "BLOCK",
            "severity": "UNKNOWN",
            "detectors": {},
            "blocking_detectors": [],
            "error": "Connection timeout",
            "message": "Zscaler AI Guard blocked the user prompt. Severity: UNKNOWN. Policy: unknown.",
        }

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
        return {
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
            "message": "",
        }

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
            message="Zscaler AI Guard blocked the user prompt. Severity: CRITICAL. Policy: PolicyApp01. Detectors: pii, secrets. Transaction: txn-001.",
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
                message="Zscaler AI Guard blocked the LLM response. Severity: CRITICAL. Policy: PolicyApp01. Detectors: pii. Transaction: txn-002.",
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
    assert result["action"] == "ALLOW"
    assert result["message"] == ""

    result = await call_zscaler_aiguard_api(text="", direction="IN")
    assert result["action"] == "ALLOW"
    assert result["message"] == ""


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

        assert result["action"] == "ALLOW"
        assert result["policy_name"] == "DefaultPolicy"
        assert result["transaction_id"] == "txn-abc-123"
        assert "toxicity" in result["detectors"]
        assert "pii" in result["detectors"]
        assert result["blocking_detectors"] == []
        assert result["message"] == ""
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

        assert result["action"] == "BLOCK"
        assert result["severity"] == "CRITICAL"
        assert result["blocking_detectors"] == ["credentials"]
        assert result["detectors"]["credentials"]["triggered"] is True
        assert result["detectors"]["toxicity"]["triggered"] is False
        assert "Severity: CRITICAL" in result["message"]
        assert "Policy: StrictPolicy" in result["message"]
        assert "Detectors: credentials" in result["message"]


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

        assert result["action"] == "BLOCK"
        assert result["severity"] == "UNKNOWN"
        assert result["policy_name"] == "unknown"
        assert result["transaction_id"] is None
        assert "Connection refused" in result["error"]
        assert "message" in result
        assert result["message"] != ""


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

        assert result["action"] == "BLOCK"
        assert result["severity"] == "UNKNOWN"
        assert result["policy_name"] == "unknown"
        assert result["transaction_id"] is None
        assert "message" in result
        assert result["message"] != ""


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

        assert result["action"] == "ALLOW"
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

        assert result["action"] == "ALLOW"
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

        assert result["action"] == "ALLOW"
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
