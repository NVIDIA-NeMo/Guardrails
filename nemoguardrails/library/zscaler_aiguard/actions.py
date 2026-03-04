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

"""
Zscaler AI Guard integration for NeMo Guardrails.

Scans prompts and LLM responses using the Zscaler AI Guard DAS API
(resolve-and-execute-policy) via the zscaler-sdk-python SDK.

Required environment variables:
    AIGUARD_API_KEY  - Zscaler AI Guard API key (Bearer token)
    AIGUARD_CLOUD    - Cloud region, e.g. us1, us2, eu1 (default: us1)
"""

import asyncio
import logging
import os
from typing import Any, Optional

from nemoguardrails.actions import action

log = logging.getLogger(__name__)

_sdk_client = None


def _get_sdk_client():
    """Lazily initialise the Zscaler AI Guard SDK client."""
    global _sdk_client
    if _sdk_client is None:
        try:
            from zscaler.zaiguard.legacy import LegacyZGuardClientHelper
        except ImportError:
            raise ImportError(
                "zscaler-sdk-python is required for the Zscaler AI Guard integration. "
                "Install it with: pip install zscaler-sdk-python"
            )

        cloud = os.environ.get("AIGUARD_CLOUD", "us1")
        _sdk_client = LegacyZGuardClientHelper(cloud=cloud)
    return _sdk_client


def _get_attr(obj, name, default=None):
    """Access an attribute by name from either a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _scan_sync(content: str, direction: str):
    """Call the AI Guard API synchronously (wrapped with asyncio.to_thread)."""
    client = _get_sdk_client()
    result, _response, error = client.policy_detection.resolve_and_execute_policy(
        content=content,
        direction=direction,
    )
    if error:
        raise RuntimeError(f"AI Guard API error: {error}")
    return result


def call_zscaler_aiguard_api_mapping(result: dict) -> bool:
    """
    Output mapping for call_zscaler_aiguard_api.

    Returns True (block) when the AI Guard policy action is anything
    other than ALLOW, implementing fail-closed semantics.
    """
    action_val = str(result.get("action", "BLOCK")).upper()
    return action_val != "ALLOW"


@action(is_system_action=True, output_mapping=call_zscaler_aiguard_api_mapping)
async def call_zscaler_aiguard_api(
    text: Optional[str] = None,
    direction: str = "IN",
    **kwargs,
) -> dict[str, Any]:
    """
    Scan content using Zscaler AI Guard.

    Args:
        text: Content to scan (user prompt or bot response).
        direction: "IN" for user prompts, "OUT" for bot responses.

    Returns:
        Dict containing:
            action       - Policy verdict (ALLOW / BLOCK / DETECT)
            severity     - Severity level of the detection
            policy_name  - Name of the policy that was evaluated
            transaction_id - Unique transaction ID for debugging
            detectors    - Dict of detector names to their verdicts
    """
    if not text:
        return {"action": "ALLOW", "severity": "NONE", "detectors": {}}

    log.debug("AI Guard scanning %s content (%d chars)", direction, len(text))

    try:
        result = await asyncio.to_thread(_scan_sync, text, direction)

        if result is None:
            log.warning("AI Guard returned None — blocking by default")
            return {"action": "BLOCK", "severity": "UNKNOWN", "detectors": {}}

        action_val = str(_get_attr(result, "action", "BLOCK")).upper()
        severity = _get_attr(result, "severity", "unknown")
        policy_name = _get_attr(result, "policy_name") or _get_attr(
            result, "policyName", "unknown"
        )
        transaction_id = _get_attr(result, "transaction_id") or _get_attr(
            result, "transactionId"
        )

        detector_responses = (
            _get_attr(result, "detector_responses")
            or _get_attr(result, "detectorResponses")
            or {}
        )
        detectors = {}
        blocking_detectors = []
        for name, det in (
            detector_responses.items() if isinstance(detector_responses, dict) else []
        ):
            det_action = str(_get_attr(det, "action", "unknown")).upper()
            det_triggered = _get_attr(det, "triggered", False)
            detectors[name] = {
                "action": det_action,
                "triggered": det_triggered,
                "severity": _get_attr(det, "severity"),
            }
            if det_action == "BLOCK":
                blocking_detectors.append(name)

        if action_val != "ALLOW":
            log.info(
                "AI Guard BLOCKED [txn=%s, policy=%s, severity=%s, detectors=%s]",
                transaction_id,
                policy_name,
                severity,
                blocking_detectors,
            )
        else:
            log.debug(
                "AI Guard ALLOWED [txn=%s, policy=%s]",
                transaction_id,
                policy_name,
            )

        return {
            "action": action_val,
            "severity": severity,
            "policy_name": policy_name,
            "transaction_id": transaction_id,
            "detectors": detectors,
            "blocking_detectors": blocking_detectors,
        }

    except Exception as e:
        log.error("AI Guard scan failed: %s — %s", type(e).__name__, e)
        return {
            "action": "BLOCK",
            "severity": "UNKNOWN",
            "detectors": {},
            "error": str(e),
        }
