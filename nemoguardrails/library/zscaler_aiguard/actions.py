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
via the zscaler-sdk-python SDK.

By default, uses resolve-and-execute-policy (automatic policy selection).
Set AIGUARD_POLICY_ID to use execute-policy with a specific policy.

Required environment variables:
    AIGUARD_API_KEY  - Zscaler AI Guard API key (Bearer token)
    AIGUARD_CLOUD    - Cloud region, e.g. us1, us2, eu1 (default: us1)

Optional environment variables:
    AIGUARD_POLICY_ID - Specific policy ID to use (default: auto-resolved)
"""

import asyncio
import logging
import os
from typing import Any, Optional

from nemoguardrails.actions import action

log = logging.getLogger(__name__)

_sdk_client = None
_sdk_client_cloud = None


def _get_sdk_client():
    """Lazily initialise the Zscaler AI Guard SDK client."""
    global _sdk_client, _sdk_client_cloud

    cloud = os.environ.get("AIGUARD_CLOUD", "us1")

    if _sdk_client is None or _sdk_client_cloud != cloud:
        try:
            from zscaler.zaiguard.legacy import LegacyZGuardClientHelper
        except ImportError:
            raise ImportError(
                "zscaler-sdk-python is required for the Zscaler AI Guard integration. "
                "Install it with: pip install zscaler-sdk-python"
            )

        if not os.environ.get("AIGUARD_API_KEY"):
            raise EnvironmentError(
                "AIGUARD_API_KEY environment variable is required for the Zscaler AI Guard integration."
            )

        _sdk_client = LegacyZGuardClientHelper(cloud=cloud)
        _sdk_client_cloud = cloud
    return _sdk_client


def _get_attr(obj, name, default=None):
    """Access an attribute by name from either a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _scan_sync(content: str, direction: str, policy_id: Optional[int] = None):
    """Call the AI Guard API synchronously (wrapped with asyncio.to_thread)."""
    client = _get_sdk_client()

    if policy_id is not None:
        result, _response, error = client.policy_detection.execute_policy(
            content=content,
            direction=direction,
            policy_id=policy_id,
        )
    else:
        result, _response, error = client.policy_detection.resolve_and_execute_policy(
            content=content,
            direction=direction,
        )

    if error:
        raise RuntimeError(f"AI Guard API error: {error}")
    return result


def _build_block_message(
    direction: str,
    severity: str,
    policy_name: str,
    blocking_detectors: list,
    transaction_id: Optional[str] = None,
) -> str:
    """Build a human-readable block message with full context."""
    target = "user prompt" if direction == "IN" else "LLM response"
    parts = [f"Zscaler AI Guard blocked the {target}."]
    parts.append(f"Severity: {severity}.")
    parts.append(f"Policy: {policy_name}.")
    if blocking_detectors:
        parts.append(f"Detectors: {', '.join(blocking_detectors)}.")
    if transaction_id:
        parts.append(f"Transaction: {transaction_id}.")
    return " ".join(parts)


@action(is_system_action=True)
async def call_zscaler_aiguard_api(
    text: Optional[str] = None,
    direction: str = "IN",
    policy_id: Optional[int] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Scan content using Zscaler AI Guard.

    Args:
        text: Content to scan (user prompt or bot response).
        direction: "IN" for user prompts, "OUT" for bot responses.
        policy_id: Optional policy ID. If provided, calls execute_policy
            instead of resolve_and_execute_policy. Can also be set via
            AIGUARD_POLICY_ID environment variable.

    Returns:
        Dict containing:
            action       - Policy verdict (ALLOW / BLOCK / DETECT)
            severity     - Severity level of the detection
            policy_name  - Name of the policy that was evaluated
            transaction_id - Unique transaction ID for debugging
            detectors    - Dict of detector names to their verdicts
            message      - Pre-built human-readable message for exceptions
    """
    if not text:
        return {
            "action": "ALLOW",
            "severity": "NONE",
            "policy_name": "none",
            "transaction_id": None,
            "detectors": {},
            "blocking_detectors": [],
            "message": "",
        }

    effective_policy_id = policy_id
    if effective_policy_id is None:
        env_policy_id = os.environ.get("AIGUARD_POLICY_ID")
        if env_policy_id:
            try:
                effective_policy_id = int(env_policy_id)
            except ValueError:
                raise ValueError(
                    f"AIGUARD_POLICY_ID={env_policy_id!r} is not a valid integer. "
                    "Fix the environment variable or remove it to use automatic policy resolution."
                )

    log.debug("AI Guard scanning %s content (%d chars)", direction, len(text))

    try:
        result = await asyncio.to_thread(_scan_sync, text, direction, effective_policy_id)

        if result is None:
            log.warning("AI Guard returned None — blocking by default")
            return {
                "action": "BLOCK",
                "severity": "UNKNOWN",
                "policy_name": "unknown",
                "transaction_id": None,
                "detectors": {},
                "blocking_detectors": [],
                "message": _build_block_message(direction, "UNKNOWN", "unknown", []),
            }

        action_val = str(_get_attr(result, "action", "BLOCK")).upper()
        severity = _get_attr(result, "severity", "unknown")
        policy_name = _get_attr(result, "policy_name") or _get_attr(result, "policyName", "unknown")
        transaction_id = _get_attr(result, "transaction_id") or _get_attr(result, "transactionId")

        detector_responses = _get_attr(result, "detector_responses") or _get_attr(result, "detectorResponses") or {}
        detectors = {}
        blocking_detectors = []
        if not isinstance(detector_responses, dict):
            log.warning(
                "AI Guard returned non-dict detector_responses (%s) — skipping detector parsing",
                type(detector_responses).__name__,
            )
            detector_responses = {}
        for name, det in detector_responses.items():
            det_action = str(_get_attr(det, "action", "unknown")).upper()
            det_triggered = _get_attr(det, "triggered", False)
            detectors[name] = {
                "action": det_action,
                "triggered": det_triggered,
                "severity": _get_attr(det, "severity"),
            }
            if det_action == "BLOCK":
                blocking_detectors.append(name)

        if action_val == "BLOCK":
            message = _build_block_message(direction, severity, policy_name, blocking_detectors, transaction_id)
            log.info("AI Guard BLOCKED: %s", message)
        elif action_val != "ALLOW":
            message = ""
            log.info(
                "AI Guard %s [txn=%s, policy=%s, severity=%s]",
                action_val,
                transaction_id,
                policy_name,
                severity,
            )
        else:
            message = ""
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
            "message": message,
        }

    except (ImportError, EnvironmentError, ValueError):
        raise
    except Exception as e:
        log.error("AI Guard scan failed: %s — %s", type(e).__name__, e)
        message = _build_block_message(direction, "UNKNOWN", "unknown", [])
        return {
            "action": "BLOCK",
            "severity": "UNKNOWN",
            "policy_name": "unknown",
            "transaction_id": None,
            "detectors": {},
            "blocking_detectors": [],
            "error": str(e),
            "message": message,
        }
