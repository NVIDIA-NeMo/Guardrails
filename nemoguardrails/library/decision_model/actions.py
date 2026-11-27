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

"""Apply one named decision rule to input or output text."""

import logging
from collections.abc import Mapping
from typing import Any

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import HTTPClient
from nemoguardrails.library.decision_model.model import DecisionModelConfigurationError
from nemoguardrails.library.decision_model.rail_config import NoulQuestion
from nemoguardrails.library.decision_model.request import DecisionModelAuthError, DecisionModelError

log = logging.getLogger(__name__)


def _failure_outcome(source: str, rule: str, fail_open: bool, reason: str) -> RailOutcome:
    metadata = {"source": source, "rule": rule, "error": reason, "fail_open": fail_open}
    if fail_open:
        return RailOutcome.allow(metadata=metadata)
    return RailOutcome.block(reason=reason, metadata=metadata)


@action(is_system_action=True)
async def decision_model_check(
    source: str,
    text: str,
    rule: str,
    model_name: str,
    config: RailsConfig,
    llms: Mapping[str, Any],
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    """Ask the selected model one question and apply the rule's inclusive threshold.

    Configuration and credential errors raise regardless of fail_open. Provider
    failures block by default; authentication failures raise a system error.
    """
    if source not in ("input", "output"):
        raise ValueError("Decision model source must be input or output.")
    settings = config.rails.config.decision_model
    if settings is None or rule not in settings.rules:
        raise DecisionModelConfigurationError("The selected decision model rule is not configured.")
    model = llms.get(model_name)
    if not callable(getattr(model, "decide", None)):
        raise DecisionModelConfigurationError("The selected model must implement decide().")
    selected_rule = settings.rules[rule]
    metadata: dict[str, Any] = {
        "source": source,
        "rule": rule,
        "scores": {},
        "thresholds": {},
        "triggered": [],
        "violations": [],
    }
    # Tool-call-only and empty turns have no text to screen or send to a provider.
    if not (text or "").strip():
        return RailOutcome.allow(metadata=metadata)
    try:
        answer = await model.decide(
            name=rule,
            question=NoulQuestion(instructions=selected_rule.question.instructions),
            state={"user_message" if source == "input" else "bot_message": text},
            http_client=http_client,
        )
    except DecisionModelConfigurationError:
        raise
    except DecisionModelAuthError:
        log.error("Decision model authentication failed.")
        if not settings.fail_open:
            raise
        log.warning("fail_open is enabled; allowing text despite the authentication failure.")
        return _failure_outcome(source, rule, True, "Decision model authentication failed.")
    except DecisionModelError:
        # Adapter exceptions can carry provider content; use a fixed public error.
        reason = "Decision model assessment failed."
        log.warning("%s %s text.", reason, "Allowing" if settings.fail_open else "Blocking")
        return _failure_outcome(source, rule, settings.fail_open, reason)

    threshold = selected_rule.block_if.probability_at_least
    blocked = answer.probability >= threshold
    metadata.update(
        scores={rule: answer.probability},
        thresholds={rule: threshold},
        triggered=[rule] if blocked else [],
        violations=[{"question": rule, "score": answer.probability, "threshold": threshold}] if blocked else [],
    )
    if answer.model is not None:
        metadata["model"] = answer.model
    if blocked:
        return RailOutcome.block(reason=f"Decision model flagged: {rule}", metadata=metadata)
    return RailOutcome.allow(metadata=metadata)
