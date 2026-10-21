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

"""Text assessment using a configured decision model service."""

import logging
import os
from typing import Any, Dict, Optional

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import HTTPClient
from nemoguardrails.library.decision_model.rail_config import DecisionModelConfig
from nemoguardrails.library.decision_model.request import decision_model_request

log = logging.getLogger(__name__)

VALID_SOURCES = ("input", "output")


def _failure_outcome(fail_open: bool, reason: str) -> RailOutcome:
    metadata = {"error": reason, "fail_open": fail_open}
    if fail_open:
        return RailOutcome.allow(metadata=metadata)
    return RailOutcome.block(reason=reason, metadata=metadata)


@action(is_system_action=True)
async def decision_model_check(
    source: str,
    text: str,
    config: RailsConfig,
    http_client: Optional[HTTPClient] = None,
    **kwargs,
) -> RailOutcome:
    """Ask the decision model the configured yes/no questions about ``text`` and block on a high probability.

    Args:
        source: Either "input" or "output"; selects which configured questions to ask.
        text: The text to assess.
        config: The rails configuration object.

    Returns:
        RailOutcome.block() if any question's probability reaches its threshold, or if the decision model
        cannot be reached and ``fail_open`` is false; RailOutcome.allow() otherwise. Metadata
        holds the per-question ``scores``, ``triggered`` question ids, and the resolved
        ``model`` when supplied by the provider, never the text.

    Raises:
        ValueError: If ``source`` is invalid, the rail is unconfigured, or its configured
            API-key environment variable is unset. These are configuration errors.
    """
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Decision model can only be defined in the following flows: {list(VALID_SOURCES)}. Got '{source}'."
        )

    decision_model_config: Optional[DecisionModelConfig] = config.rails.config.decision_model
    if decision_model_config is None:
        raise ValueError("rails.config.decision_model must configure server_endpoint and model.")
    questions = getattr(decision_model_config, source).questions
    # Nothing to ask, or nothing to assess (e.g. an empty or tool-call-only assistant turn): skip the
    # API call and the API-key requirement rather than sending an empty state.
    if not questions or not (text or "").strip():
        return RailOutcome.allow(metadata={"scores": {}, "triggered": []})

    api_key = os.environ.get(decision_model_config.api_key_env_var) or None
    if api_key is None:
        raise ValueError(f"{decision_model_config.api_key_env_var} environment variable not set.")

    try:
        assessment = await decision_model_request(
            text,
            {qid: q.instructions for qid, q in questions.items()},
            decision_model_config.server_endpoint,
            decision_model_config.model,
            api_key,
            http_client=http_client,
            timeout=decision_model_config.timeout,
        )
    except ValueError as err:
        log.warning(
            "Decision model assessment failed (%s); %s.",
            err,
            "allowing text" if decision_model_config.fail_open else "blocking text",
        )
        return _failure_outcome(decision_model_config.fail_open, str(err))

    scores = assessment.scores
    triggered = [
        qid
        for qid, score in scores.items()
        if score
        >= (questions[qid].threshold if questions[qid].threshold is not None else decision_model_config.threshold)
    ]
    metadata: Dict[str, Any] = {"scores": scores, "triggered": triggered}
    if assessment.model is not None:
        metadata["model"] = assessment.model
    if triggered:
        return RailOutcome.block(reason=f"Decision model flagged: {', '.join(triggered)}", metadata=metadata)
    return RailOutcome.allow(metadata=metadata)
