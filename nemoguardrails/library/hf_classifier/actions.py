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

"""HuggingFace classifier-based detection actions."""

import logging
from typing import Any, List, Optional, Tuple

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.http import HTTPClient
from nemoguardrails.library.hf_classifier.backends import get_backend

log = logging.getLogger(__name__)


async def _classify_and_check(
    classifier_name: str,
    text: str,
    config: Any | None,
    http_client: HTTPClient | None = None,
) -> bool:
    """Classify *text* and check against blocked labels.

    Returns ``True`` if allowed, ``False`` if blocked.
    """
    classifiers = getattr(config.rails.config, "hf_classifier", None) if config else None
    if not classifiers:
        raise ValueError(
            "hf_classifier action called but no 'hf_classifier' section found in "
            "rails.config. Check your config.yml for typos."
        )

    classifier_config = classifiers.get(classifier_name)
    if classifier_config is None:
        raise ValueError(f"Unknown classifier '{classifier_name}'. Available: {list(classifiers)}")

    if not text:
        return True

    backend_kwargs = {"http_client": http_client} if http_client is not None else {}
    backend = get_backend(classifier_config, name=classifier_name, **backend_kwargs)
    results = await backend.classify(text)

    if text and not results and getattr(classifier_config, "task", None) == "text-classification":
        log.warning(
            "HF classifier '%s' returned no results for non-empty input — "
            "possible API compatibility issue with the '%s' backend.",
            classifier_name,
            classifier_config.engine,
        )

    blocked = set(classifier_config.blocked_labels)
    threshold = classifier_config.threshold

    triggered: List[Tuple[str, float]] = [
        (r["label"], r["score"]) for r in results if r["label"] in blocked and r["score"] >= threshold
    ]

    if triggered:
        log.info(
            "HF classifier '%s': blocked (detections: %s)",
            classifier_name,
            triggered,
        )
        return False

    log.info("HF classifier '%s': allowed", classifier_name)
    return True


def _extract_text(context: Optional[dict], key: str) -> str:
    return (context.get(key) or "") if context else ""


def _hf_classifier_outcome(allowed: bool) -> RailOutcome:
    if allowed:
        return RailOutcome.allow()
    return RailOutcome.block()


def _hf_classifier_retrieval_outcome(allowed: bool) -> RailOutcome:
    if allowed:
        return RailOutcome.allow()
    return RailOutcome.transform([(TransformTarget.RELEVANT_CHUNKS, "")])


@action()
async def hf_classifier_check_input(
    classifier: str,
    config: Any | None = None,
    context: Optional[dict] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    allowed = await _classify_and_check(classifier, _extract_text(context, "user_message"), config, http_client)
    return _hf_classifier_outcome(allowed)


@action()
async def hf_classifier_check_output(
    classifier: str,
    config: Any | None = None,
    context: Optional[dict] = None,
    model_name: Optional[str] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    # Streaming output rail path doesn't resolve flow variables — $classifier
    # arrives as a literal string. Fall back to model_name which the streaming
    # engine extracts from the flow_id.
    if classifier.startswith("$") and model_name:
        classifier = model_name
    allowed = await _classify_and_check(classifier, _extract_text(context, "bot_message"), config, http_client)
    return _hf_classifier_outcome(allowed)


@action()
async def hf_classifier_check_retrieval(
    classifier: str,
    config: Any | None = None,
    context: Optional[dict] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    allowed = await _classify_and_check(classifier, _extract_text(context, "relevant_chunks"), config, http_client)
    return _hf_classifier_retrieval_outcome(allowed)
