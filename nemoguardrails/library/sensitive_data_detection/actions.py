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

import logging
from functools import lru_cache
from typing import Any

try:
    from presidio_analyzer import PatternRecognizer  # type: ignore[reportMissingImports]
    from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore[reportMissingImports]
    from presidio_anonymizer import AnonymizerEngine  # type: ignore[reportMissingImports]
    from presidio_anonymizer.entities import OperatorConfig  # type: ignore[reportMissingImports]
except ImportError:
    PatternRecognizer = None
    NlpEngineProvider = None
    AnonymizerEngine = None
    OperatorConfig = None

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.rails.llm.config import (
    SensitiveDataDetection,
    SensitiveDataDetectionOptions,
)

log = logging.getLogger(__name__)


@lru_cache
def _get_analyzer(score_threshold: float = 0.4):
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be a float between 0 and 1 (inclusive).")
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[reportMissingImports]

    except ImportError:
        raise ImportError(
            "Could not import presidio, please install it with `pip install presidio-analyzer presidio-anonymizer`."
        )

    try:
        import spacy  # type: ignore[reportMissingImports]
    except ImportError:
        raise RuntimeError("The spacy module is not installed. Please install it using pip: pip install spacy.")

    if not spacy.util.is_package("en_core_web_lg"):
        raise RuntimeError(
            "The en_core_web_lg Spacy model was not found. "
            "Please install using `python -m spacy download en_core_web_lg`"
        )

    # We provide this explicitly to avoid the default warning.
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }

    # Create NLP engine based on configuration
    if NlpEngineProvider is None:
        raise ImportError(
            "Could not import presidio, please install it with `pip install presidio-analyzer presidio-anonymizer`."
        )

    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # TODO: One needs to experiment with the score threshold to get the right value
    return AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=score_threshold)


def _get_ad_hoc_recognizers(sdd_config: SensitiveDataDetection):
    """Helper to compute the ad hoc recognizers for a config."""
    if PatternRecognizer is None:
        raise ImportError(
            "Could not import presidio, please install it with `pip install presidio-analyzer presidio-anonymizer`."
        )

    ad_hoc_recognizers = []
    for recognizer in sdd_config.recognizers:
        ad_hoc_recognizers.append(PatternRecognizer.from_dict(recognizer))
    return ad_hoc_recognizers


def _sensitive_data_detection_outcome(has_sensitive_data: bool) -> RailOutcome:
    if has_sensitive_data:
        return RailOutcome.block(has_sensitive_data=has_sensitive_data)
    return RailOutcome.allow(has_sensitive_data=has_sensitive_data)


def _mask_sensitive_data_outcome(source: str, original_text: str, masked_text: str) -> RailOutcome:
    target_by_source = {
        "input": TransformTarget.USER_MESSAGE,
        "output": TransformTarget.BOT_MESSAGE,
        "retrieval": TransformTarget.RELEVANT_CHUNKS,
    }
    metadata = {
        "source": source,
        "text": original_text,
        "masked_text": masked_text,
    }
    if masked_text != original_text:
        return RailOutcome.transform([(target_by_source[source], masked_text)], **metadata)
    return RailOutcome.allow(**metadata)


@action(is_system_action=True)
async def detect_sensitive_data(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
) -> RailOutcome:
    """Checks whether the provided text contains any sensitive data.

    Args
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns
        RailOutcome.block() if sensitive data has been detected, RailOutcome.allow() otherwise.
    """
    sdd_config = config.rails.config.sensitive_data_detection
    if sdd_config is None:
        raise ValueError("sensitive_data_detection configuration is required.")
    if source not in ["input", "output", "retrieval"]:
        raise ValueError("source must be one of 'input', 'output', or 'retrieval'")
    options: SensitiveDataDetectionOptions = getattr(sdd_config, source)
    default_score_threshold = getattr(options, "score_threshold")

    if len(options.entities) == 0:
        return _sensitive_data_detection_outcome(False)

    analyzer = _get_analyzer(score_threshold=default_score_threshold)
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=options.entities,
        ad_hoc_recognizers=_get_ad_hoc_recognizers(sdd_config),
    )

    if results:
        return _sensitive_data_detection_outcome(True)

    return _sensitive_data_detection_outcome(False)


@action(is_system_action=True)
async def mask_sensitive_data(source: str, text: str, config: RailsConfig) -> RailOutcome:
    """Checks whether the provided text contains any sensitive data.

    Args
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns
        RailOutcome.transform() with the altered text if it changed, RailOutcome.allow() otherwise.
    """
    sdd_config = config.rails.config.sensitive_data_detection
    if sdd_config is None:
        raise ValueError("sensitive_data_detection configuration is required.")
    assert source in ["input", "output", "retrieval"]
    options: SensitiveDataDetectionOptions = getattr(sdd_config, source)

    if len(options.entities) == 0:
        return _mask_sensitive_data_outcome(source, text, text)

    analyzer = _get_analyzer()
    if OperatorConfig is None or AnonymizerEngine is None:
        raise ImportError(
            "Could not import presidio, please install it with `pip install presidio-analyzer presidio-anonymizer`."
        )

    operators: dict[str, Any] = {}
    for entity in options.entities:
        operators[entity] = OperatorConfig("replace")

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=options.entities,
        ad_hoc_recognizers=_get_ad_hoc_recognizers(sdd_config),
    )
    anonymizer = AnonymizerEngine()
    masked_results = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)

    return _mask_sensitive_data_outcome(source, text, masked_results.text)
