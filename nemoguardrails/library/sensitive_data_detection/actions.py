# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    from presidio_analyzer import PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError:
    # The exception about installing presidio will be on the first call to the analyzer
    pass

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
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
        from presidio_analyzer import AnalyzerEngine

    except ImportError:
        raise ImportError(
            "Could not import presidio, please install it with "
            "`pip install presidio-analyzer presidio-anonymizer`."
        )

    try:
        import spacy
    except ImportError:
        raise RuntimeError(
            "The spacy module is not installed. Please install it using pip: pip install spacy."
        )

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
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # TODO: One needs to experiment with the score threshold to get the right value
    return AnalyzerEngine(
        nlp_engine=nlp_engine, default_score_threshold=score_threshold
    )


def _get_ad_hoc_recognizers(sdd_config: SensitiveDataDetection):
    """Helper to compute the ad hoc recognizers for a config."""
    ad_hoc_recognizers = []
    for recognizer in sdd_config.recognizers:
        ad_hoc_recognizers.append(PatternRecognizer.from_dict(recognizer))
    return ad_hoc_recognizers


def detect_sensitive_data_mapping(result: bool) -> bool:
    """
    Mapping for detect_sensitive_data.

    Since the function returns True when sensitive data is detected,
    we block if result is True.
    """
    return result


@action(is_system_action=True, output_mapping=detect_sensitive_data_mapping)
async def detect_sensitive_data(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
):
    """Checks whether the provided text contains any sensitive data.

    Args
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns
        True if any sensitive data has been detected, False otherwise.
    """
    # Based on the source of the data, we use the right options
    sdd_config = config.rails.config.sensitive_data_detection
    if source not in ["input", "output", "retrieval"]:
        raise ValueError("source must be one of 'input', 'output', or 'retrieval'")
    options: SensitiveDataDetectionOptions = getattr(sdd_config, source)
    default_score_threshold = getattr(options, "score_threshold")

    # If we don't have any entities specified, we stop
    if len(options.entities) == 0:
        return False

    analyzer = _get_analyzer(score_threshold=default_score_threshold)
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=options.entities,
        ad_hoc_recognizers=_get_ad_hoc_recognizers(sdd_config),
    )

    # If we have any
    if results:
        return True

    return False


@action(is_system_action=True)
async def mask_sensitive_data(source: str, text: str, config: RailsConfig):
    """Checks whether the provided text contains any sensitive data.

    Args
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns
        The altered text, if applicable.
    """
    # Based on the source of the data, we use the right options
    sdd_config = config.rails.config.sensitive_data_detection
    assert source in ["input", "output", "retrieval"]
    options: SensitiveDataDetectionOptions = getattr(sdd_config, source)

    # If we don't have any entities specified, we stop
    if len(options.entities) == 0:
        return text

    analyzer = _get_analyzer()
    operators = {}
    for entity in options.entities:
        operators[entity] = OperatorConfig("replace")

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=options.entities,
        ad_hoc_recognizers=_get_ad_hoc_recognizers(sdd_config),
    )
    anonymizer = AnonymizerEngine()
    masked_results = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators
    )

    return masked_results.text
