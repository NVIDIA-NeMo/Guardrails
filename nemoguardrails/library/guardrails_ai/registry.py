# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Dict

from .errors import GuardrailsAIConfigError

log = logging.getLogger(__name__)

VALIDATOR_REGISTRY = {
    "toxic_language": {
        "module": "guardrails.hub",
        "class": "ToxicLanguage",
        "hub_path": "hub://guardrails/toxic_language",
        "default_params": {"on_fail": "noop"},
    },
    "detect_jailbreak": {
        "module": "guardrails.hub",
        "class": "DetectJailbreak",
        "hub_path": "hub://guardrails/detect_jailbreak",
        "default_params": {"on_fail": "noop"},
    },
    "guardrails_pii": {
        "module": "guardrails.hub",
        "class": "GuardrailsPII",
        "hub_path": "hub://guardrails/guardrails_pii",
        "default_params": {"on_fail": "noop"},
    },
    "competitor_check": {
        "module": "guardrails.hub",
        "class": "CompetitorCheck",
        "hub_path": "hub://guardrails/competitor_check",
        "default_params": {"on_fail": "noop"},
    },
    "restricttotopic": {
        "module": "guardrails.hub",
        "class": "RestrictToTopic",
        "hub_path": "hub://tryolabs/restricttotopic",
        "default_params": {"on_fail": "noop"},
    },
    "provenance_llm": {
        "module": "guardrails.hub",
        "class": "ProvenanceLLM",
        "hub_path": "hub://guardrails/provenance_llm",
        "default_params": {"on_fail": "noop"},
    },
    "regex_match": {
        "module": "guardrails.hub",
        "class": "RegexMatch",
        "hub_path": "hub://guardrails/regex_match",
        "default_params": {"on_fail": "noop"},
    },
    "one_line": {
        "module": "guardrails.hub",
        "class": "OneLine",
        "hub_path": "hub://guardrails/one_line",
        "default_params": {"on_fail": "noop"},
    },
    "valid_json": {
        "module": "guardrails.hub",
        "class": "ValidJson",
        "hub_path": "hub://guardrails/valid_json",
        "default_params": {"on_fail": "noop"},
    },
    "valid_length": {
        "module": "guardrails.hub",
        "class": "ValidLength",
        "hub_path": "hub://guardrails/valid_length",
        "default_params": {"on_fail": "noop"},
    },
}


def get_validator_info(validator_path: str) -> Dict[str, str]:
    """Get validator information from registry or hub.

    Args:
        validator_path: Either a simple name (e.g., "toxic_language") or
                       a full hub path (e.g., "guardrails/toxic_language")

    Returns:
        Dict with module, class, and hub_path information
    """
    if validator_path in VALIDATOR_REGISTRY:
        return VALIDATOR_REGISTRY[validator_path]

    for _, info in VALIDATOR_REGISTRY.items():
        if info["hub_path"] == f"hub://{validator_path}":
            return info

    # not in registry, try to fetch from hub
    try:
        try:
            from guardrails.hub.validator_package_service import get_validator_manifest
        except ImportError:
            raise GuardrailsAIConfigError(
                "Could not import get_validator_manifest. "
                "Make sure guardrails-ai is properly installed."
            )

        log.info(
            f"Validator '{validator_path}' not found in registry. "
            f"Attempting to fetch from Guardrails Hub..."
        )

        manifest = get_validator_manifest(validator_path)

        if manifest.exports:
            class_name = manifest.exports[0]
        else:
            # fallback: construct class name from package name
            class_name = "".join(
                word.capitalize() for word in manifest.package_name.split("_")
            )

        validator_info = {
            "module": "guardrails.hub",
            "class": class_name,
            "hub_path": f"hub://{manifest.namespace}/{manifest.package_name}",
        }

        log.info(
            f"Using validator '{validator_path}' that is not in the built-in registry. "
            f"Consider adding it to VALIDATOR_REGISTRY for better performance. "
            f"Install with: guardrails hub install {validator_info['hub_path']}"
        )

        return validator_info

    except ImportError:
        raise GuardrailsAIConfigError(
            "Could not import get_validator_manifest. "
            "Make sure guardrails-ai is properly installed."
        )
    except Exception as e:
        raise GuardrailsAIConfigError(
            f"Failed to fetch validator info for '{validator_path}': {str(e)}"
        )
