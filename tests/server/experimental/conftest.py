# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import ValidationError as SchemaValidationError

from nemoguardrails.server.experimental.provider.projection_policy import EXTENSION
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy


def _reject_case_aliases(validator, reviewed, instance, schema):
    """Check exact Unicode case folding against names derived from the export."""
    if not isinstance(instance, dict):
        return
    names = set(reviewed)
    folded_names = {name.casefold() for name in names}
    for name in instance:
        if name not in names and name.casefold() in folded_names:
            yield SchemaValidationError(f"Member {name!r} differs from a reviewed name only by case")


_ExportPolicyValidator = validators.extend(Draft202012Validator, {"x-test-reviewed-properties": _reject_case_aliases})


def _lower_export(node: object, policy: UnknownContentFieldPolicy) -> object:
    """Derive a JSON Schema validator input from an exported payload schema alone.

    Configurable objects accept unknown members only under trusted ALLOW. Any
    member that matches a listed property only after Unicode case folding is
    rejected when the exported object explicitly declares that rule.
    """
    if isinstance(node, list):
        return [_lower_export(child, policy) for child in node]
    if not isinstance(node, dict):
        return node
    lowered = {key: _lower_export(value, policy) for key, value in node.items() if key != EXTENSION}
    if "properties" in node:
        if node.get(EXTENSION, {}).get("unknown_fields") == "configurable":
            lowered["additionalProperties"] = policy == UnknownContentFieldPolicy.ALLOW
        if node.get(EXTENSION, {}).get("reject_case_aliases") is True:
            lowered["x-test-reviewed-properties"] = list(node["properties"])
        lowered["properties"] = {name: _lower_export(child, policy) for name, child in node["properties"].items()}
    return lowered


@pytest.fixture
def export_policy_validator():
    def build(exported, policy):
        return _ExportPolicyValidator(_lower_export(exported, policy))

    return build
