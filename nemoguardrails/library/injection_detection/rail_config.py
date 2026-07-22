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

from typing import Dict, List, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class InjectionDetection(RailConfigBaseModel):
    injections: List[str] = Field(
        default_factory=list,
        description="The list of injection types to detect. Options are 'sqli', 'template', 'code', 'xss'."
        "Currently, only SQL injection, template injection, code injection, "
        "and markdown cross-site scripting are supported. "
        "Custom rules can be added, provided they are in the `yara_path` and have a `.yara` file extension.",
    )
    action: str = Field(
        default="reject",
        pattern=r"^(reject|omit)$",
        description="Action to take. Options are 'reject' to offer a rejection message, "
        "'omit' to mask the offending content, and 'sanitize' to pass the content as-is in the safest way. "
        "These options are listed in descending order of relative safety. 'sanitize' is not implemented at this time.",
    )
    yara_path: Optional[str] = Field(
        default="",
        description="Location on disk where YARA rules are located. If this parameter is an empty string, "
        "the default location defined in injection_detection's actions.py file will be used.",
    )
    yara_rules: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Dictionary mapping rule names to YARA rule strings. If provided, these rules will be used "
        "instead of loading rules from yara_path. Each rule should be a valid YARA rule string.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[InjectionDetection],
        field_info=rail_field(
            default_factory=InjectionDetection,
            description="Configuration for injection detection.",
        ),
        exports={"InjectionDetection": InjectionDetection},
    )
