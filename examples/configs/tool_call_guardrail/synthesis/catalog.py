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

"""The trust boundary.

This is the *only* place a scanner finding becomes a real `Rule`. A finding does
not carry code; it carries an `attack_class` (mapped here to a vetted rule
factory) and parameters (fed into that factory). The worst a poisoned source can
do is propose parameters to a factory that already exists — and that still has to
clear the human review gate before it is ever applied.

Extending what the scanner is allowed to propose is therefore a deliberate,
reviewed act: add a factory to `RULE_FACTORIES` and a class mapping to
`CLASS_TO_FACTORY`. Nothing else widens the boundary.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from policy import (
    Rule,
    arg_matches_pattern,
    deny_arg_values,
    max_numeric_arg,
    require_owns_arg,
    require_principal_attr,
)

# Vetted rule factories the scanner may select from. Adding an entry is the
# explicit way to broaden what can be auto-proposed.
RULE_FACTORIES: Mapping[str, Callable[..., Rule]] = {
    "require_owns_arg": require_owns_arg,
    "max_numeric_arg": max_numeric_arg,
    "deny_arg_values": deny_arg_values,
    "arg_matches_pattern": arg_matches_pattern,
    "require_principal_attr": require_principal_attr,
}

# How an attack taxonomy class maps to a factory. A finding whose class is absent
# here yields no candidate (fail closed — unknown classes are never auto-acted).
CLASS_TO_FACTORY: Mapping[str, str] = {
    "ownership-bypass": "require_owns_arg",
    "unbounded-arg": "max_numeric_arg",
    "disallowed-target": "deny_arg_values",
    "argument-injection": "arg_matches_pattern",
    "privilege-escalation": "require_principal_attr",
}

# Plain-language definition of the *control* each class represents. Fed to the
# LLM extractor so it classifies on whether the control actually mitigates the
# technique — not on surface wording — and chooses "novel" when none fits.
CLASS_DESCRIPTIONS: Mapping[str, str] = {
    "ownership-bypass": (
        "The principal acts on a resource it does not own. The control requires "
        "the target resource to belong to the principal before the call runs. "
        "Use only for missing/insufficient ownership checks."
    ),
    "unbounded-arg": (
        "A single numeric argument on one call exceeds a safe maximum. The control "
        "caps that one argument at a ceiling. It does NOT address sequences, loops, "
        "or repeated calls — only the magnitude of one argument on one call."
    ),
    "disallowed-target": (
        "An argument may name a specifically forbidden target (a sanctioned "
        "account, a protected/system resource). The control blocks the call when "
        "the argument is in a denylist. Use for enumerable prohibited values, not "
        "for formats or numeric bounds."
    ),
    "argument-injection": (
        "An argument can carry malformed or injected content — path traversal, "
        "shell/SQL metacharacters, or a bad identifier. The control requires the "
        "argument to fully match an allowlisted pattern. Use when the risk is the "
        "FORM of one argument, not its magnitude, ownership, or a specific value."
    ),
    "privilege-escalation": (
        "A sensitive operation is invoked without the principal holding the "
        "required clearance. The control requires a specific principal attribute "
        "(e.g. mfa_verified, elevated) to be set. Use when the gap is missing "
        "step-up authorization, not resource ownership."
    ),
}


@dataclass(frozen=True)
class RuleCandidate:
    """A proposed rule: a vetted factory key plus parameters, with provenance.

    Inert until `materialize()` is called, which only happens after the human
    gate. `finding_id`/`source` are carried so a reviewer sees the why and where.
    """

    finding_id: str
    source: str
    tool: str
    factory_key: str
    params: Mapping[str, object] = field(default_factory=dict)
    rationale: str = ""

    def validation_error(self) -> Optional[str]:
        """Why this candidate cannot be applied, or None if it is sound.

        Checks both that the factory is vetted and that the proposed params
        actually bind to its signature — so a malformed proposal is caught when
        the candidate is built/reviewed, not when `apply()` runs `materialize()`.
        """
        if self.factory_key not in RULE_FACTORIES:
            return f"factory '{self.factory_key}' is not in the vetted catalog"
        factory = RULE_FACTORIES[self.factory_key]
        try:
            inspect.signature(factory).bind(**dict(self.params))
        except TypeError as exc:
            return f"params {dict(self.params)} do not fit {self.factory_key}(): {exc}"
        return None

    def is_valid(self) -> bool:
        """True only if the candidate is sound (vetted factory + fitting params)."""
        return self.validation_error() is None

    def materialize(self) -> Rule:
        """Build the concrete `Rule`. Raises if the candidate is not valid."""
        error = self.validation_error()
        if error is not None:
            raise ValueError(error)
        return RULE_FACTORIES[self.factory_key](**dict(self.params))

    def to_dict(self) -> dict:
        return {
            "approved": False,
            "finding_id": self.finding_id,
            "source": self.source,
            "tool": self.tool,
            "factory_key": self.factory_key,
            "params": dict(self.params),
            "rationale": self.rationale,
            "validation": self.validation_error(),  # null when sound; reason when not
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RuleCandidate":
        return cls(
            finding_id=str(data["finding_id"]),
            source=str(data.get("source", "")),
            tool=str(data["tool"]),
            factory_key=str(data["factory_key"]),
            params=dict(data.get("params", {})),
            rationale=str(data.get("rationale", "")),
        )
