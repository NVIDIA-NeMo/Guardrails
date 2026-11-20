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

"""The hooks server's Cedar rules: an agent start or a tool call decided by Cedar policies alone, with `cedarpy`.

The vocabulary, in the `Hooks` namespace, is in `docs/run-rails/using-hooks-server/cedar.mdx` and in
`examples/configs/agent_hooks/cedar/hooks.cedarschema`. Cedar has no "for every" over a set, so a start is asked once
for the agent and once for each part, model and policy, each as the request's resource, and is allowed only when Cedar
allows every one. A tool call is asked once.

Cedar's own rules apply: deny unless a `permit` matches, and any matching `forbid` wins. Cedar skips a policy that
fails to evaluate; here an evaluation error denies, so a `forbid` that fails never lets a request through.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from nemoguardrails.guardrails.guardrails_types import quoted_identity
from nemoguardrails.server.hooks.schemas import AgentStartRequest

log = logging.getLogger(__name__)

NAMESPACE = "Hooks"
START = "start"
CALL_TOOL = "call_tool"
# The annotation on a `forbid` that selects what a tool call it decides ends in, beyond a deny.
OUTCOME_ANNOTATION = "outcome"
OUTCOMES = ("stop", "quarantine")
_INSTALL_HINT = "Install it with `pip install nemoguardrails[cedar]`."
# Cedar's JSON reads a record with one of these keys as an entity or an extension value, not as a record, so a
# caller's arguments must never carry them into a request.
_CEDAR_JSON_ESCAPES = frozenset({"__entity", "__extn", "__expr"})
_LONG_MIN, _LONG_MAX = -(2**63), 2**63 - 1


class CedarPolicyError(ValueError):
    """A Cedar policy set the server can't use, found when it starts."""


def _cedarpy() -> Any:
    """The `cedarpy` module, imported only when a config asks for Cedar."""
    try:
        import cedarpy
    except ImportError as exc:
        raise CedarPolicyError(f"The config's Cedar rules need the `cedarpy` package. {_INSTALL_HINT}") from exc
    return cedarpy


@dataclass(frozen=True)
class CedarDecision:
    """What Cedar said about one request: whether it is allowed, the policies that denied it, and the outcome."""

    allowed: bool
    # The labels of the `forbid` policies that matched, an `@id` or the file and position; empty for a default deny.
    denied_by: tuple[str, ...] = ()
    # "deny", "stop" or "quarantine": the strongest `@outcome` among the matching `forbid` policies.
    outcome: str = "deny"
    failed: tuple[str, ...] = ()
    # The policy language the reason names: "Cedar", or "Dogwood" for policies that read the run's history.
    language: str = "Cedar"

    def reason(self, subject: str) -> Optional[str]:
        """Why *subject* is denied, naming the deciding policies; None when it is allowed."""
        if self.allowed:
            return None
        if self.failed:
            labels = ", ".join(f"'{label}'" for label in self.failed)
            return f"{subject} is denied: {self.language} policy {labels} failed to evaluate"
        if self.denied_by:
            labels = ", ".join(f"'{label}'" for label in self.denied_by)
            verb = {"stop": "stopped", "quarantine": "quarantined"}.get(self.outcome, "denied")
            return f"{subject} is {verb} by {self.language} policy {labels}"
        return f"no {self.language} policy permits {subject}"


@dataclass(frozen=True)
class CedarPolicySet:
    """A folder of `.cedar` policies, parsed once when the server starts."""

    text: str
    labels: Mapping[str, str]
    outcomes: Mapping[str, str]
    language: str = "Cedar"

    @classmethod
    def load(cls, folder: Path, schema: Optional[Path], *, section: str, outcomes_allowed: bool) -> "CedarPolicySet":
        """Read and check every `.cedar` file under *folder*; raise `CedarPolicyError` naming what is wrong."""
        _cedarpy()
        if not folder.is_dir():
            raise CedarPolicyError(f"{section}.cedar.policies names '{folder}', which is not a folder")
        files = sorted(path for path in folder.rglob("*.cedar") if path.is_file())
        if not files:
            raise CedarPolicyError(f"{section}.cedar.policies names '{folder}', which has no .cedar file")
        named = [(path.relative_to(folder).as_posix(), path.read_text(encoding="utf-8")) for path in files]
        return cls.parse(named, schema, section=f"{section}.cedar", outcomes_allowed=outcomes_allowed)

    @classmethod
    def parse(
        cls,
        named_texts: Sequence[tuple[str, str]],
        schema: Optional[Path],
        *,
        section: str,
        outcomes_allowed: bool,
        language: str = "Cedar",
    ) -> "CedarPolicySet":
        """Parse and check each `(file name, Cedar text)`; raise `CedarPolicyError` naming what is wrong."""
        cedarpy = _cedarpy()
        texts: list[str] = []
        labels: dict[str, str] = {}
        outcomes: dict[str, str] = {}
        for name, text in named_texts:
            try:
                parsed = json.loads(cedarpy.policies_to_json_str(text))
            except ValueError as exc:
                raise CedarPolicyError(f"{section}: '{name}' doesn't parse as Cedar: {exc}") from None
            if parsed.get("templates"):
                raise CedarPolicyError(
                    f"{section}: '{name}' has a template; the hooks server reads static policies only"
                )
            # Cedar numbers the policies of the joined text in order, so file order fixes each policy's id.
            for position, policy in enumerate(_in_order(parsed.get("staticPolicies", {}))):
                cedar_id = f"policy{len(labels)}"
                annotations = policy.get("annotations") or {}
                labels[cedar_id] = annotations.get("id") or f"{name}#{position}"
                outcome = annotations.get(OUTCOME_ANNOTATION)
                if outcome is None:
                    continue
                where = f"{section}: policy '{labels[cedar_id]}' in '{name}'"
                if not outcomes_allowed:
                    raise CedarPolicyError(
                        f"{where} has @{OUTCOME_ANNOTATION}, which only a tool call's policy can have"
                    )
                if policy.get("effect") != "forbid":
                    raise CedarPolicyError(f"{where} has @{OUTCOME_ANNOTATION} on a permit; only a forbid selects one")
                if outcome not in OUTCOMES:
                    raise CedarPolicyError(
                        f'{where} has @{OUTCOME_ANNOTATION}("{outcome}"); use "stop" or "quarantine"'
                    )
                outcomes[cedar_id] = outcome
            texts.append(text)
        joined = "\n".join(texts)
        if schema is not None:
            if not schema.is_file():
                raise CedarPolicyError(f"{section}.schema names '{schema}', which is not a file")
            result = cedarpy.validate_policies(joined, schema.read_text(encoding="utf-8"))
            if not result.validation_passed:
                errors = "; ".join(str(error) for error in result.errors)
                raise CedarPolicyError(f"{section}: the policies fail the check against the schema: {errors}")
        return cls(text=joined, labels=labels, outcomes=outcomes, language=language)

    @property
    def quarantines(self) -> bool:
        """Whether a policy can quarantine an agent."""
        return "quarantine" in self.outcomes.values()

    def decide(
        self,
        principal: Mapping[str, Any],
        action: str,
        resource: Mapping[str, Any],
        context: Mapping[str, Any],
        entities: Sequence[Mapping[str, Any]],
        *,
        action_type: str = f"{NAMESPACE}::Action",
    ) -> CedarDecision:
        """Ask Cedar about one request; an engine error or a policy that fails to evaluate denies it."""
        request = {
            "principal": principal,
            "action": {"type": action_type, "id": action},
            "resource": resource,
            "context": dict(context),
        }
        try:
            result = _cedarpy().is_authorized(request, self.text, list(entities))
            decision = getattr(result.decision, "value", None)
            reasons = tuple(sorted(str(reason) for reason in result.diagnostics.reasons))
            errors = [str(error) for error in result.diagnostics.errors]
        except Exception as exc:  # Whatever the engine raises, the request is denied, never allowed.
            log.warning("Cedar couldn't decide a %s request: %s", action, type(exc).__name__)
            return CedarDecision(allowed=False, failed=("the engine",), language=self.language)
        if errors:
            # The error names the policy and the attribute, never a request's values.
            log.warning("A Cedar policy failed to evaluate on a %s request: %s", action, "; ".join(errors))
            failed = tuple(label for cedar_id, label in self.labels.items() if f"`{cedar_id}`" in " ".join(errors))
            return CedarDecision(allowed=False, failed=failed or ("unknown",), language=self.language)
        if decision == "Allow":
            return CedarDecision(allowed=True)
        selected = {self.outcomes.get(reason) for reason in reasons}
        outcome = "quarantine" if "quarantine" in selected else "stop" if "stop" in selected else "deny"
        return CedarDecision(
            allowed=False,
            denied_by=tuple(self.labels.get(reason, reason) for reason in reasons),
            outcome=outcome,
            language=self.language,
        )


def _in_order(policies: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A file's policies in the order Cedar numbers them, `policy0` first."""
    return [policies[key] for key in sorted(policies, key=lambda key: int(key.removeprefix("policy")))]


def _uid(entity_type: str, entity_id: str) -> dict[str, str]:
    return {"type": f"{NAMESPACE}::{entity_type}", "id": entity_id}


def _agent_entity(name: str, digest: Optional[str]) -> dict[str, Any]:
    attrs: dict[str, Any] = {"name": name}
    if digest:
        attrs["digest"] = digest
    return {"uid": _uid("Agent", name), "attrs": attrs, "parents": []}


_DROP = object()


def cedar_value(value: Any) -> Any:
    """*value* as a Cedar value, or `_DROP` when Cedar has none: null, a fractional or out-of-range number."""
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value if _LONG_MIN <= value <= _LONG_MAX else _DROP
    if isinstance(value, list):
        items = [cedar_value(item) for item in value]
        return _DROP if any(item is _DROP for item in items) else items
    if isinstance(value, dict):
        return cedar_record(value)
    return _DROP


def cedar_record(values: Mapping[str, Any]) -> dict[str, Any]:
    """*values* as a Cedar record, leaving out what Cedar can't hold and the keys Cedar's JSON reads specially."""
    record = {}
    for key, value in values.items():
        if key in _CEDAR_JSON_ESCAPES:
            continue
        converted = cedar_value(value)
        if converted is not _DROP:
            record[key] = converted
    return record


@dataclass(frozen=True)
class CedarToolRules:
    """`hooks_server.cedar`: each tool call is asked as `Hooks::Action::"call_tool"`."""

    policies: CedarPolicySet

    def decide(
        self, name: str, arguments: Mapping[str, Any], agent_name: Optional[str], agent_digest: Optional[str]
    ) -> tuple[str, Optional[str]]:
        """The outcome, "allow", "deny", "stop" or "quarantine", and the reason for any but allow."""
        agent = _agent_entity(agent_name or "", agent_digest)
        tool = {"uid": _uid("Tool", name), "attrs": {"name": name}, "parents": []}
        context = {"arguments": cedar_record(arguments), "argument_names": sorted(arguments)}
        decision = self.policies.decide(agent["uid"], CALL_TOOL, tool["uid"], context, [agent, tool])
        if decision.allowed:
            return "allow", None
        return decision.outcome, decision.reason(f"tool call '{quoted_identity(name)}'")


@dataclass(frozen=True)
class CedarAgentStartRules:
    """`hooks_server.agent_start.cedar`: a start is asked as `Hooks::Action::"start"` for the agent and each of its
    parts, models and policies, and allowed only when every answer is allow."""

    policies: CedarPolicySet

    def decide(self, request: AgentStartRequest) -> Optional[str]:
        """Why Cedar denies the start, naming the first resource it denies; None to allow."""
        agent = _agent_entity(request.agent_name, request.agent_digest)
        context = {"enforcers_present": sorted(set(request.enforcers_present))}
        asked: list[tuple[dict[str, Any], str]] = [(agent, f"agent '{quoted_identity(request.agent_name)}'")]
        for part in request.parts:
            attrs: dict[str, Any] = {"id": part.id, "type": part.type, "source": part.source, "local": part.local}
            if part.version is not None:
                attrs["version"] = part.version
            if part.digest:
                attrs["digest"] = part.digest
            entity = {"uid": _uid("Part", part.id), "attrs": attrs, "parents": [agent["uid"]]}
            asked.append((entity, f"{part.type} '{quoted_identity(part.id)}'"))
        for model in request.models:
            entity = {"uid": _uid("Model", model), "attrs": {"name": model}, "parents": [agent["uid"]]}
            asked.append((entity, f"model '{quoted_identity(model)}'"))
        for policy in request.policies:
            attrs = {"id": policy.id, "required": policy.required, "enforcers": sorted(set(policy.enforcers))}
            entity = {"uid": _uid("Policy", policy.id), "attrs": attrs, "parents": [agent["uid"]]}
            asked.append((entity, f"policy '{quoted_identity(policy.id)}'"))
        for entity, subject in asked:
            entities = [agent] if entity is agent else [agent, entity]
            decision = self.policies.decide(agent["uid"], START, entity["uid"], context, entities)
            if not decision.allowed:
                return decision.reason(subject)
        return None
