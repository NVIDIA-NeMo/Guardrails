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

"""The hooks server's Dogwood rules, decided by Dogwood's own library.

Every parse, lowering, type check and decision is Dogwood's (`dogwood-ops`, through the `nemoguardrails-dogwood`
binding); this module only turns the server's requests and the decision record into a Dogwood event trace. The trace
is the run's recorded events, with the times they were recorded, followed by the request being decided; Dogwood
replays it through a fresh authorizer, and the last verdict is the decision. The decision record stays the one
history.

The policy language is Dogwood's whole language: temporal conditions, macros, and information providers. The
vocabulary of tool-call rules is AgentCore Gateway's, and their event schema is the one AgentCore documents
(`AGENTCORE_EVENT_SCHEMA`).
"""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from nemoguardrails.guardrails.guardrails_types import quoted_identity
from nemoguardrails.server.hooks.cedar_policy import CedarPolicyError

log = logging.getLogger(__name__)

MAX_WINDOW = timedelta(hours=24)
SUFFIXES = (".dw", ".dogwood")
OUTCOMES = ("stop", "quarantine")
_INSTALL_HINT = "Install it with `pip install nemoguardrails[dogwood]`."
_LONG_MIN, _LONG_MAX = -(2**63), 2**63 - 1
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# AgentCore Gateway's temporal event schema, as its documentation gives it ("Authoring temporal policies"): a request
# event for each request, a response when the tool returns, an error otherwise, each with the call's inputs and the
# principal and resource. The principal is pinned, so a principal's history is its own.
AGENTCORE_EVENT_SCHEMA = """decision event <A>::request {
    ...inputs(A),
    pin eventPrincipal: principalType(A) = principal,
    eventResource: resourceType(A),
    requestId: String,
}
event <A>::response {
    ...inputs(A),
    ...outputs(A),
    pin eventPrincipal: principalType(A) = principal,
    eventResource: resourceType(A),
    requestId: String,
}
event <A>::error {
    ...inputs(A),
    pin eventPrincipal: principalType(A) = principal,
    eventResource: resourceType(A),
    requestId: String,
}
"""


class DogwoodPolicyError(CedarPolicyError):
    """A Dogwood policy set the server can't use, found when it starts."""


def _dogwood() -> Any:
    """The `nemoguardrails_dogwood` binding, imported only when a config asks for Dogwood."""
    try:
        import nemoguardrails_dogwood
    except ImportError as exc:
        raise DogwoodPolicyError(
            f"The config's Dogwood rules need the `nemoguardrails-dogwood` package. {_INSTALL_HINT}"
        ) from exc
    return nemoguardrails_dogwood


@dataclass(frozen=True)
class Event:
    """One recorded event: a tool call asked about (`request`), completed (`response`) or failed (`error`), or an
    event recorded from outside a run, with the time it was recorded and the inputs the schema declares."""

    kind: str
    action: str
    at: datetime
    inputs: Mapping[str, Any] = field(default_factory=dict)


def tool_action(tool_name: str) -> str:
    """The AgentCore action of a hook's tool: a Claude Code MCP tool `mcp__<server>__<tool>` is `<server>___<tool>`,
    the `<target>___<tool>` form AgentCore and Agentform use; any other name is used as it is."""
    match = re.fullmatch(r"mcp__([^_]+(?:_[^_]+)*)__(.+)", tool_name)
    return f"{match.group(1)}___{match.group(2)}" if match else tool_name


# --- Dogwood trace text -------------------------------------------------------------------------------------------


def _string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _literal(value: Any) -> Optional[str]:
    """*value* as a Dogwood trace literal, or None when it has none (null, a fractional or out-of-range number)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value) if _LONG_MIN <= value <= _LONG_MAX else None
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            text = _literal(item)
            if text is None:
                return None
            items.append(text)
        return "[" + ", ".join(items) + "]"
    if isinstance(value, dict):
        return _record(value)
    return None


def _record(values: Mapping[str, Any]) -> str:
    fields = []
    for key, value in values.items():
        text = _literal(value)
        if text is not None:
            fields.append(f"{key if _IDENTIFIER.fullmatch(key) else _string(key)}: {text}")
    return "{ " + ", ".join(fields) + " }" if fields else "{}"


def _uid(entity_type: str, entity_id: str) -> str:
    return f"{entity_type}::{_string(entity_id)}"


class _Timeline:
    """Whole-second timestamps that never go back, as Dogwood's trace requires."""

    def __init__(self) -> None:
        self.last: Optional[int] = None

    def at(self, moment: datetime) -> int:
        seconds = int(moment.timestamp())
        self.last = seconds if self.last is None else max(self.last, seconds)
        return self.last


# --- A loaded policy set ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _PolicySet:
    """A folder of Dogwood files, parsed, lowered and type-checked by Dogwood when the server starts."""

    source: str
    action_schema: str
    event_schema: Optional[str]
    providers: Optional[str]
    macros: Optional[str]
    # Per rule, in Dogwood's rule order: its `@id` or file and position, and its `@outcome`.
    labels: tuple[str, ...]
    outcomes: Mapping[int, str]
    # Per action, the input fields the schema declares: the only inputs passed to Dogwood or recorded.
    declared_inputs: Mapping[str, frozenset[str]]

    @classmethod
    def load(
        cls,
        folder: Path,
        action_schema: Path,
        *,
        section: str,
        event_schema: Optional[str],
        providers: Optional[Path],
        macros: Optional[Path],
        outcomes_allowed: bool,
    ) -> "_PolicySet":
        dogwood = _dogwood()
        if not folder.is_dir():
            raise DogwoodPolicyError(f"{section}.policies names '{folder}', which is not a folder")
        files = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix in SUFFIXES)
        if not files:
            raise DogwoodPolicyError(f"{section}.policies names '{folder}', which has no .dw or .dogwood file")
        for path, what in ((action_schema, "schema"), (providers, "providers"), (macros, "macros")):
            if path is not None and not path.is_file():
                raise DogwoodPolicyError(f"{section}.{what} names '{path}', which is not a file")
        schema_text = action_schema.read_text(encoding="utf-8")
        providers_text = None if providers is None else providers.read_text(encoding="utf-8")
        macros_text = None if macros is None else macros.read_text(encoding="utf-8")
        source = "\n".join(path.read_text(encoding="utf-8") for path in files)
        try:
            report = json.loads(dogwood.validate(source, schema_text, event_schema, providers_text, macros_text))
            artifacts = json.loads(dogwood.lower(source, schema_text, event_schema, providers_text, macros_text))
        except ValueError as exc:
            raise DogwoodPolicyError(f"{section}: Dogwood refuses the policies: {_message(exc)}") from None
        if not report["passed"]:
            first = report["errors"][0]["message"] if report["errors"] else "the type check failed"
            raise DogwoodPolicyError(f"{section}: the policies fail Dogwood's type check: {first}")
        labels, outcomes = _annotations(artifacts["cedar_policies"], section, outcomes_allowed)
        return cls(
            source,
            schema_text,
            event_schema,
            providers_text,
            macros_text,
            labels,
            outcomes,
            _declared_inputs(json.loads(artifacts["cedar_schema_json"])),
        )

    @property
    def quarantines(self) -> bool:
        return "quarantine" in self.outcomes.values()

    def verdicts(self, lines: Sequence[str]) -> list[dict[str, Any]]:
        """Dogwood's verdict for each decision event of the trace *lines*, in order."""
        report = json.loads(
            _dogwood().replay(
                self.source, self.action_schema, "\n".join(lines), self.event_schema, self.providers, self.macros
            )
        )
        return report["verdicts"]

    def reason(self, verdict: Mapping[str, Any], subject: str) -> tuple[str, str]:
        """The outcome, "deny", "stop" or "quarantine", and why *subject* is not allowed."""
        if verdict.get("errors"):
            return "deny", f"{subject} is denied: Dogwood could not decide it: {verdict['errors'][0]}"
        rules = list(verdict.get("determining_rules") or [])
        if not rules:
            return "deny", f"no Dogwood policy permits {subject}"
        selected = {self.outcomes.get(rule) for rule in rules}
        outcome = "quarantine" if "quarantine" in selected else "stop" if "stop" in selected else "deny"
        verb = {"stop": "stopped", "quarantine": "quarantined"}.get(outcome, "denied")
        names = ", ".join(f"'{self.labels[rule]}'" if rule < len(self.labels) else f"'rule {rule}'" for rule in rules)
        return outcome, f"{subject} is {verb} by Dogwood policy {names}"

    def inputs(self, action: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        names = self.declared_inputs.get(action, frozenset())
        return {name: arguments[name] for name in sorted(names) if name in arguments}


def _message(exc: ValueError) -> str:
    try:
        return str(json.loads(str(exc)).get("message") or exc)
    except ValueError:
        return str(exc)


def _annotations(cedar: str, section: str, outcomes_allowed: bool) -> tuple[tuple[str, ...], dict[int, str]]:
    """The label and `@outcome` of each lowered policy, in Dogwood's rule order."""
    import cedarpy

    parsed = json.loads(cedarpy.policies_to_json_str(cedar))
    policies = parsed.get("staticPolicies", {})
    ordered = [policies[key] for key in sorted(policies, key=lambda key: int(key.removeprefix("policy")))]
    labels, outcomes = [], {}
    for index, policy in enumerate(ordered):
        annotations = policy.get("annotations") or {}
        labels.append(annotations.get("id") or f"rule {index}")
        outcome = annotations.get("outcome")
        if outcome is None:
            continue
        where = f"{section}: policy '{labels[-1]}'"
        if not outcomes_allowed:
            raise DogwoodPolicyError(f"{where} has @outcome, which only a tool call's policy can have")
        if policy.get("effect") != "forbid":
            raise DogwoodPolicyError(f"{where} has @outcome on a permit; only a forbid selects one")
        if outcome not in OUTCOMES:
            raise DogwoodPolicyError(f'{where} has @outcome("{outcome}"); use "stop" or "quarantine"')
        outcomes[index] = outcome
    return tuple(labels), outcomes


def _declared_inputs(schema: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """Each action's declared `context.input` fields, from Dogwood's Cedar JSON schema."""
    declared: dict[str, frozenset[str]] = {}
    for namespace in schema.values():
        common = namespace.get("commonTypes") or {}
        for action, definition in (namespace.get("actions") or {}).items():
            context = ((definition.get("appliesTo") or {}).get("context")) or {}
            if context.get("type") in ("EntityOrCommon", "CommonType") and context.get("name") in common:
                context = common[context["name"]]
            entry = (context.get("attributes") or {}).get("input") or {}
            if entry.get("type") in ("EntityOrCommon", "CommonType") and entry.get("name") in common:
                entry = common[entry["name"]]
            declared[action] = frozenset((entry.get("attributes") or {}).keys())
    return declared


# --- Tool calls ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DogwoodToolRules:
    """`hooks_server.dogwood`: each tool call is asked as `AgentCore::Action::"<tool>"` on the configured gateway, by
    an `AgentCore::OAuthUser` named after the agent, after the run's recorded events."""

    policies: _PolicySet
    gateway: str
    # What a denied call is in history: "request", an attempt, as the Dogwood language keeps it; or "error".
    denied_calls: str = "request"

    @classmethod
    def load(
        cls,
        folder: Path,
        schema: Path,
        gateway: str,
        denied_calls: str = "request",
        providers: Optional[Path] = None,
        macros: Optional[Path] = None,
    ) -> "DogwoodToolRules":
        policies = _PolicySet.load(
            folder,
            schema,
            section="hooks_server.dogwood",
            event_schema=AGENTCORE_EVENT_SCHEMA,
            providers=providers,
            macros=macros,
            outcomes_allowed=True,
        )
        return cls(policies, gateway, denied_calls)

    @property
    def quarantines(self) -> bool:
        return self.policies.quarantines

    def kept_inputs(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """The arguments of a call that the schema declares for its action, and only those: what Dogwood sees of the
        call, and what the record keeps."""
        return self.policies.inputs(tool_action(tool_name), arguments)

    def _line(self, timeline: _Timeline, event: Event, agent: str, request_id: int) -> Optional[str]:
        if event.action not in self.policies.declared_inputs:
            # An action the schema doesn't declare has no events in Dogwood's history.
            return None
        principal = _uid("AgentCore::OAuthUser", agent)
        resource = _uid("AgentCore::Gateway", self.gateway)
        inputs = _record(self.policies.inputs(event.action, event.inputs))
        return (
            f"@{timeline.at(event.at)} scope(principal: {principal}, resource: {resource}) "
            f"request_context(input: {inputs}) AgentCore::Action::{_string(event.action)}::{event.kind}("
            f"input: {inputs}, eventPrincipal: {principal}, eventResource: {resource}, "
            f'requestId: "r{request_id}")'
        )

    def decide(
        self,
        name: str,
        arguments: Mapping[str, Any],
        agent_name: Optional[str],
        agent_digest: Optional[str],
        history: Optional[Sequence[Event]],
        now: datetime,
        unavailable: str = "the run's history is unavailable",
    ) -> tuple[str, Optional[str]]:
        """The outcome, "allow", "deny", "stop" or "quarantine", and the reason for any but allow. Without history the
        call is denied, never allowed."""
        subject = f"tool call '{quoted_identity(name)}'"
        if history is None:
            return "deny", f"{subject} is denied: {unavailable}, and Dogwood rules need it"
        action = tool_action(name)
        if action not in self.policies.declared_inputs:
            return "deny", f"{subject} is denied: the Dogwood schema declares no action '{quoted_identity(action)}'"
        timeline, agent = _Timeline(), agent_name or ""
        lines = [line for i, e in enumerate(history) if (line := self._line(timeline, e, agent, i)) is not None]
        # The call being decided is the last event; Dogwood decides it after the ones before it.
        lines.append(self._line(timeline, Event("request", action, now, arguments), agent, len(history)) or "")
        try:
            verdicts = self.policies.verdicts(lines)
        except ValueError as exc:
            return "deny", f"{subject} is denied: Dogwood could not read the run's history: {_message(exc)}"
        if not verdicts:
            return "deny", f"{subject} is denied: Dogwood gave no decision"
        if verdicts[-1]["verdict"] == "allow":
            return "allow", None
        return self.policies.reason(verdicts[-1], subject)
