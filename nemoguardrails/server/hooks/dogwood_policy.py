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

"""The hooks server's Dogwood rules: tool-call policies whose `when temporal` block reads what already happened in the
run, in the language and vocabulary AgentCore Gateway's temporal policies use.

The server evaluates a fixed subset of Dogwood's temporal forms itself, from the run's history in the decision record:
`formerly within`, `!P since within Q`, and `count` and `sum` over `formerly within`. Each `when temporal { ... }`
block is computed to true or false and replaced with `when { context.temporal.pN }`, and Cedar then decides the
policies as it decides the server's Cedar rules. Anything outside the subset stops the server at start.

The subset and what it leaves out are in `docs/run-rails/using-hooks-server/dogwood.mdx`.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

from nemoguardrails.guardrails.guardrails_types import quoted_identity
from nemoguardrails.server.hooks.cedar_policy import CedarPolicyError, CedarPolicySet, _agent_entity, cedar_record

log = logging.getLogger(__name__)

ACTION_TYPE = "AgentCore::Action"
GATEWAY_TYPE = "AgentCore::Gateway"
EVENT_KINDS = ("request", "response", "error")
MAX_WINDOW = timedelta(hours=24)
MAX_OPERATORS = 3
SUFFIXES = (".dw", ".dogwood")
_UNITS = {"s": 1, "m": 60, "h": 3600}
_COMPARE = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}


class DogwoodPolicyError(CedarPolicyError):
    """A Dogwood policy the server can't evaluate, found when it starts."""


# --- Tokens -------------------------------------------------------------------------------------------------------

_TOKEN = re.compile(
    r'(?P<space>\s+|//[^\n]*)|(?P<string>"(?:[^"\\]|\\.)*")|(?P<word>[A-Za-z_][A-Za-z0-9_]*)'
    r"|(?P<number>[0-9]+[A-Za-z]*)|(?P<op>::|&&|\|\||==|!=|>=|<=|[(){}\[\],.:;!<>=@+\-*])"
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int
    end: int


def _tokens(text: str, name: str) -> list[_Token]:
    tokens, position = [], 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise DogwoodPolicyError(f"'{name}': can't read the character {text[position]!r} at offset {position}")
        if match.lastgroup != "space":
            tokens.append(_Token(match.lastgroup or "", match.group(), match.start(), match.end()))
        position = match.end()
    return tokens


# --- The supported forms ------------------------------------------------------------------------------------------

# A field constraint's value: the current call's input, a literal, or the name a `sum` binds.
_Value = Union[tuple[str, str], tuple[str, Any]]


@dataclass(frozen=True)
class Pattern:
    """`AgentCore::Action::"A"::kind{ eventResource: resource, input.f: value, ... }`."""

    action: str
    kind: str
    constraints: tuple[tuple[str, _Value], ...]
    namespace: str = "AgentCore"

    def matches(self, event: "Event", inputs: Mapping[str, Any]) -> bool:
        if event.kind != self.kind or event.action != self.action or event.namespace != self.namespace:
            return False
        for name, (source, value) in self.constraints:
            if name not in event.inputs:
                return False
            if source == "input" and (value not in inputs or event.inputs[name] != inputs[value]):
                return False
            # `_` asks only that the field is there.
            if source == "literal" and event.inputs[name] != value:
                return False
            if source == "bound" and (not isinstance(event.inputs[name], int) or isinstance(event.inputs[name], bool)):
                return False
        return True


@dataclass(frozen=True)
class Formerly:
    window: timedelta
    pattern: Pattern

    def holds(self, events: Sequence["Event"], inputs: Mapping[str, Any], now: datetime) -> bool:
        return any(now - e.at <= self.window and self.pattern.matches(e, inputs) for e in events)


@dataclass(frozen=True)
class Since:
    """`!P since within W Q`: a `Q` within the window, and no `P` after it."""

    window: timedelta
    absent: Pattern
    anchor: Pattern

    def holds(self, events: Sequence["Event"], inputs: Mapping[str, Any], now: datetime) -> bool:
        for anchor in events:
            if now - anchor.at <= self.window and self.anchor.matches(anchor, inputs):
                if not any(e.at > anchor.at and self.absent.matches(e, inputs) for e in events):
                    return True
        return False


@dataclass(frozen=True)
class Aggregate:
    """`count` or `sum` over the events `formerly within` the window that match, compared with a number."""

    function: str
    window: timedelta
    pattern: Pattern
    operator: str
    limit: int
    summed: Optional[str] = None

    def holds(self, events: Sequence["Event"], inputs: Mapping[str, Any], now: datetime) -> bool:
        matched = [e for e in events if now - e.at <= self.window and self.pattern.matches(e, inputs)]
        value = len(matched) if self.function == "count" else sum(e.inputs[self.summed or ""] for e in matched)
        return _COMPARE[self.operator](value, self.limit)


Condition = Union[Formerly, Since, Aggregate]


@dataclass(frozen=True)
class Event:
    """One event of the run's history: a tool call allowed (`request`), completed (`response`) or denied or failed
    (`error`), with the inputs a policy reads."""

    kind: str
    action: str
    at: datetime
    inputs: Mapping[str, Any] = field(default_factory=dict)
    namespace: str = "AgentCore"


# --- Parsing a temporal block -------------------------------------------------------------------------------------


class _Parser:
    def __init__(
        self, tokens: list[_Token], where: str, namespace: Optional[str] = "AgentCore", require_scope: bool = True
    ) -> None:
        # AgentCore's actions and its rule that every pattern says `eventResource: resource`, unless the caller, such as
        # a conformance check over the reference examples, reads another namespace with the language's own rules.
        self.tokens, self.position, self.where = tokens, 0, where
        self.namespace, self.require_scope = namespace, require_scope
        self.bound: set[str] = set()

    def fail(self, message: str) -> DogwoodPolicyError:
        near = self.tokens[self.position].text if self.position < len(self.tokens) else "the end of the block"
        return DogwoodPolicyError(f"{self.where}: {message} (at {near!r})")

    def peek(self, offset: int = 0) -> Optional[str]:
        index = self.position + offset
        return self.tokens[index].text if index < len(self.tokens) else None

    def take(self, *expected: str) -> str:
        text = self.peek()
        if text is None or (expected and text not in expected):
            raise self.fail(f"expected {' or '.join(repr(e) for e in expected) or 'more'}")
        self.position += 1
        return text

    def word(self) -> str:
        if self.position >= len(self.tokens) or self.tokens[self.position].kind != "word":
            raise self.fail("expected a name")
        return self.take()

    def conjunction(self) -> list[Condition]:
        conditions = [self.condition()]
        while self.peek() == "&&":
            self.take("&&")
            conditions.append(self.condition())
        if self.peek() is not None:
            raise self.fail("only conditions joined with && are supported in a temporal block")
        return conditions

    def condition(self) -> Condition:
        if self.peek() == "formerly":
            self.take("formerly")
            return Formerly(self.window(), self.pattern())
        if self.peek() == "!":
            self.take("!")
            absent = self.pattern()
            self.take("since")
            window = self.window()
            return Since(window, absent, self.pattern())
        if self.peek() == "exists":
            return self.aggregate()
        raise self.fail("expected formerly, !... since, or exists")

    def window(self) -> timedelta:
        self.take("within")
        token = self.take()
        match = re.fullmatch(r"([1-9][0-9]*)([smh])", token)
        if match is None:
            raise self.fail(f"the window {token!r} is not a whole number of seconds, minutes or hours")
        window = timedelta(seconds=int(match.group(1)) * _UNITS[match.group(2)])
        if window > MAX_WINDOW:
            raise self.fail(f"the window {token} is longer than 24 hours, the most history that is kept")
        return window

    def pattern(self) -> Pattern:
        namespace = self.word() if self.namespace is None else self.take(self.namespace)
        self.take("::")
        self.take("Action")
        self.take("::")
        token = self.take()
        if not token.startswith('"'):
            raise self.fail("expected the action's name in quotes")
        action = _unquote(token)
        self.take("::")
        kind = self.word()
        if kind not in EVENT_KINDS:
            raise self.fail(f"the event kind {kind!r} is not request, response or error")
        self.take("{")
        constraints: list[tuple[str, _Value]] = []
        scoped = False
        while self.peek() != "}":
            name = self.word()
            if name == "eventResource":
                self.take(":")
                self.take("resource")
                scoped = True
            elif name == "input":
                self.take(".")
                field_name = self.word()
                self.take(":")
                constraints.append((field_name, self.value()))
            else:
                raise self.fail(f"the field {name!r} is not supported; only eventResource and input.<name> are")
            if self.peek() == ",":
                self.take(",")
        self.take("}")
        if not scoped and self.require_scope:
            raise self.fail("every event pattern must say eventResource: resource")
        return Pattern(action, kind, tuple(constraints), namespace)

    def value(self) -> _Value:
        token = self.take()
        if token == "context":
            self.take(".")
            self.take("input")
            self.take(".")
            return ("input", self.word())
        if token == "_":
            return ("any", None)
        if token in ("true", "false"):
            return ("literal", token == "true")
        if token.startswith('"'):
            return ("literal", _unquote(token))
        if token == "-" and (self.peek() or "").isdigit():
            return ("literal", -int(self.take()))
        if token.isdigit():
            return ("literal", int(token))
        if token in self.bound:
            return ("bound", token)
        raise self.fail(f"the value {token!r} is not context.input.<name>, a literal or a bound name")

    def aggregate(self) -> Aggregate:
        self.take("exists")
        self.take("(")
        result = self.word()
        self.take(":")
        self.take("Long")
        self.take(")")
        self.take(".")
        # The body may be wrapped in one more pair of parentheses, as the language's own examples write it.
        wrapped = self.peek() == "(" and self.peek(1) == "("
        if wrapped:
            self.take("(")
        self.take("(")
        function = self.take("count", "sum")
        summed = None
        if function == "sum":
            summed = self.word()
            self.take("for")
            self.take("(")
            self.take(summed)
            self.take(":")
            self.take("Long")
            self.take(")")
            self.take(",")
            self.bound.add(summed)
        else:
            self.take("for")
        self.take("(")
        point = self.word()
        self.take(":")
        self.take("Timepoint")
        self.take(")")
        self.take(".")
        self.take("where")
        self.take("(")
        self.take("formerly")
        window = self.window()
        self.take("(")
        pattern = self.pattern()
        self.take("&&")
        self.take("tp")
        self.take("(")
        self.take(point)
        self.take(")")
        self.take(")")
        self.take(")")
        self.take(")")
        self.take("==")
        self.take(result)
        self.take("&&")
        self.take(result)
        operator = self.take(*_COMPARE)
        limit = self.value()
        if wrapped:
            self.take(")")
        if limit[0] != "literal" or not isinstance(limit[1], int) or isinstance(limit[1], bool):
            raise self.fail("an aggregate is compared with a whole number")
        self.bound.discard(summed or "")
        summed_field = None
        if summed is not None:
            summed_field = next((name for name, value in pattern.constraints if value == ("bound", summed)), None)
            if summed_field is None:
                raise self.fail(f"sum {summed} names no input.<name>: {summed} in its event pattern")
        return Aggregate(function, window, pattern, operator, limit[1], summed_field)


def _unquote(token: str) -> str:
    return token[1:-1].replace('\\"', '"').replace("\\\\", "\\")


# --- Loading ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Block:
    slot: str
    conditions: tuple[Condition, ...]

    def holds(self, events: Sequence[Event], inputs: Mapping[str, Any], now: datetime) -> bool:
        return all(condition.holds(events, inputs, now) for condition in self.conditions)


def rewrite(
    text: str, name: str, first_slot: int, *, namespace: Optional[str] = "AgentCore", require_scope: bool = True
) -> tuple[str, list[_Block]]:
    """*text* with each `when temporal { ... }` block replaced by `when { context.temporal.pN }`, and the blocks."""
    tokens = _tokens(text, name)
    blocks: list[_Block] = []
    pieces: list[str] = []
    copied = 0
    effect: Optional[str] = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.kind == "word"
            and token.text in ("permit", "forbid")
            and (index == 0 or tokens[index - 1].text in (";", ")"))
        ):
            effect = token.text
        if token.text == "temporal" and token.kind == "word":
            if (
                index == 0
                or tokens[index - 1].text != "when"
                or index + 1 >= len(tokens)
                or tokens[index + 1].text != "{"
            ):
                raise DogwoodPolicyError(f"'{name}': temporal is only supported as `when temporal {{ ... }}`")
            depth, end = 0, index + 1
            while end < len(tokens):
                depth += {"{": 1, "}": -1}.get(tokens[end].text, 0)
                if depth == 0:
                    break
                end += 1
            if depth != 0:
                raise DogwoodPolicyError(f"'{name}': a `when temporal` block is not closed")
            slot = f"p{first_slot + len(blocks)}"
            parser = _Parser(
                tokens[index + 2 : end], f"'{name}', temporal block {len(blocks) + 1}", namespace, require_scope
            )
            conditions = parser.conjunction()
            if len(conditions) > MAX_OPERATORS:
                raise DogwoodPolicyError(
                    f"'{name}': a temporal block has {len(conditions)} conditions; at most {MAX_OPERATORS} are allowed"
                )
            if effect == "forbid" and any(_reads(condition, "response") for condition in conditions):
                log.warning(
                    "'%s': a forbid reads ::response events, which only Claude Code's PostToolUse hook reports; on "
                    "other hook paths it never matches",
                    name,
                )
            blocks.append(_Block(slot, tuple(conditions)))
            pieces.append(text[copied : token.start] + f"{{ context.temporal.{slot} }}")
            copied = tokens[end].end
            index = end
        index += 1
    pieces.append(text[copied:])
    return "".join(pieces), blocks


def _reads(condition: Condition, kind: str) -> bool:
    patterns = [condition.pattern] if not isinstance(condition, Since) else [condition.absent, condition.anchor]
    return any(pattern.kind == kind for pattern in patterns)


def _patterns(condition: Condition) -> list[Pattern]:
    return [condition.anchor, condition.absent] if isinstance(condition, Since) else [condition.pattern]


def tool_action(tool_name: str) -> str:
    """The AgentCore action of a hook's tool: a Claude Code MCP tool `mcp__<server>__<tool>` is `<server>___<tool>`,
    the `<target>___<tool>` form AgentCore and Agentform use; any other name is used as it is."""
    match = re.fullmatch(r"mcp__([^_]+(?:_[^_]+)*)__(.+)", tool_name)
    return f"{match.group(1)}___{match.group(2)}" if match else tool_name


@dataclass(frozen=True)
class DogwoodToolRules:
    """`hooks_server.dogwood`: each tool call is asked as `AgentCore::Action::"<tool>"` on the configured gateway,
    with the run's history computed into `context.temporal`."""

    policies: CedarPolicySet
    blocks: tuple[_Block, ...]
    gateway: str
    # For each action, the input fields some policy reads from its events: the only values the record keeps.
    read_inputs: Mapping[str, frozenset[str]]

    @classmethod
    def load(cls, folder: Path, gateway: str) -> "DogwoodToolRules":
        section = "hooks_server.dogwood"
        if not folder.is_dir():
            raise DogwoodPolicyError(f"{section}.policies names '{folder}', which is not a folder")
        files = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix in SUFFIXES)
        if not files:
            raise DogwoodPolicyError(f"{section}.policies names '{folder}', which has no .dw or .dogwood file")
        named: list[tuple[str, str]] = []
        blocks: list[_Block] = []
        for path in files:
            name = path.relative_to(folder).as_posix()
            text, found = rewrite(path.read_text(encoding="utf-8"), name, len(blocks))
            named.append((name, text))
            blocks.extend(found)
        policies = CedarPolicySet.parse(named, None, section=section, outcomes_allowed=True, language="Dogwood")
        read: dict[str, set[str]] = {}
        for block in blocks:
            for condition in block.conditions:
                for pattern in _patterns(condition):
                    read.setdefault(pattern.action, set()).update(name for name, _ in pattern.constraints)
        return cls(policies, tuple(blocks), gateway, {action: frozenset(names) for action, names in read.items()})

    @property
    def quarantines(self) -> bool:
        return self.policies.quarantines

    def kept_inputs(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """The arguments of a call that some policy reads from history, and only those."""
        names = self.read_inputs.get(tool_action(tool_name), frozenset())
        return {name: arguments[name] for name in sorted(names) if name in arguments}

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
        # The call being decided is an event of its own history, as AgentCore counts it.
        events = [*history, Event("request", action, now, self.kept_inputs(name, arguments))]
        temporal = {block.slot: block.holds(events, arguments, now) for block in self.blocks}
        agent = _agent_entity(agent_name or "", agent_digest)
        gateway = {"uid": {"type": GATEWAY_TYPE, "id": self.gateway}, "attrs": {}, "parents": []}
        context = {"input": cedar_record(arguments), "temporal": temporal}
        decision = self.policies.decide(
            agent["uid"], action, gateway["uid"], context, [agent, gateway], action_type=ACTION_TYPE
        )
        if decision.allowed:
            return "allow", None
        return decision.outcome, decision.reason(subject)
