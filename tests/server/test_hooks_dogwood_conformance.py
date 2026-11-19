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

"""The hooks server's Dogwood evaluator against the reference language's own examples.

Each bundle under `dogwood_conformance/` is a policy, an event trace and the verdicts the reference `dogwood replay`
CLI gives for it (see its README). The trace is replayed through this server's evaluator, with history kept per
principal as the reference's default event schema does, and the verdicts compared.

A denied call stays in history as a `request`, an attempt, as the reference keeps it; that is the server's default
(`hooks_server.dogwood.denied_calls: request`). With `denied_calls: error`, AgentCore's documented reading, a denied call
is an `error` event instead, and `alert_login_in_last_hour` is the one bundle whose verdicts change.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nemoguardrails.server.hooks.cedar_policy import CedarPolicySet
from nemoguardrails.server.hooks.dogwood_policy import Event, rewrite

cedarpy = pytest.importorskip("cedarpy")

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
TOKEN = re.compile(
    r'\s*(?:(?P<s>"(?:[^"\\]|\\.)*")|(?P<n>-?[0-9]+(?:\.[0-9]+)?)|(?P<w>[A-Za-z_][A-Za-z0-9_]*)|(?P<p>::|[(){}\[\],:]))'
)


class Reader:
    def __init__(self, text):
        self.toks = [(m.lastgroup, m.group(m.lastgroup)) for m in TOKEN.finditer(text)]
        self.i = 0

    def peek(self, k=0):
        return self.toks[self.i + k][1] if self.i + k < len(self.toks) else None

    def take(self, want=None):
        t = self.toks[self.i][1]
        if want is not None and t != want:
            raise ValueError(f"expected {want} got {t}")
        self.i += 1
        return t

    def name_path(self):
        parts = [self.take()]
        while self.peek() == "::" and self.peek(1) is not None and not self.peek(1).startswith('"'):
            self.take("::")
            parts.append(self.take())
        return parts

    def value(self):
        t = self.peek()
        if t == "{":
            self.take("{")
            rec = {}
            while self.peek() != "}":
                k = self.take()
                k = json.loads(k) if k.startswith('"') else k
                self.take(":")
                rec[k] = self.value()
                if self.peek() == ",":
                    self.take(",")
            self.take("}")
            return rec
        if t == "[":
            self.take("[")
            items = []
            while self.peek() != "]":
                items.append(self.value())
                if self.peek() == ",":
                    self.take(",")
            self.take("]")
            return items
        if t.startswith('"'):
            return json.loads(self.take())
        if re.fullmatch(r"-?[0-9]+", t):
            return int(self.take())
        if t in ("true", "false"):
            return self.take() == "true"
        path = self.name_path()
        if self.peek() == "::":  # entity uid
            self.take("::")
            return {"__entity": {"type": "::".join(path), "id": json.loads(self.take())}}
        if self.peek() == "(":  # extension call
            self.take("(")
            arg = self.value()
            self.take(")")
            return {"__extn": {"fn": path[-1], "arg": arg}}
        raise ValueError(f"unreadable value at {t}")

    def fields(self):
        self.take("(")
        rec = {}
        while self.peek() != ")":
            k = self.take()
            self.take(":")
            rec[k] = self.value()
            if self.peek() == ",":
                self.take(",")
        self.take(")")
        return rec


def parse_line(line):
    m = re.match(r"@(-?[0-9]+)\s+(.*)$", line.strip())
    t, rest = int(m.group(1)), m.group(2)
    r = Reader(rest)
    env = {}
    while r.peek() in ("scope", "request_context"):
        name = r.take()
        env[name] = r.fields()
    path = r.name_path()
    r.take("::")
    action = json.loads(r.take())
    r.take("::")
    kind = r.take()
    rec = r.fields()
    return {"t": t, "env": env, "ns": path[0], "action": action, "kind": kind, "rec": rec}


def entity_uid(v):
    return {"type": v["__entity"]["type"], "id": v["__entity"]["id"]}


def replay(example: Path, denied_request_stays: bool) -> tuple[list[str], list[str]]:
    """Our verdicts for the bundle's trace, and the reference CLI's from `expected.out`."""
    text = (example / "policy.dw").read_text()
    lowered, blocks = rewrite(text, example.name, 0, namespace=None, require_scope=False)
    policies = CedarPolicySet.parse([(example.name, lowered)], None, section=example.name, outcomes_allowed=True)
    history = {}  # principal -> list[Event]
    got = []
    for line in (example / "trace.log").read_text().splitlines():
        if not line.strip():
            continue
        ev = parse_line(line)
        now = EPOCH + timedelta(seconds=ev["t"])
        inputs = ev["rec"].get("input", {})
        principal = ev["env"].get("scope", {}).get("principal") or ev["rec"].get("callerPrincipal")
        resource = ev["env"].get("scope", {}).get("resource") or ev["rec"].get("callerResource")
        key = json.dumps(principal, sort_keys=True)
        past = history.setdefault(key, [])
        if ev["kind"] != "request":
            past.append(Event(ev["kind"], ev["action"], now, inputs, ev["ns"]))
            continue
        current = Event("request", ev["action"], now, inputs, ev["ns"])
        events = [*past, current]
        ctx_input = ev["env"].get("request_context", {}).get("input", inputs)
        temporal = {b.slot: b.holds(events, ctx_input, now) for b in blocks}
        context = {k: v for k, v in ev["env"].get("request_context", {}).items()}
        context["temporal"] = temporal
        req = {
            "principal": entity_uid(principal),
            "action": {"type": f"{ev['ns']}::Action", "id": ev["action"]},
            "resource": entity_uid(resource),
            "context": context,
        }
        ents = [
            {"uid": entity_uid(principal), "attrs": {}, "parents": []},
            {"uid": entity_uid(resource), "attrs": {}, "parents": []},
        ]
        try:
            res = cedarpy.is_authorized(req, policies.text, ents)
            allowed = res.decision.value == "Allow" and not res.diagnostics.errors
        except Exception:
            allowed = False
        got.append(f"@{ev['t']}: {'ALLOW' if allowed else 'DENY'}")
        past.append(
            current if (allowed or denied_request_stays) else Event("error", ev["action"], now, inputs, ev["ns"])
        )
    want = [
        re.match(r"@(-?[0-9]+) \(time point [0-9]+\): (ALLOW|DENY)", line).expand(r"@\1: \2")
        for line in (example / "expected.out").read_text().splitlines()
        if re.match(r"@-?[0-9]+ \(time point", line)
    ]
    return got, want


BUNDLES = sorted(p for p in (Path(__file__).parent / "dogwood_conformance").iterdir() if p.is_dir())
DENIED_REQUEST_MATTERS = {"alert_login_in_last_hour"}


@pytest.mark.parametrize("bundle", BUNDLES, ids=lambda p: p.name)
def test_the_default_gives_the_reference_verdicts(bundle):
    got, want = replay(bundle, denied_request_stays=True)
    assert want and got == want


@pytest.mark.parametrize("bundle", BUNDLES, ids=lambda p: p.name)
def test_denied_calls_as_errors_differ_only_where_known(bundle):
    got, want = replay(bundle, denied_request_stays=False)
    assert (got != want) == (bundle.name in DENIED_REQUEST_MATTERS), (got, want)
