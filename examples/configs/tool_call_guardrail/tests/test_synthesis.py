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

"""Offline tests for the scanner -> policy bridge.

Covers the trust boundary end to end: findings map to vetted rule factories (and
unknown classes are dropped), candidates validate their params against the factory
signature, the review queue gates every candidate behind explicit approval, and
`apply` merges approvals without ever opening role access on its own.
"""

import json
import os

from policy import (
    Principal,
    ToolCall,
    ToolCallGuard,
    ToolPolicy,
    max_numeric_arg,
)
from synthesis.catalog import CLASS_TO_FACTORY, RuleCandidate
from synthesis.findings import Finding, load_findings
from synthesis.proposals import dropped_findings, find_gaps, synthesize
from synthesis.review import apply, load_approved, write_review_queue

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FINDINGS = os.path.join(os.path.dirname(HERE), "synthesis", "sample_findings.json")


def _finding(attack_class: str, tool: str = "transfer_funds", **params) -> Finding:
    return Finding(
        id=f"id-{attack_class}",
        title=f"{attack_class} technique",
        source="https://example.org/x",
        attack_class=attack_class,
        affected_tools=(tool,),
        suggested_params=params,
    )


def test_synthesize_maps_each_class_to_its_factory():
    findings = [_finding(cls) for cls in CLASS_TO_FACTORY]
    candidates = synthesize(findings)
    got = {c.finding_id: c.factory_key for c in candidates}
    assert got == {f"id-{cls}": CLASS_TO_FACTORY[cls] for cls in CLASS_TO_FACTORY}


def test_synthesize_drops_unknown_class_fail_closed():
    findings = [_finding("speculative-prompt-loop"), _finding("unbounded-arg")]
    candidates = synthesize(findings)
    assert [c.finding_id for c in candidates] == ["id-unbounded-arg"]
    assert [f.id for f in dropped_findings(findings)] == ["id-speculative-prompt-loop"]


def test_candidate_with_unknown_factory_is_invalid():
    candidate = RuleCandidate(finding_id="x", source="s", tool="t", factory_key="not_a_factory")
    assert not candidate.is_valid()
    assert "vetted catalog" in candidate.validation_error()


def test_candidate_with_bad_params_is_invalid():
    candidate = RuleCandidate(
        finding_id="x",
        source="s",
        tool="transfer_funds",
        factory_key="require_owns_arg",
        params={"wrong_kwarg": 1},
    )
    assert not candidate.is_valid()
    assert "do not fit" in candidate.validation_error()


def test_valid_candidate_materializes_into_a_working_rule():
    candidate = RuleCandidate(
        finding_id="x",
        source="s",
        tool="transfer_funds",
        factory_key="max_numeric_arg",
        params={"arg_name": "amount", "ceiling": 5000},
    )
    assert candidate.is_valid()
    rule = candidate.materialize()
    who = Principal("p")
    assert rule(ToolCall("transfer_funds", {"amount": 6000}), who) is not None
    assert rule(ToolCall("transfer_funds", {"amount": 100}), who) is None


def test_find_gaps_reports_unpoliced_tools():
    guard = ToolCallGuard({"read_account": ToolPolicy(allowed_roles=frozenset({"customer"}))})
    gaps = find_gaps(guard, ["read_account", "transfer_funds", "close_account"])
    assert {g.tool for g in gaps} == {"transfer_funds", "close_account"}


def test_review_queue_starts_all_unapproved(tmp_path):
    candidates = synthesize([_finding("unbounded-arg", arg_name="amount", ceiling=5000)])
    path = write_review_queue(candidates, [], str(tmp_path / "queue.json"))
    payload = json.loads(open(path).read())
    assert payload["candidates"]
    assert all(entry["approved"] is False for entry in payload["candidates"])


def test_load_approved_returns_only_approved_and_valid(tmp_path):
    queue = {
        "candidates": [
            {
                "approved": True,
                "finding_id": "good",
                "source": "s",
                "tool": "transfer_funds",
                "factory_key": "max_numeric_arg",
                "params": {"arg_name": "amount", "ceiling": 5000},
            },
            {  # approved but invalid -> must be filtered out
                "approved": True,
                "finding_id": "bad",
                "source": "s",
                "tool": "transfer_funds",
                "factory_key": "not_a_factory",
                "params": {},
            },
            {  # valid but not approved -> must be filtered out
                "approved": False,
                "finding_id": "unapproved",
                "source": "s",
                "tool": "transfer_funds",
                "factory_key": "max_numeric_arg",
                "params": {"arg_name": "amount", "ceiling": 1000},
            },
        ],
        "coverage_gaps": [],
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue))
    approved = load_approved(str(path))
    assert [c.finding_id for c in approved] == ["good"]


def test_apply_merges_into_existing_and_failcloses_new_tools():
    guard = ToolCallGuard(
        {
            "transfer_funds": ToolPolicy(
                allowed_roles=frozenset({"customer"}),
                rules=[max_numeric_arg("amount", ceiling=10_000)],
            )
        }
    )
    approved = [
        RuleCandidate(
            finding_id="a",
            source="s",
            tool="transfer_funds",
            factory_key="max_numeric_arg",
            params={"arg_name": "amount", "ceiling": 5000},
        ),
        RuleCandidate(
            finding_id="b",
            source="s",
            tool="close_account",
            factory_key="require_owns_arg",
            params={"arg_name": "account_id"},
        ),
    ]
    new_guard = ToolCallGuard(apply(approved, guard))

    customer = Principal("c", roles=frozenset({"customer"}))
    # Existing tool keeps its role grant but gains the tighter ceiling.
    tightened = new_guard.authorize(ToolCall("transfer_funds", {"amount": 7000}), customer)
    assert not tightened.allowed

    # Newly-policied tool is fail-closed on roles: an approved rule cannot, by
    # itself, grant access.
    new_tool = new_guard.authorize(ToolCall("close_account", {"account_id": "acct-1"}), customer)
    assert not new_tool.allowed
    assert new_guard.policy_for("close_account").allowed_roles == frozenset()


def test_sample_findings_fixture_loads():
    findings = load_findings(SAMPLE_FINDINGS)
    assert len(findings) == 3
    assert any(f.attack_class == "speculative-prompt-loop" for f in findings)
