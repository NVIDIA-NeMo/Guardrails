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
from synthesis.catalog import (
    CLASS_DESCRIPTIONS,
    CLASS_TO_FACTORY,
    CLASS_TO_OWASP,
    RULE_FACTORIES,
    RuleCandidate,
    classes_for_owasp,
    owasp_tags,
)
from synthesis.findings import Finding, load_findings
from synthesis.proposals import (
    UNTARGETED,
    cluster_uncatalogued,
    dropped_findings,
    find_gaps,
    format_factory_prompt,
    synthesize,
)
from synthesis.review import apply, load_approved, write_review_queue

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FINDINGS = os.path.join(os.path.dirname(HERE), "synthesis", "sample_findings.json")


def _finding(attack_class: str, tool: str = "run_shell", **params) -> Finding:
    return Finding(
        id=f"id-{attack_class}",
        title=f"{attack_class} technique",
        source="https://example.org/x",
        attack_class=attack_class,
        affected_tools=(tool,),
        suggested_params=params,
    )


def test_catalog_is_internally_consistent():
    # Every taxonomy class maps to a real factory and carries a control definition
    # for the LLM extractor. Guards against a factory being added to one map but
    # not the others (the exact drift that leaves a class silently undefined).
    assert set(CLASS_DESCRIPTIONS) == set(CLASS_TO_FACTORY)
    assert all(key in RULE_FACTORIES for key in CLASS_TO_FACTORY.values())


def test_every_catalogued_class_has_owasp_tags():
    # OWASP mapping must cover exactly the catalogued classes — the same no-drift
    # guard as the factory/description maps. `novel` is deliberately excluded.
    assert set(CLASS_TO_OWASP) == set(CLASS_TO_FACTORY)


def test_owasp_tags_use_garak_tag_format():
    # Tags must be garak's literal `owasp:llmNN` (01-10) so findings join directly
    # against garak probe/hitlog tags.
    valid = {f"owasp:llm{n:02d}" for n in range(1, 11)}
    for cls, tags in CLASS_TO_OWASP.items():
        assert tags, f"{cls} has no OWASP tags"
        assert all(tag in valid for tag in tags), f"{cls} has a non-OWASP tag: {tags}"


def test_owasp_primary_category_per_class():
    # The first tag is the primary category; spot-check the ones whose primary is
    # distinct from the LLM06 Excessive-Agency backbone.
    assert owasp_tags("unbounded-arg")[0] == "owasp:llm10"  # Unbounded Consumption
    assert owasp_tags("disallowed-target")[0] == "owasp:llm03"  # Supply Chain
    assert owasp_tags("argument-injection")[0] == "owasp:llm05"  # Improper Output Handling
    assert owasp_tags("privilege-escalation")[0] == "owasp:llm06"  # Excessive Agency


def test_owasp_tags_returns_empty_for_uncatalogued():
    # A novel/unknown class gets no fabricated category.
    assert owasp_tags("novel") == ()
    assert owasp_tags("does-not-exist") == ()


def test_classes_for_owasp_reverse_lookup():
    # Reverse of CLASS_TO_OWASP: llm10 is unique to unbounded-arg; llm06 is the
    # Excessive-Agency backbone carried by several classes; an unused tag -> ().
    assert classes_for_owasp("owasp:llm10") == ("unbounded-arg",)
    assert {"ownership-bypass", "privilege-escalation", "disallowed-pattern"} <= set(classes_for_owasp("owasp:llm06"))
    assert classes_for_owasp("owasp:llm99") == ()


def test_disallowed_pattern_factory_materializes_into_a_blocklist_rule():
    candidate = RuleCandidate(
        finding_id="x",
        source="s",
        tool="http_request",
        factory_key="deny_arg_matching",
        params={"arg_name": "url", "pattern": r"169\.254\.169\.254"},
    )
    assert candidate.is_valid()
    rule = candidate.materialize()
    who = Principal("p")
    assert rule(ToolCall("http_request", {"url": "http://169.254.169.254/latest"}), who) is not None
    assert rule(ToolCall("http_request", {"url": "https://api.example.com"}), who) is None
    # A missing argument passes — a blocklist only fires on a present, matching value.
    assert rule(ToolCall("http_request", {}), who) is None


def test_prefix_ownership_factory_materializes_into_a_prefix_rule():
    candidate = RuleCandidate(
        finding_id="x",
        source="s",
        tool="write_file",
        factory_key="require_arg_prefix",
        params={"arg_name": "path", "owned_attr": "owned_paths"},
    )
    assert candidate.is_valid()
    rule = candidate.materialize()
    who = Principal("p", attributes={"owned_paths": frozenset({"/workspace/alice/"})})
    assert rule(ToolCall("write_file", {"path": "/workspace/alice/notes.txt"}), who) is None
    assert rule(ToolCall("write_file", {"path": "/workspace/bob/secret"}), who) is not None
    # A missing required argument blocks (fail closed).
    assert rule(ToolCall("write_file", {}), who) is not None


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
        tool="git_push",
        factory_key="require_owns_arg",
        params={"wrong_kwarg": 1},
    )
    assert not candidate.is_valid()
    assert "do not fit" in candidate.validation_error()


def test_valid_candidate_materializes_into_a_working_rule():
    candidate = RuleCandidate(
        finding_id="x",
        source="s",
        tool="run_shell",
        factory_key="max_numeric_arg",
        params={"arg_name": "timeout_seconds", "ceiling": 300},
    )
    assert candidate.is_valid()
    rule = candidate.materialize()
    who = Principal("p")
    assert rule(ToolCall("run_shell", {"timeout_seconds": 600}), who) is not None
    assert rule(ToolCall("run_shell", {"timeout_seconds": 100}), who) is None


def test_find_gaps_reports_unpoliced_tools():
    guard = ToolCallGuard({"read_file": ToolPolicy(allowed_roles=frozenset({"developer"}))})
    gaps = find_gaps(guard, ["read_file", "run_shell", "write_file"])
    assert {g.tool for g in gaps} == {"run_shell", "write_file"}


def test_review_queue_starts_all_unapproved(tmp_path):
    candidates = synthesize([_finding("unbounded-arg", arg_name="timeout_seconds", ceiling=300)])
    path = write_review_queue(candidates, [], str(tmp_path / "queue.json"))
    payload = json.loads(open(path).read())
    assert payload["candidates"]
    assert all(entry["approved"] is False for entry in payload["candidates"])


def test_finding_to_dict_roundtrips():
    finding = _finding("novel", tool="http_request", arg_name="url")
    assert Finding.from_dict(finding.to_dict()) == finding


def test_review_queue_records_uncatalogued_for_triage(tmp_path):
    findings = [
        _finding("unbounded-arg", arg_name="timeout_seconds", ceiling=300),
        _finding("novel", tool="http_request"),
    ]
    candidates = synthesize(findings)
    uncatalogued = dropped_findings(findings)
    path = write_review_queue(candidates, [], str(tmp_path / "queue.json"), uncatalogued=uncatalogued)
    payload = json.loads(open(path).read())

    assert [c["finding_id"] for c in payload["candidates"]] == ["id-unbounded-arg"]
    assert [u["id"] for u in payload["uncatalogued"]] == ["id-novel"]
    entry = payload["uncatalogued"][0]
    assert entry["triaged"] is False
    assert entry["attack_class"] == "novel"
    assert entry["affected_tools"] == ["http_request"]


def test_write_review_queue_defaults_uncatalogued_empty(tmp_path):
    path = write_review_queue([], [], str(tmp_path / "queue.json"))
    payload = json.loads(open(path).read())
    assert payload["uncatalogued"] == []


def test_load_approved_never_loads_uncatalogued_findings(tmp_path):
    # Even a maliciously hand-edited "approved" flag on an uncatalogued entry must
    # not turn it into a rule: load_approved reads only `candidates`.
    queue = {
        "candidates": [],
        "coverage_gaps": [],
        "uncatalogued": [
            {
                "id": "id-novel",
                "title": "novel technique",
                "source": "s",
                "attack_class": "novel",
                "affected_tools": ["http_request"],
                "suggested_params": {},
                "evidence": "",
                "triaged": True,
                "approved": True,
            }
        ],
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue))
    assert load_approved(str(path)) == []


def _novel(fid: str, tool: str | None) -> Finding:
    return Finding(
        id=fid,
        title=f"{fid} technique",
        source="https://example.org/x",
        attack_class="novel",
        affected_tools=(tool,) if tool else (),
    )


def test_cluster_uncatalogued_groups_and_ranks_by_tool():
    findings = [
        _novel("n1", "http_request"),
        _novel("n2", "http_request"),
        _novel("n3", "run_shell"),
        _finding("unbounded-arg", arg_name="timeout_seconds", ceiling=300),  # catalogued -> excluded
    ]
    clusters = cluster_uncatalogued(findings)

    # ranked by count desc, so the tool with two novel findings comes first
    assert [(c.tool, c.count) for c in clusters] == [("http_request", 2), ("run_shell", 1)]
    top = clusters[0]
    assert top.attack_classes == ("novel",)
    assert top.finding_ids == ("n1", "n2")
    assert top.examples == ("n1 technique", "n2 technique")


def test_cluster_uncatalogued_min_count_filters():
    findings = [_novel("n1", "http_request"), _novel("n2", "http_request"), _novel("n3", "run_shell")]
    clusters = cluster_uncatalogued(findings, min_count=2)
    assert [c.tool for c in clusters] == ["http_request"]


def test_cluster_uncatalogued_buckets_untargeted_findings():
    clusters = cluster_uncatalogued([_novel("n1", None)])
    assert [c.tool for c in clusters] == [UNTARGETED]


def test_format_factory_prompt_when_empty():
    assert "No uncatalogued findings" in format_factory_prompt([])


def test_format_factory_prompt_renders_a_new_factory_prompt():
    clusters = cluster_uncatalogued([_novel("n1", "http_request"), _novel("n2", "http_request")])
    report = format_factory_prompt(clusters)
    assert "http_request: 2 uncatalogued finding(s)" in report
    assert "consider a new rule factory" in report


def test_load_approved_returns_only_approved_and_valid(tmp_path):
    queue = {
        "candidates": [
            {
                "approved": True,
                "finding_id": "good",
                "source": "s",
                "tool": "run_shell",
                "factory_key": "max_numeric_arg",
                "params": {"arg_name": "timeout_seconds", "ceiling": 300},
            },
            {  # approved but invalid -> must be filtered out
                "approved": True,
                "finding_id": "bad",
                "source": "s",
                "tool": "run_shell",
                "factory_key": "not_a_factory",
                "params": {},
            },
            {  # valid but not approved -> must be filtered out
                "approved": False,
                "finding_id": "unapproved",
                "source": "s",
                "tool": "run_shell",
                "factory_key": "max_numeric_arg",
                "params": {"arg_name": "timeout_seconds", "ceiling": 100},
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
            "run_shell": ToolPolicy(
                allowed_roles=frozenset({"developer"}),
                rules=[max_numeric_arg("timeout_seconds", ceiling=3600)],
            )
        }
    )
    approved = [
        RuleCandidate(
            finding_id="a",
            source="s",
            tool="run_shell",
            factory_key="max_numeric_arg",
            params={"arg_name": "timeout_seconds", "ceiling": 300},
        ),
        RuleCandidate(
            finding_id="b",
            source="s",
            tool="write_file",
            factory_key="require_principal_attr",
            params={"attr_name": "elevated"},
        ),
    ]
    new_guard = ToolCallGuard(apply(approved, guard))

    developer = Principal("c", roles=frozenset({"developer"}))
    # Existing tool keeps its role grant but gains the tighter ceiling.
    tightened = new_guard.authorize(ToolCall("run_shell", {"timeout_seconds": 600}), developer)
    assert not tightened.allowed

    # Newly-policied tool is fail-closed on roles: an approved rule cannot, by
    # itself, grant access.
    new_tool = new_guard.authorize(ToolCall("write_file", {"path": "x"}), developer)
    assert not new_tool.allowed
    assert new_guard.policy_for("write_file").allowed_roles == frozenset()


def test_sample_findings_fixture_loads():
    findings = load_findings(SAMPLE_FINDINGS)
    assert len(findings) == 3
    assert any(f.attack_class == "speculative-prompt-loop" for f in findings)
