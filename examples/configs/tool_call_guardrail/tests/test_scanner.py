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

"""Offline tests for the field scanner.

Exercises the trusted orchestration in `scan()` — the guarantees that must hold
regardless of which (replaceable) extractor produced the techniques: grounding to
the real registry, provenance, and deduplication — plus the deterministic
`KeywordExtractor` stand-in over the sample documents.
"""

import os

from scanner.scan import (
    ExtractedTechnique,
    KeywordExtractor,
    ScanContext,
    SourceDoc,
    scan,
)
from synthesis.catalog import CLASS_DESCRIPTIONS, CLASS_TO_FACTORY

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS = os.path.join(os.path.dirname(HERE), "scanner", "sample_docs")

REGISTRY = {
    "read_file": "Read a file from the workspace",
    "write_file": "Create or overwrite a file in the workspace",
    "run_shell": "Execute a shell command in the sandbox",
    "http_request": "Make an outbound HTTP request",
    "git_push": "Push commits to a git remote",
    "install_package": "Install a dependency from a package index",
}


def _ctx(**overrides) -> ScanContext:
    base = dict(
        docs_dir=SAMPLE_DOCS,
        tool_registry=REGISTRY,
        taxonomy=tuple(CLASS_TO_FACTORY),
        class_definitions=dict(CLASS_DESCRIPTIONS),
    )
    base.update(overrides)
    return ScanContext(**base)


class _FixedExtractor:
    """Test double that ignores the document and returns canned techniques."""

    def __init__(self, *techniques: ExtractedTechnique):
        self._techniques = techniques

    def extract(self, doc: SourceDoc, ctx: ScanContext):
        return list(self._techniques)


def test_keyword_scan_over_sample_docs():
    findings = scan(_ctx(), KeywordExtractor())

    by_class = {f.attack_class for f in findings}
    assert by_class == {
        "argument-injection",
        "ownership-bypass",
        "disallowed-target",
        "privilege-escalation",
        "unbounded-arg",
        "novel",
    }
    # The ungrounded "bulk deploy" doc mentions no registered tool and is dropped.
    assert not any("bulk-deploy" in f.id for f in findings)


def test_every_finding_is_grounded_and_has_provenance():
    findings = scan(_ctx(), KeywordExtractor())
    assert findings
    for f in findings:
        assert f.affected_tools  # never empty
        assert all(tool in REGISTRY for tool in f.affected_tools)  # grounded
        assert f.source  # provenance always carried through


def test_params_hint_is_parsed_from_doc():
    findings = scan(_ctx(), KeywordExtractor())
    unbounded = next(f for f in findings if f.attack_class == "unbounded-arg")
    assert unbounded.suggested_params == {"arg_name": "timeout_seconds", "ceiling": 300}


def test_scan_drops_ungrounded_techniques_even_if_extractor_emits_them():
    # The extractor over-reaches and names a tool that isn't in the registry.
    ghost = ExtractedTechnique(
        summary="phantom",
        attack_class="unbounded-arg",
        tool_mentions=["ghost_tool"],
    )
    findings = scan(_ctx(), _FixedExtractor(ghost))
    assert findings == []


def test_scan_keeps_only_registered_tools_when_extractor_overreaches():
    mixed = ExtractedTechnique(
        summary="partly real",
        attack_class="unbounded-arg",
        tool_mentions=["run_shell", "ghost_tool"],
    )
    findings = scan(_ctx(), _FixedExtractor(mixed))
    assert findings  # the grounded mention survives
    for f in findings:
        assert f.affected_tools == ("run_shell",)  # ghost_tool filtered out


def test_known_finding_ids_suppress_re_emission():
    first = scan(_ctx(), KeywordExtractor())
    seen = frozenset(f.id for f in first)
    second = scan(_ctx(known_finding_ids=seen), KeywordExtractor())
    assert second == []
