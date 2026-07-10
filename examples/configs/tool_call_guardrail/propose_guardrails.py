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

"""Scan advisory/research docs and PROPOSE tool-call guardrails — never apply.

The real adoption path: point it at new advisories, extract findings, synthesize
vetted rule candidates, and write an UNAPPROVED review queue for a human. Writes
`findings.json` + `review_queue.json` to the current directory and applies
nothing — default-deny holds until a human flips a candidate to approved.

    python3 propose_guardrails.py --docs advisories/
    python3 propose_guardrails.py --docs advisories/ --extractor llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_policies import PRINCIPAL_ATTRS, TOOL_REGISTRY, TOOL_SCHEMAS, VULNERABLE_GUARD  # noqa: E402
from scanner.scan import KeywordExtractor, ScanContext, _finding_to_dict, scan  # noqa: E402
from synthesis.catalog import CLASS_DESCRIPTIONS, CLASS_REQUIRED_PARAMS, CLASS_TO_FACTORY  # noqa: E402
from synthesis.proposals import dropped_findings, find_gaps, synthesize  # noqa: E402
from synthesis.review import write_review_queue  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Scan docs and propose tool-call guardrails (no apply).")
    p.add_argument("--docs", default="advisories", help="docs folder to scan")
    p.add_argument("--extractor", choices=("keyword", "llm"), default="keyword", help="keyword (offline) or llm")
    p.add_argument("--model", default="azure/openai/gpt-4o-mini", help="model id for the llm extractor")
    p.add_argument("--base-url", default="https://inference-api.nvidia.com/v1", help="OpenAI-compatible base URL")
    p.add_argument("--api-key-env", default="NVIDIA_LITELLM_KEY", help="env var holding the API key for --base-url")
    p.add_argument("--findings-out", default="findings.json", help="where to write the scanner findings")
    p.add_argument("--queue-out", default="review_queue.json", help="where to write the unapproved review queue")
    args = p.parse_args()

    ctx = ScanContext(
        docs_dir=args.docs,
        tool_registry=dict(TOOL_REGISTRY),
        taxonomy=tuple(CLASS_TO_FACTORY),
        class_definitions=dict(CLASS_DESCRIPTIONS),
        class_params=dict(CLASS_REQUIRED_PARAMS),
        tool_schemas=dict(TOOL_SCHEMAS),
        principal_attrs=tuple(PRINCIPAL_ATTRS),
    )

    catch: tuple = ()
    if args.extractor == "llm":
        try:
            from scanner.llm_extractor import ExtractionError, LLMExtractor, nvidia_chat
        except ImportError:
            from llm_extractor import ExtractionError, LLMExtractor, nvidia_chat
        extractor = LLMExtractor(nvidia_chat(model=args.model, api_key_env=args.api_key_env, base_url=args.base_url))
        catch = (ExtractionError,)
    else:
        extractor = KeywordExtractor()

    try:
        findings = scan(ctx, extractor)
    except catch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with open(args.findings_out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps([_finding_to_dict(f) for f in findings], indent=2) + "\n")

    # Synthesis + review queue — but NEVER apply. Candidates land unapproved; the
    # human flips the ones they trust before any rule takes effect.
    candidates = synthesize(findings)
    gaps = find_gaps(VULNERABLE_GUARD, TOOL_REGISTRY)
    uncatalogued = dropped_findings(findings)
    write_review_queue(candidates, gaps, args.queue_out, uncatalogued=uncatalogued)

    print(f"scanned {args.docs}/ — {len(findings)} finding(s) → {args.findings_out}")
    print(
        f"proposed {len(candidates)} rule candidate(s), all approved=false"
        f" (+{len(uncatalogued)} routed to triage) → {args.queue_out}"
    )
    print("nothing applied: a human approves in the queue before any rule takes effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
