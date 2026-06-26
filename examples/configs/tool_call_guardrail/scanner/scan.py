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

"""Produce findings from source documents.

`scan()` is the trusted orchestration: it lists sources, hands each to a
swappable `Extractor`, and then enforces two guarantees that must NOT live in the
(replaceable) extractor — every emitted finding is **grounded** to a tool that
actually exists in the registry, and carries **provenance** (a source). The
offline `KeywordExtractor` stands in for an LLM so the pipeline runs
deterministically; a production `LLMExtractor` would implement the same protocol.

    python3 scanner/scan.py            # print findings extracted from sample_docs
    python3 scanner/scan.py --out f.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Protocol, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis.catalog import (  # noqa: E402
    CLASS_DESCRIPTIONS,
    CLASS_REQUIRED_PARAMS,
    CLASS_TO_FACTORY,
)
from synthesis.findings import Finding  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS = os.path.join(HERE, "sample_docs")


@dataclass(frozen=True)
class SourceDoc:
    """One raw document pulled from a source (a file, in this first slice)."""

    id: str  # stable id (filename stem)
    url: str  # provenance
    text: str


@dataclass(frozen=True)
class ArgSpec:
    """One argument of a tool. Fed to the LLM extractor so it grounds an
    `arg_name` it proposes against the tool's real argument names."""

    name: str
    type: str = "string"
    description: str = ""


@dataclass(frozen=True)
class ScanContext:
    docs_dir: str
    tool_registry: Mapping[str, str]  # tool name -> description
    taxonomy: Sequence[str]  # allowed attack_class values
    known_finding_ids: frozenset = frozenset()  # dedup ledger from prior runs
    class_definitions: Mapping[str, str] = field(
        default_factory=dict
    )  # attack_class -> the control it represents (used by the LLM extractor)
    class_params: Mapping[str, Sequence[str]] = field(
        default_factory=dict
    )  # attack_class -> suggested_params keys it requires (used by the LLM extractor)
    tool_schemas: Mapping[str, Sequence[ArgSpec]] = field(
        default_factory=dict
    )  # tool name -> its argument specs (used by the LLM extractor to ground arg names)
    principal_attrs: Sequence[str] = ()  # recognized principal attributes (for attr_name)


@dataclass(frozen=True)
class ExtractedTechnique:
    """The extractor's read of one document. Tool mentions are *candidates*;
    scan() is what filters them against the real registry."""

    summary: str
    attack_class: str
    tool_mentions: Sequence[str]
    suggested_params: Mapping[str, object] = field(default_factory=dict)
    excerpt: str = ""


class Extractor(Protocol):
    """The one judgement-heavy seam. In production this is an LLM call.

    Returns every technique found in the document (empty if none) — a full paper
    may describe several, so the contract is a sequence, not a single result.
    """

    def extract(self, doc: SourceDoc, ctx: ScanContext) -> Sequence[ExtractedTechnique]: ...


# --- Offline stand-in extractor --------------------------------------------
# Deterministic keyword/registry matching. It proves the plumbing, not
# classification quality; swap in an LLM-backed Extractor to handle real prose.

_CLASS_KEYWORDS: Mapping[str, Sequence[str]] = {
    # More specific classes are checked first so a doc that names a prefix- or
    # pattern-based control is not stolen by the broader ownership/denylist classes.
    "prefix-ownership-bypass": (
        "workspace prefix",
        "path prefix",
        "directory namespace",
    ),
    "disallowed-pattern": (
        "regular expression",
        "url pattern",
        "metadata service",
        "ssrf",
    ),
    "ownership-bypass": (
        "confused deputy",
        "ownership check",
        "does not own",
        "ownership",
    ),
    "unbounded-arg": (
        "per-call limit",
        "ceiling",
        "blast radius",
        "just under",
    ),
    "disallowed-target": (
        "denylist",
        "blocklist",
        "sanctioned",
        "forbidden target",
    ),
    "argument-injection": (
        "path traversal",
        "injection",
        "metacharacter",
        "malformed",
    ),
    "privilege-escalation": (
        "privilege escalation",
        "step-up",
        "elevated privilege",
        "without authorization",
    ),
}

_PARAMS_HINT = re.compile(r"<!--\s*params:\s*(\{.*?\})\s*-->", re.DOTALL)


class KeywordExtractor:
    """Classifies by keyword and grounds by literal tool-name mention.

    Returns at most one technique per document — the short fixtures it targets
    describe one each. Chunking and multi-technique extraction are the LLM path's
    job (see llm_extractor.py)."""

    def extract(self, doc: SourceDoc, ctx: ScanContext) -> Sequence[ExtractedTechnique]:
        low = doc.text.lower()

        attack_class = "novel"
        for cls, keywords in _CLASS_KEYWORDS.items():
            if cls in ctx.taxonomy and any(k in low for k in keywords):
                attack_class = cls
                break

        mentions = [name for name in ctx.tool_registry if re.search(rf"\b{re.escape(name)}\b", doc.text)]

        # Nothing actionable: no known class and no tool in scope.
        if attack_class == "novel" and not mentions:
            return []

        match = _PARAMS_HINT.search(doc.text)
        params = json.loads(match.group(1)) if match else {}

        return [
            ExtractedTechnique(
                summary=_title(doc),
                attack_class=attack_class,
                tool_mentions=mentions,
                suggested_params=params,
                excerpt=_first_paragraph(doc.text),
            )
        ]


def _title(doc: SourceDoc) -> str:
    for line in doc.text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return doc.id


def _first_paragraph(text: str, limit: int = 240) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip() and not p.lstrip().startswith(("#", "Source:", "<!--"))]
    body = paras[0] if paras else ""
    body = " ".join(body.split())
    return body[:limit] + ("…" if len(body) > limit else "")


def fetch_new(ctx: ScanContext) -> Iterator[SourceDoc]:
    """List source documents. A real scanner would page a feed with a watermark;
    here it just reads *.md from the docs folder in a stable order."""
    for name in sorted(os.listdir(ctx.docs_dir)):
        if not name.endswith(".md"):
            continue
        path = os.path.join(ctx.docs_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        yield SourceDoc(id=os.path.splitext(name)[0], url=_source_url(text, path), text=text)


def _source_url(text: str, fallback_path: str) -> str:
    for line in text.splitlines():
        if line.startswith("Source:"):
            return line[len("Source:") :].strip()
    return f"file://{fallback_path}"


def scan(ctx: ScanContext, extractor: Extractor) -> list[Finding]:
    """Turn source documents into grounded, deduplicated findings."""
    findings: list[Finding] = []
    seen: set[str] = set()
    for doc in fetch_new(ctx):
        for technique in extractor.extract(doc, ctx):
            # Trusted guarantee: only tools that really exist in the registry.
            tools = [t for t in technique.tool_mentions if t in ctx.tool_registry]
            if not tools:
                continue  # ungrounded technique is not actionable

            finding_id = f"{doc.id}-{technique.attack_class}"
            if finding_id in ctx.known_finding_ids or finding_id in seen:
                continue
            seen.add(finding_id)

            findings.append(
                Finding(
                    id=finding_id,
                    title=technique.summary,
                    source=doc.url,  # trusted guarantee: provenance always present
                    attack_class=technique.attack_class,
                    affected_tools=tuple(tools),
                    suggested_params=technique.suggested_params,
                    evidence=technique.excerpt,
                )
            )
    return findings


def _finding_to_dict(f: Finding) -> dict:
    data = dataclasses.asdict(f)
    data["affected_tools"] = list(f.affected_tools)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan source docs into findings.")
    parser.add_argument("--docs", default=SAMPLE_DOCS, help="docs folder to scan")
    parser.add_argument("--out", help="write findings JSON here (default: stdout)")
    parser.add_argument(
        "--extractor",
        choices=("keyword", "llm"),
        default="keyword",
        help="keyword (offline, default) or llm (needs openai; see --base-url / --api-key-env)",
    )
    parser.add_argument(
        "--model",
        default="azure/openai/gpt-4o-mini",
        help="model id for the llm extractor",
    )
    parser.add_argument(
        "--base-url",
        default="https://inference-api.nvidia.com/v1",
        help="OpenAI-compatible base URL for the llm extractor; point at a local "
        "server (e.g. http://localhost:11434/v1) to run without the gateway",
    )
    parser.add_argument(
        "--api-key-env",
        default="NVIDIA_LITELLM_KEY",
        help="env var holding the API key for --base-url (local servers often accept any value)",
    )
    args = parser.parse_args()

    from example_policies import PRINCIPAL_ATTRS, TOOL_REGISTRY, TOOL_SCHEMAS

    ctx = ScanContext(
        docs_dir=args.docs,
        tool_registry=dict(TOOL_REGISTRY),
        taxonomy=tuple(CLASS_TO_FACTORY),
        class_definitions=dict(CLASS_DESCRIPTIONS),
        class_params=dict(CLASS_REQUIRED_PARAMS),
        tool_schemas=dict(TOOL_SCHEMAS),
        principal_attrs=tuple(PRINCIPAL_ATTRS),
    )

    catch: tuple = ()  # keyword path raises nothing extractor-specific
    if args.extractor == "llm":
        try:
            from scanner.llm_extractor import ExtractionError, LLMExtractor, nvidia_chat
        except ImportError:
            from llm_extractor import ExtractionError, LLMExtractor, nvidia_chat
        extractor: Extractor = LLMExtractor(
            nvidia_chat(
                model=args.model,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
            )
        )
        catch = (ExtractionError,)
    else:
        extractor = KeywordExtractor()

    try:
        findings = scan(ctx, extractor)
    except catch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps([_finding_to_dict(f) for f in findings], indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        print(f"wrote {len(findings)} findings to {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
