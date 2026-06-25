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

"""Populate a scan corpus from real sources, then let scan.py run on top.

Each fetched document is written once as a normalized `*.md` file into the output
folder and recorded in a JSON ledger, so re-running only writes documents that are
genuinely new (the document-level watermark `fetch_new()`'s comment anticipated).
The flow is two decoupled steps:

    python3 scanner/acquire.py \\
        --arxiv 'cat:cs.CR AND abs:"LLM agent"' --arxiv-full-text \\
        --feed https://example.org/security/feed.xml \\
        --out-dir corpus/
    python3 scanner/scan.py --extractor llm --docs corpus/ --out findings.json

A bad source logs a warning and is skipped rather than aborting the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # works whether imported as a package or run from the scanner/ dir
    from scanner.sources import ArxivFetcher, FeedFetcher, Fetcher, urllib_http
except ImportError:  # pragma: no cover - import shim
    from sources import ArxivFetcher, FeedFetcher, Fetcher, urllib_http


@dataclass(frozen=True)
class AcquireResult:
    """What one acquisition run did."""

    written: tuple  # filenames written this run
    skipped: int  # documents already in the ledger
    total_seen: int  # ledger size after the run


def _load_ledger(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        return set(json.load(fh).get("seen", []))


def _save_ledger(path: str, seen: set) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"seen": sorted(seen)}, fh, indent=2)
        fh.write("\n")


def _safe_filename(doc_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", doc_id).strip("-")
    return (safe or "doc")[:120]


def acquire(
    fetchers: Sequence[Fetcher],
    out_dir: str,
    ledger_path: str,
    dry_run: bool = False,
) -> AcquireResult:
    """Fetch from each source, write only documents not already in the ledger.

    Dedup is by `SourceDoc.id`, both against prior runs (the ledger) and within a
    single run (so two feeds surfacing the same item write it once).
    """
    seen = _load_ledger(ledger_path)
    written: list = []
    skipped = 0
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)

    for fetcher in fetchers:
        try:
            docs = list(fetcher.fetch())
        except Exception as exc:  # one bad source must not abort the run
            print(f"warning: {type(fetcher).__name__} failed: {exc}", file=sys.stderr)
            continue
        for doc in docs:
            if doc.id in seen:
                skipped += 1
                continue
            seen.add(doc.id)
            filename = _safe_filename(doc.id) + ".md"
            written.append(filename)
            if not dry_run:
                body = doc.text if doc.text.endswith("\n") else doc.text + "\n"
                with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as fh:
                    fh.write(body)

    if not dry_run:
        _save_ledger(ledger_path, seen)
    return AcquireResult(tuple(written), skipped, len(seen))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch source documents into a scan corpus (watermarked).")
    parser.add_argument(
        "--arxiv",
        action="append",
        default=[],
        metavar="QUERY",
        help="arXiv search_query expression (repeatable)",
    )
    parser.add_argument(
        "--feed",
        action="append",
        default=[],
        metavar="URL",
        help="RSS or Atom feed URL (repeatable)",
    )
    parser.add_argument("--out-dir", required=True, help="folder to write *.md into")
    parser.add_argument(
        "--ledger",
        help="watermark JSON of seen document ids (default: <out-dir>/.acquired.json)",
    )
    parser.add_argument("--max-results", type=int, default=25, help="max items per source")
    parser.add_argument(
        "--arxiv-full-text",
        action="store_true",
        help="for --arxiv sources, fetch each paper's rendered HTML full text "
        "(arxiv.org/html/<id>) instead of just the abstract; falls back to the "
        "abstract when no HTML rendering exists",
    )
    parser.add_argument(
        "--full-text-max-chars",
        type=int,
        default=40_000,
        help="cap on full-text characters kept per document (default 40000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be fetched without writing files or the ledger",
    )
    args = parser.parse_args()

    if not args.arxiv and not args.feed:
        parser.error("provide at least one --arxiv or --feed source")

    ledger_path = args.ledger or os.path.join(args.out_dir, ".acquired.json")
    http = urllib_http()
    fetchers: list[Fetcher] = [
        ArxivFetcher(
            q,
            http,
            max_results=args.max_results,
            full_text=args.arxiv_full_text,
            full_text_max_chars=args.full_text_max_chars,
        )
        for q in args.arxiv
    ] + [FeedFetcher(url, http, max_results=args.max_results) for url in args.feed]

    result = acquire(fetchers, args.out_dir, ledger_path, dry_run=args.dry_run)

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"{verb} {len(result.written)} new document(s) to {args.out_dir} "
        f"(skipped {result.skipped} already seen; ledger now {result.total_seen})"
    )
    for name in result.written:
        print(f"  + {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
