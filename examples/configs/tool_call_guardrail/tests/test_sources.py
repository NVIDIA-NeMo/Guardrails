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

"""Offline tests for the acquisition layer.

The HTTP transport is injected as a fake returning fixture XML, so fetchers are
exercised with no network. Covers arXiv/RSS/Atom parsing, that fetched documents
round-trip through the same `scan.py` helpers that read hand-written docs, the
scheme guard on the real transport, and the watermark that makes re-runs only
write genuinely new documents.
"""

import pytest
from scanner.acquire import acquire
from scanner.scan import SourceDoc, _source_url, _title
from scanner.sources import ArxivFetcher, FeedFetcher, urllib_http

ARXIV_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2406.10000v1</id>
    <title>Confused Deputy Attacks on LLM Agents</title>
    <summary>Tool-using agents can be turned into confused deputies.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2406.10001v2</id>
    <title>Bounding Tool Arguments</title>
    <summary>A study of per-call limits on agent tool arguments.</summary>
  </entry>
</feed>
"""

RSS_FIXTURE = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Sec Advisories</title>
  <item>
    <title>Malicious dependency installs</title>
    <link>https://example.org/a1</link>
    <guid>ex-a1</guid>
    <description>Agents steered into installing typosquatted packages.</description>
  </item>
  <item>
    <title>Runaway shell timeouts</title>
    <link>https://example.org/a2</link>
    <guid>ex-a2</guid>
    <description>Inflated run_shell timeouts drain compute.</description>
  </item>
</channel></rss>
"""

ATOM_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>tag:example.org,2026:1</id>
    <title>Path traversal in agent file reads</title>
    <link href="https://example.org/atom1"/>
    <summary>Injection via the path argument of a file read.</summary>
  </entry>
</feed>
"""


def _http_returning(payload: bytes):
    captured = {}

    def http(url: str) -> bytes:
        captured["url"] = url
        return payload

    http.captured = captured
    return http


class _CannedFetcher:
    """Fetcher test double that yields a fixed list of SourceDocs."""

    def __init__(self, *docs: SourceDoc):
        self._docs = docs

    def fetch(self):
        return iter(self._docs)


def test_arxiv_fetcher_parses_entries():
    docs = list(ArxivFetcher("cat:cs.CR", _http_returning(ARXIV_FIXTURE)).fetch())
    assert [d.id for d in docs] == ["arxiv-2406.10000v1", "arxiv-2406.10001v2"]
    assert docs[0].url == "http://arxiv.org/abs/2406.10000v1"
    assert "Confused Deputy" in docs[0].text


def test_arxiv_query_is_sent_url_encoded():
    http = _http_returning(ARXIV_FIXTURE)
    list(ArxivFetcher('abs:"LLM agent"', http).fetch())
    assert "search_query=abs" in http.captured["url"]
    assert "max_results=25" in http.captured["url"]


def test_feed_fetcher_parses_rss():
    docs = list(FeedFetcher("https://x/rss", _http_returning(RSS_FIXTURE)).fetch())
    assert len(docs) == 2
    assert docs[0].url == "https://example.org/a1"
    assert "Malicious dependency" in docs[0].text
    assert all(d.id.startswith("feed-") for d in docs)


def test_feed_fetcher_parses_atom():
    [doc] = list(FeedFetcher("https://x/atom", _http_returning(ATOM_FIXTURE)).fetch())
    assert doc.url == "https://example.org/atom1"
    assert "Path traversal" in doc.text


def test_fetched_doc_round_trips_through_scan_helpers():
    # A fetched doc must be indistinguishable from a hand-written sample doc to
    # the scanner: scan.py recovers the same title and provenance from its text.
    doc = next(FeedFetcher("https://x/atom", _http_returning(ATOM_FIXTURE)).fetch())
    assert _title(doc) == "Path traversal in agent file reads"
    assert _source_url(doc.text, "fallback") == doc.url


def test_urllib_http_rejects_non_http_schemes():
    with pytest.raises(ValueError):
        urllib_http()("file:///etc/passwd")


def test_acquire_writes_then_watermark_suppresses_rerun(tmp_path):
    docs = [
        SourceDoc(id="arxiv-1", url="u1", text="# A\n\nSource: u1\n\nbody\n"),
        SourceDoc(id="arxiv-2", url="u2", text="# B\n\nSource: u2\n\nbody\n"),
    ]
    out_dir = str(tmp_path / "corpus")
    ledger = str(tmp_path / "seen.json")

    first = acquire([_CannedFetcher(*docs)], out_dir, ledger)
    assert len(first.written) == 2
    assert first.skipped == 0
    assert first.total_seen == 2

    # Same documents on a second run: nothing new is written.
    second = acquire([_CannedFetcher(*docs)], out_dir, ledger)
    assert second.written == ()
    assert second.skipped == 2


def test_acquire_dry_run_writes_nothing(tmp_path):
    docs = [SourceDoc(id="x-1", url="u", text="# T\n\nSource: u\n\nbody\n")]
    out_dir = str(tmp_path / "corpus")
    ledger = str(tmp_path / "seen.json")
    result = acquire([_CannedFetcher(*docs)], out_dir, ledger, dry_run=True)
    assert len(result.written) == 1  # reported...
    assert not tmp_path.joinpath("corpus").exists()  # ...but nothing written
    assert not tmp_path.joinpath("seen.json").exists()


def test_acquire_skips_a_failing_source(tmp_path):
    class _Boom:
        def fetch(self):
            raise RuntimeError("source down")

    docs = [SourceDoc(id="ok-1", url="u", text="# T\n\nSource: u\n\nbody\n")]
    result = acquire(
        [_Boom(), _CannedFetcher(*docs)],
        str(tmp_path / "corpus"),
        str(tmp_path / "seen.json"),
    )
    assert len(result.written) == 1  # the healthy source still produced its doc
