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

"""Acquisition layer — where the scan corpus comes from.

`scan.py`'s `fetch_new()` reads documents from a local folder; this module is how
that folder gets populated from the field. A `Fetcher` pulls entries from a remote
source (the arXiv API, an RSS/Atom feed) and normalizes each into the very same
`SourceDoc` markdown the scanner already consumes — so `scan.py` runs unchanged on
top of whatever is acquired.

Two seams keep it testable and honest:
  - an injected `Http` transport (url -> bytes), so unit tests parse fixtures with
    no network, mirroring how the LLM extractor injects its `Chat` backend; and
  - fetched content is untrusted *data* — it is only ever parsed and written out
    as text, never executed, and provenance (the source URL) is always carried
    through so a reviewer can trace any finding back to where it came from.

Stdlib only (urllib + ElementTree), so the acquisition path adds no dependencies.
For production, parse feeds with `defusedxml` instead of `xml.etree` to harden
against hostile XML (entity-expansion DoS); the `Http`/parse split here leaves
that a drop-in change.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, Iterator, Optional, Protocol

try:  # works whether imported as a package or run from the scanner/ dir
    from scanner.scan import SourceDoc
except ImportError:  # pragma: no cover - import shim
    from scan import SourceDoc

# url -> raw response bytes. Inject a fake in tests; `urllib_http()` is the default.
Http = Callable[[str], bytes]

_ATOM = "{http://www.w3.org/2005/Atom}"


def urllib_http(timeout: float = 30.0, user_agent: str = "tool-call-guardrail-scanner/0.1") -> Http:
    """Default transport: stdlib urllib, http(s) only, with a timeout.

    The scheme is checked before the request so a feed URL can never be coerced
    into reading `file://` or other local schemes.
    """

    def http(url: str) -> bytes:
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ValueError(f"refusing non-http(s) URL: {url!r}")
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    return http


class Fetcher(Protocol):
    """Pulls documents from one remote source. It only reads (through the injected
    transport) and yields `SourceDoc`s — it never touches the filesystem, so it is
    pure enough to unit-test against fixtures."""

    def fetch(self) -> Iterator[SourceDoc]: ...


def render_markdown(title: str, url: str, body: str) -> str:
    """The on-disk shape `scan.py` expects: an `# H1` title, a `Source:` line for
    provenance, then the body. `SourceDoc.text` carries this verbatim so a fetched
    doc and a hand-written sample doc are indistinguishable to the scanner."""
    return f"# {title}\n\nSource: {url}\n\n{body}\n"


def _clean(text: Optional[str]) -> str:
    """Collapse whitespace; tolerate missing fields."""
    return " ".join((text or "").split())


def _slug(value: str, max_len: int = 80) -> str:
    """A filesystem- and id-safe slug. Long values are truncated with a stable
    hash suffix so distinct sources never collide on the same id."""
    base = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    if len(base) <= max_len:
        return base or "x"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{base[:max_len]}-{digest}"


class ArxivFetcher:
    """Query the arXiv API and yield one `SourceDoc` per result (abstract as body).

    `query` is an arXiv `search_query` expression, e.g.
    `cat:cs.CR AND (abs:"LLM agent" OR abs:"tool use")`. Results are newest-first.
    """

    API = "https://export.arxiv.org/api/query"

    def __init__(self, query: str, http: Http, max_results: int = 25):
        self._query = query
        self._http = http
        self._max_results = max_results

    def fetch(self) -> Iterator[SourceDoc]:
        params = urllib.parse.urlencode(
            {
                "search_query": self._query,
                "start": 0,
                "max_results": self._max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        root = ET.fromstring(self._http(f"{self.API}?{params}"))
        for entry in root.findall(f"{_ATOM}entry"):
            doc = self._entry_to_doc(entry)
            if doc is not None:
                yield doc

    @staticmethod
    def _entry_to_doc(entry: ET.Element) -> Optional[SourceDoc]:
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        if not raw_id:
            return None
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        title = _clean(entry.findtext(f"{_ATOM}title")) or arxiv_id
        summary = _clean(entry.findtext(f"{_ATOM}summary"))
        return SourceDoc(
            id=f"arxiv-{arxiv_id}",
            url=raw_id,
            text=render_markdown(title, raw_id, summary),
        )


class FeedFetcher:
    """Fetch an RSS 2.0 or Atom feed and yield a `SourceDoc` per item/entry."""

    def __init__(self, url: str, http: Http, max_results: int = 50):
        self._url = url
        self._http = http
        self._max_results = max_results

    def fetch(self) -> Iterator[SourceDoc]:
        root = ET.fromstring(self._http(self._url))
        tag = root.tag.lower()
        if tag.endswith("rss"):
            items = root.findall(".//item")
            parse = self._rss_item
        elif tag.endswith("feed"):
            items = root.findall(f"{_ATOM}entry")
            parse = self._atom_entry
        else:  # unrecognized document; nothing to yield
            return
        for item in items[: self._max_results]:
            doc = parse(item)
            if doc is not None:
                yield doc

    @staticmethod
    def _rss_item(item: ET.Element) -> Optional[SourceDoc]:
        link = _clean(item.findtext("link"))
        ident = _clean(item.findtext("guid")) or link
        if not ident:
            return None
        title = _clean(item.findtext("title")) or ident
        body = _clean(item.findtext("description"))
        return SourceDoc(
            id=f"feed-{_slug(ident)}",
            url=link or ident,
            text=render_markdown(title, link or ident, body),
        )

    @staticmethod
    def _atom_entry(entry: ET.Element) -> Optional[SourceDoc]:
        ident = _clean(entry.findtext(f"{_ATOM}id"))
        link_el = entry.find(f"{_ATOM}link")
        link = link_el.get("href", "") if link_el is not None else ""
        ident = ident or link
        if not ident:
            return None
        title = _clean(entry.findtext(f"{_ATOM}title")) or ident
        body = _clean(entry.findtext(f"{_ATOM}summary")) or _clean(entry.findtext(f"{_ATOM}content"))
        return SourceDoc(
            id=f"feed-{_slug(ident)}",
            url=link or ident,
            text=render_markdown(title, link or ident, body),
        )
