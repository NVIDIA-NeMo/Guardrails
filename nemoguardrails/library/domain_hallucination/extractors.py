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

"""Unified extraction module for URLs, domains, and GitHub repos."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

URL_RE = re.compile(
    r"""
    (?:
        https?://[^\s<>()\[\]{}"']+
        |
        www\.[^\s<>()\[\]{}"']+
        |
        github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+[^\s<>()\[\]{}"']*
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

DOMAIN_RE = re.compile(
    r"(?<![@\w-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}(?![\w-])",
    re.IGNORECASE,
)

IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

TRAILING_CHARS = " \t\r\n.。．，,;；:：!！?？)）]】}>\"'`"
DOMAIN_TRAILING_CHARS = " \t\r\n。．，,;；:：!！?？)）]】}>\"'`"

GITHUB_RESERVED_PATHS = {
    "features",
    "topics",
    "search",
    "marketplace",
    "pricing",
    "login",
    "signup",
    "explore",
    "collections",
    "events",
    "sponsors",
    "about",
    "enterprise",
    "customer-stories",
    "readme",
    "trending",
    "new",
    "organizations",
    "orgs",
    "users",
    "apps",
    "settings",
    "notifications",
    "codespaces",
    "site",
    "contact",
}

_TLD_EXTRACTOR = None
try:
    import tldextract

    _TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None, cache_dir=None)
except Exception:
    pass


def clean_url(raw_url: str) -> str:
    """Clean URL from trailing punctuation and Markdown wrappers."""
    if not raw_url:
        return ""
    url = raw_url.strip()
    url = url.strip("`'\"")
    url = url.rstrip(TRAILING_CHARS)
    return url


def normalize_url(raw_url: str, default_scheme: str = "https") -> str:
    """Normalize URL by adding scheme and cleaning."""
    url = clean_url(raw_url)
    if not url:
        return ""
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = f"{default_scheme}://{url}"
    return url


def parse_url(url: str) -> Dict[str, Any]:
    """Parse URL into components."""
    parsed = urlparse(url)
    host = parsed.hostname.lower() if parsed.hostname else ""
    if host.startswith("www."):
        host = host[4:]
    return {
        "scheme": parsed.scheme.lower() if parsed.scheme else "",
        "host": host,
        "path": parsed.path or "",
        "query": parsed.query or "",
        "fragment": parsed.fragment or "",
    }


def extract_urls(text: str, source: str = "model_answer") -> List[Dict[str, Any]]:
    """Extract and normalize URLs from text."""
    if not text:
        return []

    results: List[Dict[str, Any]] = []
    seen = set()

    for match in URL_RE.finditer(text):
        raw = match.group(0)
        normalized = normalize_url(raw)

        if not normalized or normalized in seen:
            continue

        parsed = parse_url(normalized)
        if not parsed["host"]:
            continue

        seen.add(normalized)
        results.append(
            {
                "raw": clean_url(raw),
                "normalized": normalized,
                "scheme": parsed["scheme"],
                "host": parsed["host"],
                "path": parsed["path"],
                "query": parsed["query"],
                "fragment": parsed["fragment"],
                "source": source,
            }
        )

    return results


def normalize_domain(raw: str) -> Optional[str]:
    """Normalize domain name."""
    if not raw:
        return None

    domain = raw.strip().lower()
    domain = domain.strip(DOMAIN_TRAILING_CHARS)

    if domain.startswith("http://") or domain.startswith("https://"):
        parsed = urlparse(domain)
        domain = parsed.hostname or ""

    if domain.startswith("www."):
        domain = domain[4:]

    if "/" in domain:
        domain = domain.split("/", 1)[0]

    if ":" in domain:
        domain = domain.split(":", 1)[0]

    domain = domain.strip(".").strip(DOMAIN_TRAILING_CHARS)

    if not domain or IPV4_RE.match(domain):
        return None

    try:
        ascii_domain = domain.encode("idna").decode("ascii")
    except UnicodeError:
        return None

    if len(ascii_domain) > 253 or "." not in ascii_domain:
        return None

    labels = ascii_domain.split(".")
    if any(not label or len(label) > 63 for label in labels):
        return None

    if any(label.startswith("-") or label.endswith("-") for label in labels):
        return None

    if not re.fullmatch(r"[a-z0-9.-]+", ascii_domain):
        return None

    return ascii_domain


def split_domain(domain: str) -> Dict[str, str]:
    """Split domain into parts."""
    if _TLD_EXTRACTOR is not None:
        ext = _TLD_EXTRACTOR(domain)
        registered_domain = ".".join(part for part in [ext.domain, ext.suffix] if part)
        return {
            "host": domain,
            "registered_domain": registered_domain,
            "subdomain": ext.subdomain or "",
            "domain": ext.domain or "",
            "suffix": ext.suffix or "",
        }

    parts = domain.split(".")
    if len(parts) >= 2:
        registered_domain = ".".join(parts[-2:])
        subdomain = ".".join(parts[:-2])
        main_domain = parts[-2]
        suffix = parts[-1]
    else:
        registered_domain = domain
        subdomain = ""
        main_domain = domain
        suffix = ""

    return {
        "host": domain,
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "domain": main_domain,
        "suffix": suffix,
    }


def extract_domains(
    text: str, urls: List[Dict[str, Any]] | None = None, source: str = "model_answer"
) -> List[Dict[str, Any]]:
    """Extract domains from URLs and bare text."""
    urls = urls or []
    results: List[Dict[str, Any]] = []
    seen = set()

    # From URLs
    for item in urls:
        raw_host = str(item.get("host") or "")
        normalized = normalize_domain(raw_host)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        domain_info = split_domain(normalized)
        results.append(
            {
                **domain_info,
                "raw_host": raw_host,
                "source_url": item.get("normalized", ""),
                "source": item.get("source", source),
                "from_url": True,
            }
        )

    # Bare domains
    for match in DOMAIN_RE.finditer(text):
        raw = match.group(0)
        normalized = normalize_domain(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        domain_info = split_domain(normalized)
        results.append(
            {
                **domain_info,
                "raw": raw,
                "raw_host": raw,
                "source_url": "",
                "source": source,
                "from_url": False,
            }
        )

    return results


def is_github_host(host: str) -> bool:
    """Check if host is GitHub."""
    return host.lower() in {"github.com", "www.github.com"}


def parse_github_url(url: str, source: str = "model_answer") -> Optional[Dict[str, Any]]:
    """Parse GitHub repository URL."""
    if not url:
        return None

    parsed = urlparse(url)
    if not is_github_host(parsed.hostname or ""):
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None

    owner = parts[0]
    repo = parts[1].removesuffix(".git")

    if owner.lower() in GITHUB_RESERVED_PATHS:
        return None

    link_type = "repo"
    branch = None
    inner_path = None

    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        link_type = parts[2]
        branch = parts[3]
        inner_path = "/".join(parts[4:]) if len(parts) > 4 else None
    elif len(parts) >= 3 and parts[2] in {"issues", "pull", "releases", "wiki"}:
        link_type = parts[2]
        inner_path = "/".join(parts[3:]) if len(parts) > 3 else None

    return {
        "platform": "github",
        "url": url,
        "owner": owner,
        "repo": repo,
        "full_name": f"{owner}/{repo}",
        "branch": branch,
        "path": inner_path,
        "link_type": link_type,
        "source": source,
    }


def extract_github_repos(urls: List[Dict[str, Any]], source: str = "model_answer") -> List[Dict[str, Any]]:
    """Extract GitHub repositories from URLs."""
    results: List[Dict[str, Any]] = []
    seen = set()

    for item in urls:
        normalized_url = item.get("normalized") or ""
        parsed = parse_github_url(normalized_url, source=item.get("source", source))

        if not parsed:
            continue

        key = (
            parsed["owner"].lower(),
            parsed["repo"].lower(),
            parsed["link_type"],
            parsed.get("branch") or "",
            parsed.get("path") or "",
        )
        if key in seen:
            continue

        seen.add(key)
        results.append(parsed)

    return results


def extract_all(text: str, source: str = "model_answer") -> Dict[str, Any]:
    """Extract all entities from text."""
    urls = extract_urls(text, source=source)
    domains = extract_domains(text, urls=urls, source=source)
    github_repos = extract_github_repos(urls, source=source)

    # Fast pass: if no URLs/domains/GitHub, return early
    if not urls and not domains and not github_repos:
        return {
            "urls": [],
            "domains": [],
            "github_repos": [],
            "no_links": True,
        }

    return {
        "urls": urls,
        "domains": domains,
        "github_repos": github_repos,
        "no_links": False,
    }
