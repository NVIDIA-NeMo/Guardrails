"""Verification module for DNS, HTTP, and GitHub checks."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_domain(domain: str, timeout: float = 4.0) -> Dict[str, Any]:
    """Resolve DNS records for a domain."""
    domain = (domain or "").strip().lower()
    started = time.perf_counter()

    if not domain or "." not in domain:
        return {
            "source": "dns",
            "domain": domain,
            "status": "invalid_domain",
            "resolves": False,
            "addresses": [],
            "public_address_count": 0,
            "error": "invalid domain",
            "confidence": "high",
            "use_in_scoring": True,
            "checked_at_utc": utc_now(),
            "latency_ms": 0,
        }

    try:
        infos = socket.getaddrinfo(domain, 443, proto=socket.IPPROTO_TCP)
        addresses = sorted({info[4][0] for info in infos})
        public_addresses = []

        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
                if not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved):
                    public_addresses.append(address)
            except ValueError:
                public_addresses.append(address)

        status = "resolved" if addresses else "no_data"

        return {
            "source": "dns",
            "domain": domain,
            "status": status,
            "resolves": bool(addresses),
            "addresses": addresses[:10],
            "public_address_count": len(public_addresses),
            "confidence": "high",
            "use_in_scoring": True,
            "checked_at_utc": utc_now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    except socket.gaierror as exc:
        return {
            "source": "dns",
            "domain": domain,
            "status": "nxdomain_or_no_data",
            "resolves": False,
            "addresses": [],
            "public_address_count": 0,
            "error": str(exc),
            "confidence": "high",
            "use_in_scoring": True,
            "checked_at_utc": utc_now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    except (TimeoutError, socket.timeout) as exc:
        return {
            "source": "dns",
            "domain": domain,
            "status": "dns_timeout",
            "resolves": False,
            "addresses": [],
            "public_address_count": 0,
            "error": str(exc),
            "confidence": "high",
            "use_in_scoring": True,
            "checked_at_utc": utc_now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    except Exception as exc:
        return {
            "source": "dns",
            "domain": domain,
            "status": "dns_error",
            "resolves": False,
            "addresses": [],
            "public_address_count": 0,
            "error": str(exc),
            "confidence": "high",
            "use_in_scoring": True,
            "checked_at_utc": utc_now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def check_http_domain(target: str, timeout: float = 6.0) -> Dict[str, Any]:
    """Check URL/domain accessibility via HTTP."""
    target = (target or "").strip()
    started = time.perf_counter()

    if not target:
        return {
            "source": "http",
            "target": target,
            "domain": "",
            "status": "invalid_url",
            "reachable": False,
            "error": "empty target",
            "confidence": "high",
            "use_in_scoring": True,
            "checked_at_utc": utc_now(),
            "latency_ms": 0,
        }

    urls = []
    if target.startswith("http://") or target.startswith("https://"):
        urls = [target]
    else:
        urls = [f"https://{target}/", f"http://{target}/"]

    last: Dict[str, Any] = {}
    for url in urls:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        started = time.perf_counter()

        request = Request(url, method="HEAD", headers={"User-Agent": "DomainGuard/0.1"})

        try:
            with urlopen(request, timeout=timeout) as response:
                status_code = int(getattr(response, "status", 200))
                return {
                    "source": "http",
                    "target": target,
                    "domain": domain,
                    "status": "http_ok",
                    "reachable": True,
                    "url": url,
                    "status_code": status_code,
                    "final_url": response.geturl(),
                    "confidence": "high",
                    "use_in_scoring": True,
                    "checked_at_utc": utc_now(),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
        except HTTPError as exc:
            if exc.code in {403, 405} and url.startswith("https://"):
                continue
            return {
                "source": "http",
                "target": target,
                "domain": domain,
                "status": "http_error_status",
                "reachable": False,
                "url": url,
                "status_code": exc.code,
                "final_url": url,
                "error": str(exc),
                "confidence": "high",
                "use_in_scoring": True,
                "checked_at_utc": utc_now(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except (TimeoutError, socket.timeout) as exc:
            last = {
                "source": "http",
                "target": target,
                "domain": domain,
                "status": "http_timeout",
                "reachable": False,
                "url": url,
                "status_code": None,
                "final_url": url,
                "error": str(exc),
                "confidence": "high",
                "use_in_scoring": True,
                "checked_at_utc": utc_now(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            last = {
                "source": "http",
                "target": target,
                "domain": domain,
                "status": "http_error",
                "reachable": False,
                "url": url,
                "status_code": None,
                "final_url": url,
                "error": str(exc),
                "confidence": "high",
                "use_in_scoring": True,
                "checked_at_utc": utc_now(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }

    return last


OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def check_github_repo(repo_item: Dict[str, Any], timeout: float = 6.0, token: str | None = None) -> Dict[str, Any]:
    """Check if a GitHub repository exists."""
    owner = str((repo_item or {}).get("owner") or "").strip()
    repo = str((repo_item or {}).get("repo") or "").strip().removesuffix(".git")
    full_name = f"{owner}/{repo}" if owner and repo else ""

    if not owner or not repo:
        return {
            "source": "github",
            "owner": owner,
            "repo": repo,
            "full_name": full_name,
            "url": str((repo_item or {}).get("url") or ""),
            "status": "invalid_repo_item",
            "format_valid": False,
            "exists": False,
            "error": "missing owner or repo",
            "confidence": "high",
            "use_in_scoring": True,
            "checked_at_utc": utc_now(),
            "latency_ms": 0,
        }

    if not OWNER_RE.fullmatch(owner) or not REPO_RE.fullmatch(repo):
        return {
            "source": "github",
            "owner": owner,
            "repo": repo,
            "full_name": full_name,
            "url": str((repo_item or {}).get("url") or ""),
            "status": "invalid_repo_item",
            "format_valid": False,
            "exists": False,
            "error": "invalid_repo_format",
            "confidence": "high",
            "use_in_scoring": True,
            "checked_at_utc": utc_now(),
            "latency_ms": 0,
        }

    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    started = time.perf_counter()

    headers = {
        "User-Agent": "DomainGuard/0.1",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(api_url, method="GET", headers=headers)

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body else {}
            return {
                "source": "github",
                "owner": owner,
                "repo": repo,
                "full_name": full_name,
                "url": str((repo_item or {}).get("url") or ""),
                "status": "repo_exists",
                "format_valid": True,
                "exists": True,
                "api_url": api_url,
                "html_url": data.get("html_url"),
                "description": data.get("description") or "",
                "stars": data.get("stargazers_count"),
                "forks": data.get("forks_count"),
                "open_issues": data.get("open_issues_count"),
                "archived": data.get("archived"),
                "disabled": data.get("disabled"),
                "private": data.get("private"),
                "language": data.get("language"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "pushed_at": data.get("pushed_at"),
                "confidence": "high",
                "use_in_scoring": True,
                "checked_at_utc": utc_now(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
    except HTTPError as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if exc.code == 404:
            return {
                "source": "github",
                "owner": owner,
                "repo": repo,
                "full_name": full_name,
                "url": str((repo_item or {}).get("url") or ""),
                "status": "repo_not_found",
                "format_valid": True,
                "exists": False,
                "api_url": api_url,
                "status_code": 404,
                "confidence": "high",
                "use_in_scoring": True,
                "checked_at_utc": utc_now(),
                "latency_ms": latency_ms,
            }

        if exc.code in {403, 429}:
            return {
                "source": "github",
                "owner": owner,
                "repo": repo,
                "full_name": full_name,
                "url": str((repo_item or {}).get("url") or ""),
                "status": "github_rate_limited",
                "format_valid": True,
                "exists": None,
                "api_url": api_url,
                "status_code": exc.code,
                "error": str(exc),
                "confidence": "high",
                "use_in_scoring": False,
                "checked_at_utc": utc_now(),
                "latency_ms": latency_ms,
            }

        return {
            "source": "github",
            "owner": owner,
            "repo": repo,
            "full_name": full_name,
            "url": str((repo_item or {}).get("url") or ""),
            "status": "github_http_error",
            "format_valid": True,
            "exists": None,
            "api_url": api_url,
            "status_code": exc.code,
            "error": str(exc),
            "confidence": "high",
            "use_in_scoring": False,
            "checked_at_utc": utc_now(),
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        return {
            "source": "github",
            "owner": owner,
            "repo": repo,
            "full_name": full_name,
            "url": str((repo_item or {}).get("url") or ""),
            "status": "github_error",
            "format_valid": True,
            "exists": None,
            "api_url": api_url,
            "error": str(exc),
            "confidence": "high",
            "use_in_scoring": False,
            "checked_at_utc": utc_now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
