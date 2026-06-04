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

"""Verification module for DNS, HTTP, TLS, WHOIS/RDAP, and GitHub checks."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
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


def check_tls(domain: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Verify TLS certificate status for a domain.

    Returns status values such as:
    - tls_ok
    - tls_expired
    - tls_expiring_soon
    - tls_hostname_mismatch
    - tls_untrusted_chain
    - tls_verification_failed
    - tls_timeout
    - tls_error
    - tls_unchecked
    """
    domain = (domain or "").strip().lower()
    started = time.perf_counter()

    if not domain or "." not in domain:
        return {
            "source": "tls",
            "domain": domain,
            "status": "tls_unchecked",
            "confidence": "low",
            "use_in_scoring": False,
            "error": "invalid domain",
            "checked_at_utc": utc_now(),
            "latency_ms": 0,
        }

    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as secure_sock:
                cert = secure_sock.getpeercert()
                not_after = cert.get("notAfter", "")
                expiry_ts = ssl.cert_time_to_seconds(not_after) if not_after else None
                days_until_expiry = None
                status = "tls_ok"
                if expiry_ts is not None:
                    days_until_expiry = int((expiry_ts - time.time()) / 86400)
                    if days_until_expiry < 0:
                        status = "tls_expired"
                    elif days_until_expiry <= 30:
                        status = "tls_expiring_soon"

                return {
                    "source": "tls",
                    "domain": domain,
                    "status": status,
                    "days_until_expiry": days_until_expiry,
                    "not_after": not_after,
                    "confidence": "medium",
                    "use_in_scoring": True,
                    "checked_at_utc": utc_now(),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
    except ssl.SSLCertVerificationError as exc:
        message = str(exc).lower()
        if "hostname" in message or "not valid for" in message:
            status = "tls_hostname_mismatch"
        elif "self-signed" in message or "verify failed" in message or "unable to get local issuer" in message:
            status = "tls_untrusted_chain"
        else:
            status = "tls_verification_failed"
        return {
            "source": "tls",
            "domain": domain,
            "status": status,
            "confidence": "medium",
            "use_in_scoring": True,
            "error": str(exc),
            "checked_at_utc": utc_now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except (TimeoutError, socket.timeout) as exc:
        return {
            "source": "tls",
            "domain": domain,
            "status": "tls_timeout",
            "confidence": "low",
            "use_in_scoring": False,
            "error": str(exc),
            "checked_at_utc": utc_now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        return {
            "source": "tls",
            "domain": domain,
            "status": "tls_error",
            "confidence": "low",
            "use_in_scoring": False,
            "error": str(exc),
            "checked_at_utc": utc_now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def _parse_whois_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, list) and value:
        return _parse_whois_date(value[0])
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _calculate_age_days(created: datetime | None) -> int | None:
    if created is None:
        return None
    return (datetime.now(timezone.utc) - created).days


def _calculate_days_until(expires: datetime | None) -> int | None:
    if expires is None:
        return None
    return (expires - datetime.now(timezone.utc)).days


def _build_whois_payload(
    *,
    domain: str,
    status: str,
    enabled: bool,
    registrar: Any = None,
    created: datetime | None = None,
    expires: datetime | None = None,
    source_method: str,
    started: float,
    error: str | None = None,
    confidence: str = "low",
    use_in_scoring: bool = True,
) -> Dict[str, Any]:
    age_days = _calculate_age_days(created)
    is_recent = age_days is not None and age_days < 30
    payload: Dict[str, Any] = {
        "source": "whois",
        "domain": domain,
        "status": status,
        "enabled": enabled,
        "source_method": source_method,
        "registrar": registrar,
        "created_at": created.isoformat() if created else None,
        "expires_at": expires.isoformat() if expires else None,
        "age_days": age_days,
        "days_until_expiration": _calculate_days_until(expires),
        "is_recent_domain": is_recent if age_days is not None else None,
        "risk_hint": "recent_domain" if is_recent else None,
        "confidence": confidence,
        "use_in_scoring": use_in_scoring and age_days is not None,
        "checked_at_utc": utc_now(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if error:
        payload["error"] = error
    return payload


def _extract_rdap_event_date(data: Dict[str, Any], actions: set[str]) -> datetime | None:
    events = data.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        action = str(event.get("eventAction") or "").strip().lower()
        if action in actions:
            parsed = _parse_whois_date(event.get("eventDate"))
            if parsed is not None:
                return parsed
    return None


def _extract_rdap_registrar(data: Dict[str, Any]) -> str | None:
    entities = data.get("entities")
    if not isinstance(entities, list):
        return None
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        roles = [str(role).lower() for role in entity.get("roles", []) if isinstance(role, str)]
        if "registrar" not in roles:
            continue
        vcard = entity.get("vcardArray")
        if not (isinstance(vcard, list) and len(vcard) >= 2 and isinstance(vcard[1], list)):
            continue
        for item in vcard[1]:
            if isinstance(item, list) and item and item[0] == "fn" and len(item) >= 4:
                return str(item[3])
    return None


def _registered_domain_candidates(domain: str) -> list[str]:
    parts = [part for part in domain.split(".") if part]
    candidates = [domain]
    if len(parts) >= 3 and parts[-2] in {"com", "net", "org", "gov", "edu", "ac", "co"} and len(parts[-1]) == 2:
        candidates.append(".".join(parts[-3:]))
    elif len(parts) >= 2:
        candidates.append(".".join(parts[-2:]))
    return list(dict.fromkeys(candidates))


def _check_rdap(domain: str, timeout: float, started: float) -> Dict[str, Any]:
    last_error: Exception | None = None
    queried_domain = domain
    data: Dict[str, Any] | None = None

    for candidate in _registered_domain_candidates(domain):
        queried_domain = candidate
        url = f"https://rdap.org/domain/{candidate}"
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": "DomainGuard/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(256_000).decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("RDAP response is not a JSON object")
            data = parsed
            break
        except Exception as exc:
            last_error = exc

    if data is None:
        if last_error is not None:
            raise last_error
        raise ValueError("RDAP lookup failed")

    created = _extract_rdap_event_date(data, {"registration", "registered"})
    expires = _extract_rdap_event_date(data, {"expiration", "expiry"})
    registrar = _extract_rdap_registrar(data)
    return _build_whois_payload(
        domain=domain,
        status="ok" if created or expires or registrar else "rdap_no_registration_dates",
        enabled=True,
        registrar=registrar,
        created=created,
        expires=expires,
        source_method="rdap",
        started=started,
        confidence="medium" if created else "low",
        use_in_scoring=True,
    ) | {"queried_domain": queried_domain}


def check_whois(domain: str, timeout: float = 6.0) -> Dict[str, Any]:
    """Check basic WHOIS metadata and flag very recent registrations.

    RDAP is used for batch experiments because it uses HTTPS. Traditional
    WHOIS libraries often open raw port-43 sockets, which are slow or blocked
    in proxied network environments and can stall large experiment runs.
    """
    domain = (domain or "").strip().lower()
    started = time.perf_counter()
    if not domain:
        return _build_whois_payload(
            domain=domain,
            status="invalid_domain",
            enabled=False,
            source_method="none",
            started=started,
            error="invalid domain",
            use_in_scoring=False,
        )

    try:
        return _check_rdap(domain=domain, timeout=timeout, started=started)
    except (TimeoutError, socket.timeout) as rdap_exc:
        return _build_whois_payload(
            domain=domain,
            status="whois_timeout",
            enabled=False,
            source_method="rdap",
            started=started,
            error=f"rdap timed out: {rdap_exc}",
            use_in_scoring=False,
        )
    except (urllib.error.URLError, json.JSONDecodeError, ValueError) as rdap_exc:
        return _build_whois_payload(
            domain=domain,
            status="whois_error",
            enabled=False,
            source_method="rdap",
            started=started,
            error=f"rdap failed: {rdap_exc}",
            use_in_scoring=False,
        )
    except Exception as rdap_exc:
        return _build_whois_payload(
            domain=domain,
            status="whois_error",
            enabled=False,
            source_method="rdap",
            started=started,
            error=f"rdap failed: {type(rdap_exc).__name__}: {rdap_exc}",
            use_in_scoring=False,
        )


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
