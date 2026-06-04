#!/usr/bin/env python3
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

"""End-to-end URL/domain hallucination evaluation pipeline.

Flow:
1. Select questions from a question-pool JSON.
2. Ask an LLM to answer while encouraging concrete URLs.
3. Independently verify generated URLs/domains/GitHub repos for ground truth.
4. Run one or more guard modes.
5. Save per-question progress and aggregate metrics.

This script is intentionally self-contained under files/ so it does not change
the existing NeMo library behavior or the A/B/C ablation runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
FILES_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful technical assistant. When answering questions about "
    "tools, libraries, APIs, downloads, or documentation, always include "
    "direct links (URLs) when possible. Be specific and provide actual URLs, "
    "GitHub repository links, and documentation pages."
)

POSITIVE_LABELS = {"hallucinated", "mixed"}
GUARD_MODES = {
    "domain-s1",
    "domain-s2",
    "domain-s3",
    "domain-s4",
    "domain-expert-s1",
    "domain-expert-s2",
    "domain-expert-s3",
    "domain-expert-s4",
    "nemo-hallucination",
}


@dataclass
class GuardResult:
    decision: str
    normalized: str
    latency_ms: float
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class LLMClient:
    """Small async client for OpenAI-compatible, Anthropic, or Ollama models."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 600,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, question: str) -> str:
        if self.provider in {"openai", "deepseek"}:
            return await self._call_openai_compatible(question)
        if self.provider == "anthropic":
            return await self._call_anthropic(question)
        if self.provider == "ollama":
            return await self._call_ollama(question)
        raise ValueError(f"Unknown provider: {self.provider}")

    async def _call_openai_compatible(self, question: str) -> str:
        try:
            import openai

            kwargs: Dict[str, Any] = {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            env_name = self.api_key_env or ("DEEPSEEK_API_KEY" if self.provider == "deepseek" else "OPENAI_API_KEY")
            api_key = os.environ.get(env_name)
            if api_key:
                kwargs["api_key"] = api_key

            client = openai.AsyncOpenAI(**kwargs)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            return f"[LLM_ERROR: {type(exc).__name__}: {exc}]"

    async def _call_anthropic(self, question: str) -> str:
        try:
            import anthropic

            kwargs: Dict[str, Any] = {}
            env_name = self.api_key_env or "ANTHROPIC_API_KEY"
            api_key = os.environ.get(env_name)
            if api_key:
                kwargs["api_key"] = api_key
            client = anthropic.AsyncAnthropic(**kwargs)
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": question}],
            )
            return response.content[0].text
        except Exception as exc:
            return f"[LLM_ERROR: {type(exc).__name__}: {exc}]"

    async def _call_ollama(self, question: str) -> str:
        try:
            import httpx

            url = self.base_url or "http://localhost:11434"
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(
                    f"{url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": f"{self.system_prompt}\n\nQuestion: {question}",
                        "stream": False,
                    },
                )
                return response.json().get("response", "")
        except Exception as exc:
            return f"[LLM_ERROR: {type(exc).__name__}: {exc}]"


class GroundTruthVerifier:
    """Independent verifier for generated URL/domain/GitHub claims."""

    RESERVED_DOMAINS = {"example.com", "example.org", "example.net", "localhost"}
    TRUSTED_ROOT_DOMAINS = {
        "github.com",
        "stackoverflow.com",
        "google.com",
        "python.org",
        "npmjs.com",
        "pypi.org",
        "arxiv.org",
        "wikipedia.org",
        "docker.com",
        "kubernetes.io",
        "rust-lang.org",
        "mozilla.org",
    }

    def __init__(
        self,
        *,
        github_token: Optional[str] = None,
        timeout: float = 8.0,
        enable_http: bool = True,
    ) -> None:
        self.github_token = github_token or os.getenv("GITHUB_TOKEN", "")
        self.timeout = timeout
        self.enable_http = enable_http
        self._dns_cache: Dict[str, Dict[str, Any]] = {}
        self._http_cache: Dict[str, Dict[str, Any]] = {}
        self._github_cache: Dict[str, Dict[str, Any]] = {}

    def extract_urls(self, text: str) -> List[str]:
        pattern = re.compile(r"https?://[^\s<>()\[\]{}\"'`]+", flags=re.IGNORECASE)
        urls = []
        for item in pattern.findall(text or ""):
            cleaned = item.rstrip(".,;:!?)>]}'\"")
            if cleaned:
                urls.append(cleaned)
        return sorted(set(urls))

    def extract_domains(self, urls: Iterable[str]) -> List[str]:
        domains = set()
        for url in urls:
            try:
                host = urlparse(url).hostname
                if host:
                    domains.add(host.lower())
            except Exception:
                continue
        return sorted(domains)

    def extract_github_repos(self, urls: Iterable[str]) -> List[Dict[str, str]]:
        repos: List[Dict[str, str]] = []
        for url in urls:
            try:
                parsed = urlparse(url)
            except Exception:
                continue
            if (parsed.hostname or "").lower() not in {"github.com", "www.github.com"}:
                continue
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) < 2:
                continue
            owner, repo = parts[0], parts[1].removesuffix(".git")
            if owner in {"topics", "search", "settings", "orgs", "login", "signup", "features"}:
                continue
            if repo in {"tree", "blob", "issues", "pulls", "actions", "wiki", "releases"}:
                continue
            repos.append({"owner": owner, "repo": repo, "url": url})
        seen = set()
        unique = []
        for repo in repos:
            key = f"{repo['owner'].lower()}/{repo['repo'].lower()}"
            if key not in seen:
                seen.add(key)
                unique.append(repo)
        return unique

    async def verify_dns(self, domain: str) -> Dict[str, Any]:
        domain = (domain or "").strip().lower()
        if domain in self._dns_cache:
            return {**self._dns_cache[domain], "cache_hit": True}
        if domain in self.RESERVED_DOMAINS:
            result = {"domain": domain, "exists": True, "status": "reserved", "method": "dns"}
            self._dns_cache[domain] = result
            return result

        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(loop.run_in_executor(None, socket.getaddrinfo, domain, 443), timeout=self.timeout)
            result = {"domain": domain, "exists": True, "status": "resolved", "method": "dns"}
        except (socket.gaierror, asyncio.TimeoutError, OSError) as exc:
            result = {
                "domain": domain,
                "exists": False,
                "status": "nxdomain_or_no_data",
                "method": "dns",
                "error": str(exc)[:160],
            }
        self._dns_cache[domain] = result
        return result

    async def verify_http(self, url: str) -> Dict[str, Any]:
        if url in self._http_cache:
            return {**self._http_cache[url], "cache_hit": True}
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, verify=False) as client:
                response = await client.head(url)
                if response.status_code in {403, 405}:
                    response = await client.get(url)
                status = response.status_code
                result = {
                    "url": url,
                    "reachable": status < 500 and status != 404,
                    "status_code": status,
                    "final_url": str(response.url),
                    "method": "http",
                }
        except Exception as exc:
            result = {
                "url": url,
                "reachable": None,
                "status_code": None,
                "method": "http",
                "error": str(exc)[:160],
            }
        self._http_cache[url] = result
        return result

    async def verify_github_repo(self, owner: str, repo: str) -> Dict[str, Any]:
        key = f"{owner.lower()}/{repo.lower()}"
        if key in self._github_cache:
            return {**self._github_cache[key], "cache_hit": True}
        try:
            import httpx

            headers = {"Accept": "application/vnd.github+json", "User-Agent": "DomainGuard-E2E/0.1"}
            if self.github_token:
                headers["Authorization"] = f"Bearer {self.github_token}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
            if response.status_code == 200:
                data = response.json()
                result = {
                    "owner": owner,
                    "repo": repo,
                    "exists": True,
                    "status": "repo_exists",
                    "stars": data.get("stargazers_count", 0),
                    "archived": data.get("archived", False),
                    "method": "github_api",
                }
            elif response.status_code == 404:
                result = {
                    "owner": owner,
                    "repo": repo,
                    "exists": False,
                    "status": "repo_not_found",
                    "method": "github_api",
                }
            else:
                result = {
                    "owner": owner,
                    "repo": repo,
                    "exists": None,
                    "status": f"github_status_{response.status_code}",
                    "method": "github_api",
                }
        except Exception as exc:
            result = {
                "owner": owner,
                "repo": repo,
                "exists": None,
                "status": "github_error",
                "method": "github_api",
                "error": str(exc)[:160],
            }
        self._github_cache[key] = result
        return result

    async def verify_answer(self, answer: str) -> Dict[str, Any]:
        urls = self.extract_urls(answer)
        domains = self.extract_domains(urls)
        repos = self.extract_github_repos(urls)

        if not urls and not domains:
            return {
                "entities": {"urls": [], "domains": [], "repos": []},
                "verifications": {"dns": [], "http": [], "repos": []},
                "ground_truth_label": "no_links",
                "hallucinated_items": [],
                "real_items": [],
                "unknown_items": [],
            }

        dns_results = await asyncio.gather(*(self.verify_dns(domain) for domain in domains), return_exceptions=True)
        repo_results = await asyncio.gather(
            *(self.verify_github_repo(repo["owner"], repo["repo"]) for repo in repos),
            return_exceptions=True,
        )
        http_results: List[Any] = []
        if self.enable_http:
            http_results = await asyncio.gather(*(self.verify_http(url) for url in urls), return_exceptions=True)

        hallucinated: List[Dict[str, Any]] = []
        real: List[Dict[str, Any]] = []
        unknown: List[Dict[str, Any]] = []

        for result in dns_results:
            if isinstance(result, Exception):
                unknown.append({"type": "domain", "error": str(result)})
            elif result.get("exists") is False:
                hallucinated.append({"type": "domain", "value": result["domain"], "reason": "DNS resolution failed"})
            elif result.get("exists") is True:
                real.append({"type": "domain", "value": result["domain"]})

        for result in repo_results:
            if isinstance(result, Exception):
                unknown.append({"type": "github_repo", "error": str(result)})
                continue
            full_name = f"{result.get('owner', '?')}/{result.get('repo', '?')}"
            if result.get("exists") is False:
                hallucinated.append({"type": "github_repo", "value": full_name, "reason": "GitHub API returned 404"})
            elif result.get("exists") is True:
                real.append({"type": "github_repo", "value": full_name, "stars": result.get("stars", 0)})
            else:
                unknown.append({"type": "github_repo", "value": full_name, "reason": result.get("status", "unknown")})

        for result in http_results:
            if isinstance(result, Exception):
                unknown.append({"type": "url", "error": str(result)})
                continue
            status = result.get("status_code")
            if status == 404:
                hallucinated.append({"type": "url_path", "value": result.get("url"), "reason": "HTTP 404"})
            elif result.get("reachable") is True:
                real.append({"type": "url", "value": result.get("url"), "status_code": status})
            elif result.get("reachable") is None:
                unknown.append({"type": "url", "value": result.get("url"), "reason": result.get("error", "unknown")})

        if hallucinated and real:
            label = "mixed"
        elif hallucinated:
            label = "hallucinated"
        else:
            label = "clean"

        return {
            "entities": {
                "urls": urls,
                "domains": domains,
                "repos": [f"{repo['owner']}/{repo['repo']}" for repo in repos],
            },
            "verifications": {
                "dns": [item for item in dns_results if not isinstance(item, Exception)],
                "http": [item for item in http_results if not isinstance(item, Exception)],
                "repos": [item for item in repo_results if not isinstance(item, Exception)],
            },
            "ground_truth_label": label,
            "hallucinated_items": hallucinated,
            "real_items": real,
            "unknown_items": unknown,
        }


class DomainGuardAdapter:
    """Adapter for domain_hallucination modes."""

    def __init__(self, mode: str, *, nemo_config: str, github_token: Optional[str] = None) -> None:
        self.mode = mode
        self.nemo_config = nemo_config
        self.github_token = github_token
        self._runtime = None

    def _runtime_for_expert(self) -> Any:
        if self._runtime is None:
            from run_ablation_experiments import NemoRuntime

            self._runtime = NemoRuntime(config_path=FILES_ROOT / "02_configs" / self.nemo_config)
        return self._runtime

    async def analyze_answer(self, answer: str, user_query: str) -> Dict[str, Any]:
        from nemoguardrails.library.domain_hallucination.actions import analyze_answer, self_check_domain_hallucination

        token = self.github_token or os.environ.get("GITHUB_TOKEN")
        strategy = self._strategy()
        if not strategy["expert"]:
            return await analyze_answer(
                answer=answer,
                user_query=user_query,
                verification_level=strategy["verification_level"],
                enable_semantic_check=True,
                enable_advanced_verification=True,
                skip_secondary_checks_on_dns_failure=strategy["skip_dnsfail"],
                enable_tls_verification=strategy["tls"],
                enable_whois_verification=strategy["whois"],
                github_token=token,
            )
        if strategy["expert"]:
            runtime = self._runtime_for_expert()
            return await self_check_domain_hallucination(
                llm=runtime.llm,
                context={"bot_message": answer, "user_message": user_query},
                verification_level=strategy["verification_level"],
                enable_semantic_check=True,
                enable_advanced_verification=True,
                skip_secondary_checks_on_dns_failure=strategy["skip_dnsfail"],
                enable_tls_verification=strategy["tls"],
                enable_whois_verification=strategy["whois"],
                enable_expert_review=True,
                expert_review_min_level="L2",
                github_token=token,
            )
        raise ValueError(f"Unsupported domain guard mode: {self.mode}")

    def _strategy(self) -> Dict[str, Any]:
        strategies = {
            "domain-s1": {
                "expert": False,
                "verification_level": "full",
                "skip_dnsfail": False,
                "tls": True,
                "whois": True,
            },
            "domain-s2": {
                "expert": False,
                "verification_level": "full",
                "skip_dnsfail": True,
                "tls": True,
                "whois": True,
            },
            "domain-s3": {
                "expert": False,
                "verification_level": "full",
                "skip_dnsfail": True,
                "tls": True,
                "whois": False,
            },
            "domain-s4": {
                "expert": False,
                "verification_level": "http",
                "skip_dnsfail": True,
                "tls": False,
                "whois": False,
            },
            "domain-expert-s1": {
                "expert": True,
                "verification_level": "full",
                "skip_dnsfail": False,
                "tls": True,
                "whois": True,
            },
            "domain-expert-s2": {
                "expert": True,
                "verification_level": "full",
                "skip_dnsfail": True,
                "tls": True,
                "whois": True,
            },
            "domain-expert-s3": {
                "expert": True,
                "verification_level": "full",
                "skip_dnsfail": True,
                "tls": True,
                "whois": False,
            },
            "domain-expert-s4": {
                "expert": True,
                "verification_level": "http",
                "skip_dnsfail": True,
                "tls": False,
                "whois": False,
            },
        }
        if self.mode not in strategies:
            raise ValueError(f"Unsupported domain guard mode: {self.mode}")
        return strategies[self.mode]


class NemoHallucinationAdapter:
    """Adapter for NeMo library/hallucination baseline."""

    def __init__(self, *, nemo_config: str, extra_responses: int, retries: int) -> None:
        from run_ablation_experiments import NemoRuntime

        self.runtime = NemoRuntime(config_path=FILES_ROOT / "02_configs" / nemo_config)
        self.extra_responses = extra_responses
        self.retries = retries

    async def analyze_answer(self, answer: str, user_query: str) -> Dict[str, Any]:
        from run_ablation_experiments import run_hallucination_baseline_sequential

        hallucinated, raw = await run_hallucination_baseline_sequential(
            runtime=self.runtime,
            user_query=user_query,
            bot_response=answer,
            extra_responses_count=self.extra_responses,
            max_retries=self.retries,
        )
        return {
            "decision": {"action": "block" if hallucinated else "pass"},
            "risk_score": None,
            "detection": {"issues": []},
            "raw_baseline": {
                "connected_action": "nemoguardrails.library.hallucination.actions.self_check_hallucination",
                "requires_extra_llm_calls": True,
                "hallucinated": hallucinated,
                **raw,
            },
        }


def normalize_guard_decision(decision: str) -> str:
    mapping = {"block": "hallucinated", "refine": "mixed", "warn": "mixed", "pass": "clean"}
    return mapping.get(str(decision or "").lower(), "clean")


def normalize_ground_truth(label: str) -> str:
    return "clean" if label == "no_links" else label


def is_positive(label: str) -> bool:
    return label in POSITIVE_LABELS


class IncrementalWriter:
    def __init__(self, path: Path, metadata: Dict[str, Any]) -> None:
        self.path = path
        self.metadata = metadata
        self.results: List[Dict[str, Any]] = []

        # === RESUME SUPPORT ===
        # Load existing results if partial file exists
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    existing = json.load(f)
                    self.results = existing.get("results", [])
                    print(f"[WRITER] Loaded {len(self.results)} existing results from {self.path.name}")
            except Exception as e:
                print(f"[WRITER] Could not load existing results: {e}")

    def append(self, result: Dict[str, Any]) -> None:
        self.results.append(result)
        self.save()

    def save(self) -> None:
        payload = {
            "metadata": self.metadata,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed": len(self.results),
            "results": self.results,
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


class E2EPipeline:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        verifier: GroundTruthVerifier,
        guard_adapters: Dict[str, Any],
        question_pool: Dict[str, Any],
        questions_per_category: int,
        writer: IncrementalWriter,
    ) -> None:
        self.llm = llm_client
        self.verifier = verifier
        self.guard_adapters = guard_adapters
        self.question_pool = question_pool
        self.questions_per_category = questions_per_category
        self.writer = writer

    def select_questions(self) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for category, data in self.question_pool.items():
            questions = data.get("questions", [])
            for index, question in enumerate(questions[: self.questions_per_category]):
                selected.append(
                    {
                        "id": f"{category}_{index + 1:03d}",
                        "category": category,
                        "question": question,
                        "description": data.get("description", ""),
                        "risk_level": data.get("risk_level"),
                    }
                )
        return selected

    async def run_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["question"]
        result: Dict[str, Any] = {
            **item,
            "timestamps": {},
            "guard_results": {},
        }

        t0 = time.perf_counter()
        answer = await self.llm.generate(question)
        result["timestamps"]["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["llm_answer"] = answer
        if answer.startswith("[LLM_ERROR"):
            result["error"] = answer
            return result

        t1 = time.perf_counter()
        ground_truth = await self.verifier.verify_answer(answer)
        result["timestamps"]["verification_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        result["ground_truth"] = ground_truth
        gt_label = normalize_ground_truth(ground_truth["ground_truth_label"])

        for mode, adapter in self.guard_adapters.items():
            t2 = time.perf_counter()
            try:
                guard_result = await adapter.analyze_answer(answer=answer, user_query=question)
                latency = round((time.perf_counter() - t2) * 1000, 1)
                decision = guard_result.get("decision", {}).get("action", "error")
                normalized = normalize_guard_decision(decision)
                detection = guard_result.get("detection", {}) or {}
                result["guard_results"][mode] = {
                    "decision": decision,
                    "normalized": normalized,
                    "match": normalized == gt_label,
                    "latency_ms": latency,
                    "risk_score": guard_result.get("risk_score"),
                    "issues": detection.get("issues", []),
                    "expert_review": guard_result.get("expert_review"),
                    "raw_baseline": guard_result.get("raw_baseline"),
                }
            except Exception as exc:
                result["guard_results"][mode] = {
                    "decision": "error",
                    "normalized": "clean",
                    "match": False,
                    "latency_ms": round((time.perf_counter() - t2) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        return result

    async def run_all(self) -> Dict[str, Any]:
        questions = self.select_questions()

        # === RESUME SUPPORT ===
        # Check if partial file exists and load already-completed question IDs
        completed_ids = set()
        if self.writer.path.exists():
            try:
                with open(self.writer.path, encoding="utf-8") as f:
                    partial_data = json.load(f)
                    completed_ids = {r["id"] for r in partial_data.get("results", [])}
                    if completed_ids:
                        print(f"\n[RESUME] Found {len(completed_ids)} completed questions")
            except Exception as e:
                print(f"[RESUME] Could not load partial file: {e}")

        # Filter questions: only run those not yet completed
        questions_to_run = [q for q in questions if q["id"] not in completed_ids]

        print("\n=== Domain Hallucination E2E Evaluation ===")
        print(f"LLM: {self.llm.provider}/{self.llm.model}")
        print(f"Total questions: {len(questions)} ({self.questions_per_category} per category)")
        print(f"Completed: {len(completed_ids)}")
        print(f"Remaining: {len(questions_to_run)}")
        print(f"Guard modes: {', '.join(self.guard_adapters) if self.guard_adapters else 'none'}\n")

        results = []
        for index, item in enumerate(questions_to_run, start=len(completed_ids) + 1):
            print(f"[{index:03d}/{len(questions):03d}] {item['category']:<32}", end=" ")
            single = await self.run_single(item)
            results.append(single)
            self.writer.append(single)
            if "error" in single:
                print(f"ERROR {single['error'][:80]}")
                continue
            gt = single.get("ground_truth", {}).get("ground_truth_label")
            guards = " ".join(
                f"{mode}:{data.get('decision')}" for mode, data in single.get("guard_results", {}).items()
            )
            hall = len(single.get("ground_truth", {}).get("hallucinated_items", []))
            real = len(single.get("ground_truth", {}).get("real_items", []))
            print(f"GT={gt:<12} real={real:<2} hall={hall:<2} {guards}")

        return self.aggregate(results)

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        gt_dist = Counter(result.get("ground_truth", {}).get("ground_truth_label", "error") for result in results)
        with_links = [
            result
            for result in results
            if result.get("ground_truth", {}).get("ground_truth_label") not in {"no_links", "error"}
            and "error" not in result
        ]
        hallucinated = [
            result
            for result in results
            if result.get("ground_truth", {}).get("ground_truth_label") in {"hallucinated", "mixed"}
        ]

        per_category: Dict[str, Dict[str, Any]] = {}
        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for result in results:
            by_category[result["category"]].append(result)
        for category, items in sorted(by_category.items()):
            category_with_links = [
                item for item in items if item.get("ground_truth", {}).get("ground_truth_label") != "no_links"
            ]
            category_hall = [
                item
                for item in items
                if item.get("ground_truth", {}).get("ground_truth_label") in {"hallucinated", "mixed"}
            ]
            per_category[category] = {
                "total": len(items),
                "with_links": len(category_with_links),
                "hallucinated_or_mixed": len(category_hall),
                "rate": round(len(category_hall) / len(category_with_links), 4) if category_with_links else 0.0,
            }

        guard_metrics = {mode: compute_guard_metrics(results, mode) for mode in self.guard_adapters}
        latency = {
            "llm_ms": lat_stats([r.get("timestamps", {}).get("llm_ms", 0) for r in results if "error" not in r]),
            "verification_ms": lat_stats(
                [r.get("timestamps", {}).get("verification_ms", 0) for r in results if "error" not in r]
            ),
        }
        for mode in self.guard_adapters:
            latency[f"{mode}_ms"] = lat_stats(
                [
                    r.get("guard_results", {}).get(mode, {}).get("latency_ms", 0)
                    for r in results
                    if mode in r.get("guard_results", {})
                ]
            )

        return {
            "summary": {
                "llm_model": f"{self.llm.provider}/{self.llm.model}",
                "total_questions": len(results),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "ground_truth_distribution": dict(gt_dist),
            "llm_hallucination_rate": {
                "total_with_links": len(with_links),
                "hallucinated_or_mixed": len(hallucinated),
                "rate": round(len(hallucinated) / len(with_links), 4) if with_links else 0.0,
            },
            "per_category_hallucination": per_category,
            "guard_performance": guard_metrics,
            "latency": latency,
            "detailed_results": results,
        }


def compute_guard_metrics(results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    y_true: List[str] = []
    y_pred: List[str] = []
    for result in results:
        if "error" in result:
            continue
        gt = normalize_ground_truth(result.get("ground_truth", {}).get("ground_truth_label", "clean"))
        guard = result.get("guard_results", {}).get(mode)
        if not guard or guard.get("decision") == "error":
            continue
        y_true.append(gt)
        y_pred.append(guard.get("normalized", "clean"))

    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    tp = sum(1 for true, pred in zip(y_true, y_pred) if is_positive(true) and is_positive(pred))
    fp = sum(1 for true, pred in zip(y_true, y_pred) if not is_positive(true) and is_positive(pred))
    fn = sum(1 for true, pred in zip(y_true, y_pred) if is_positive(true) and not is_positive(pred))
    tn = sum(1 for true, pred in zip(y_true, y_pred) if not is_positive(true) and not is_positive(pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    labels = sorted(set(y_true + y_pred))
    confusion = {
        true_label: {
            pred_label: sum(1 for true, pred in zip(y_true, y_pred) if true == true_label and pred == pred_label)
            for pred_label in labels
        }
        for true_label in labels
    }
    return {
        "total_evaluated": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "binary_detection": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "confusion_matrix": confusion,
    }


def lat_stats(values: List[float]) -> Dict[str, float]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "mean": round(sum(sorted_values) / len(sorted_values), 1),
        "p50": round(sorted_values[len(sorted_values) // 2], 1),
        "p95": round(sorted_values[min(int(len(sorted_values) * 0.95), len(sorted_values) - 1)], 1),
        "min": round(sorted_values[0], 1),
        "max": round(sorted_values[-1], 1),
    }


def load_question_pool(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("categories", data)


def build_guard_adapters(args: argparse.Namespace) -> Dict[str, Any]:
    adapters: Dict[str, Any] = {}
    for mode in args.guard_modes:
        if mode == "none":
            continue
        if mode.startswith("domain-"):
            adapters[mode] = DomainGuardAdapter(mode, nemo_config=args.nemo_config, github_token=args.github_token)
        elif mode == "nemo-hallucination":
            adapters[mode] = NemoHallucinationAdapter(
                nemo_config=args.nemo_config,
                extra_responses=args.hallucination_extra_responses,
                retries=args.hallucination_retries,
            )
        else:
            raise SystemExit(f"Unknown guard mode: {mode}")
    return adapters


def print_report(report: Dict[str, Any]) -> None:
    print("\n=== E2E Report ===")
    print(f"LLM: {report['summary']['llm_model']}")
    print(f"Total questions: {report['summary']['total_questions']}")
    print("Ground truth:", report["ground_truth_distribution"])
    rate = report["llm_hallucination_rate"]
    print(f"LLM hallucination rate: {rate['hallucinated_or_mixed']}/{rate['total_with_links']} ({rate['rate']:.1%})")
    print("\nGuard performance:")
    for mode, metrics in report.get("guard_performance", {}).items():
        binary = metrics.get("binary_detection", {})
        print(
            f"  {mode:<24} acc={metrics.get('accuracy', 0):.1%} "
            f"precision={binary.get('precision', 0):.4f} "
            f"recall={binary.get('recall', 0):.4f} "
            f"f1={binary.get('f1', 0):.4f} "
            f"TP={binary.get('tp', 0)} FP={binary.get('fp', 0)} "
            f"FN={binary.get('fn', 0)} TN={binary.get('tn', 0)}"
        )
    print("\nLatency:")
    for name, stats in report.get("latency", {}).items():
        print(f"  {name:<28} mean={stats['mean']:.0f}ms p95={stats['p95']:.0f}ms")


async def main_async(args: argparse.Namespace) -> None:
    question_pool_path = FILES_ROOT / "01_datasets" / args.questions
    question_pool = load_question_pool(question_pool_path)

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    llm = LLMClient(
        provider=args.llm_provider,
        model=args.model,
        base_url=args.base_url or ("https://api.deepseek.com" if args.llm_provider == "deepseek" else None),
        api_key_env=args.api_key_env,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    verifier = GroundTruthVerifier(
        github_token=args.github_token,
        timeout=args.verifier_timeout,
        enable_http=not args.disable_ground_truth_http,
    )
    adapters = build_guard_adapters(args)

    output = FILES_ROOT / "06_e2e_eval" / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question_pool": str(question_pool_path),
        "llm_provider": args.llm_provider,
        "model": args.model,
        "guard_modes": args.guard_modes,
        "questions_per_category": args.questions_per_category,
        "ground_truth_http_enabled": not args.disable_ground_truth_http,
    }
    writer = IncrementalWriter(output.with_suffix(output.suffix + ".partial.json"), metadata)
    pipeline = E2EPipeline(
        llm_client=llm,
        verifier=verifier,
        guard_adapters=adapters,
        question_pool=question_pool,
        questions_per_category=args.questions_per_category,
        writer=writer,
    )
    report = await pipeline.run_all()
    report["metadata"] = metadata

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
    print_report(report)
    print(f"\nSaved E2E report to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end domain hallucination evaluation.")
    parser.add_argument(
        "--questions", default="question_pool_v2.json", help="Question pool JSON under files/01_datasets/."
    )
    parser.add_argument("--questions-per-category", type=int, default=1, help="Questions selected from each category.")
    parser.add_argument(
        "--guard-modes",
        nargs="+",
        default=["domain-s3"],
        choices=sorted(GUARD_MODES | {"none"}),
        help="Guard modes to evaluate.",
    )
    parser.add_argument(
        "--llm-provider",
        default="deepseek",
        choices=["deepseek", "openai", "anthropic", "ollama"],
        help="LLM provider for generating answers.",
    )
    parser.add_argument("--model", default="deepseek-chat", help="Model name.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible or Ollama base URL.")
    parser.add_argument("--api-key-env", default=None, help="API key environment variable override.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--nemo-config", default="deepseek_config.yml", help="NeMo config under files/02_configs/.")
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--verifier-timeout", type=float, default=8.0)
    parser.add_argument("--disable-ground-truth-http", action="store_true")
    parser.add_argument("--hallucination-extra-responses", type=int, default=2)
    parser.add_argument("--hallucination-retries", type=int, default=2)
    parser.add_argument("--output", default="e2e_results.json")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
