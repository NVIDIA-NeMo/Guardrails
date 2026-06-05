# Domain Hallucination Guard Library

Domain Hallucination Guard detects unverifiable URLs, domains, and GitHub
repositories in model-generated answers for NeMo Guardrails. It extracts
external references, verifies them at a configurable depth, scores risk, and
returns an enforcement decision that can pass, warn, refine, or block an answer.

## Features

- Extracts URLs, bare domains, and GitHub repository references from text.
- Supports verification levels: `none`, `dns`, `http`, and `full`.
- Performs DNS, HTTP, TLS, WHOIS/RDAP, and GitHub repository checks.
- Blocks private and non-public IP literals before HTTP probing to reduce SSRF risk.
- Supports local seed KB data plus path-safe external KB files.
- Scores issues by type, severity, and confidence, then recalibrates with evidence.
- Provides optional semantic relevance and advanced typosquatting checks.
- Applies configurable enforcement messages for block, refine, warn, and pass outcomes.

## Architecture

```text
NeMo Guardrails flow
        |
        v
self_check_domain_hallucination / analyze_answer
        |
        +-- extract entities: URLs, domains, GitHub repos
        |
        +-- verify evidence according to verification_level
        |      none: no network verification
        |      dns: DNS only
        |      http: DNS + HTTP
        |      full: DNS + HTTP + TLS + WHOIS/RDAP + GitHub API
        |
        +-- query KB evidence
        |
        +-- aggregate issues
        |
        +-- calculate and recalibrate risk score
        |
        +-- optional semantic and advanced checks
        |
        v
make decision -> apply enforcement -> return modified answer metadata
```

## Installation

```bash
python -m pip install -e .
```

The module uses Python standard-library networking for DNS, HTTP, TLS,
WHOIS/RDAP, and GitHub API calls. No additional runtime dependency is required
for the domain hallucination module itself.

## Quick Start

### Direct Usage

```python
import asyncio

from nemoguardrails.library.domain_hallucination import actions


async def main():
    result = await actions.analyze_answer(
        answer="See https://github.com/pytorch/pytorch for details.",
        user_query="How do I use PyTorch?",
        verification_level="dns",
        enable_semantic_check=False,
        enable_advanced_verification=False,
    )

    print(result["status"])
    print(result["decision"]["action"])
    print(result["enforced_answer"]["modified_answer"])


asyncio.run(main())
```

### Rail Action

The module exposes `self_check_domain_hallucination`, which is registered as a
NeMo Guardrails action.

```co
flow self check domain hallucination
  $domain_hallucination = await SelfCheckDomainHallucinationAction()

  if $domain_hallucination["decision"]["action"] == "block"
    bot say $domain_hallucination["enforced_answer"]["modified_answer"]
    stop
```

## Verification Levels

| Level | Checks | Notes |
| --- | --- | --- |
| `none` | No network checks | Fastest mode. High-risk block/refine decisions are downgraded to warn. |
| `dns` | DNS resolution | Default balance for most uses. |
| `http` | DNS + HTTP reachability | HTTP requests block private, loopback, link-local, and reserved IP literals. |
| `full` | DNS + HTTP + TLS + WHOIS/RDAP + GitHub API | Strictest mode. GitHub API verification only runs at this level. |

Invalid verification levels raise `ValueError` instead of silently skipping
verification.

## Configuration

Create a JSON config file:

```json
{
  "verification": {
    "level": "dns",
    "dns_timeout": 4.0,
    "http_timeout": 6.0,
    "tls_timeout": 5.0,
    "whois_timeout": 6.0,
    "github_timeout": 6.0,
    "github_token": "ghp_xxxxx"
  },
  "scoring": {
    "fail_threshold": 60.0,
    "refine_threshold": 40.0,
    "warn_threshold": 20.0
  },
  "detection": {
    "enable_semantic_check": false,
    "enable_advanced_verification": false,
    "no_link_fast_pass": true
  },
  "kb": {
    "seed_kb_path": "seed_kb.json",
    "external_kb_root": "./kb"
  }
}
```

Load it with:

```python
from nemoguardrails.library.domain_hallucination import config as dh_config

loaded_config = dh_config.load_config("config.json")
```

## Environment Variables

All environment variables use the `DOMAIN_HALLUCINATION_` prefix.

| Variable | Purpose |
| --- | --- |
| `VERIFICATION_LEVEL` | One of `none`, `dns`, `http`, `full`. |
| `DNS_TIMEOUT` | DNS timeout in seconds. |
| `HTTP_TIMEOUT` | HTTP timeout in seconds. |
| `TLS_TIMEOUT` | TLS timeout in seconds. |
| `WHOIS_TIMEOUT` | WHOIS/RDAP timeout in seconds. |
| `GITHUB_TIMEOUT` | GitHub API timeout in seconds. |
| `GITHUB_TOKEN` | Optional GitHub token for higher API limits. |
| `SEMANTIC_CHECK` | `true` or `false`. |
| `ADVANCED_VERIFICATION` | `true` or `false`. |
| `FAIL_THRESHOLD` | Score threshold for block. |
| `REFINE_THRESHOLD` | Score threshold for refine. |
| `WARN_THRESHOLD` | Score threshold for warn. |
| `SEED_KB_PATH` | Path to seed KB JSON. |
| `EXTERNAL_KB_ROOT` | Root directory for external KB files. |
| `DEBUG` | `true` or `false`. |

Example:

```bash
export DOMAIN_HALLUCINATION_VERIFICATION_LEVEL=full
export DOMAIN_HALLUCINATION_GITHUB_TIMEOUT=8
```

## Knowledge Base

### Seed KB

```json
{
  "trusted_domains": [
    {"domain": "github.com", "category": "vcs"},
    "pytorch.org"
  ],
  "trusted_github_repos": [
    {"owner": "pytorch", "repo": "pytorch"},
    "tensorflow/tensorflow"
  ],
  "blacklisted_domains": [
    {"domain": "phishing.example", "reason": "Known phishing site"}
  ]
}
```

### External KB

External KB files are looked up under `external_kb_root/domains`. Domain names
are accepted only when they match `[a-z0-9._-]+`, and resolved paths must remain
under the configured KB root.

```text
kb_root/
  domains/
    github.com.json
    pytorch.org.json
    *.example.com.json
```

## Risk Scoring

Scoring combines:

1. Issue type base score.
2. Severity weight.
3. Confidence multiplier.
4. Bonus for critical or combined issue patterns.
5. Recalibration from successful DNS, HTTP, GitHub, and KB evidence.

| Score | Level | Default Action |
| --- | --- | --- |
| 0-19 | L0 Normal | pass |
| 20-39 | L1 Low | warn |
| 40-59 | L2 Medium | refine |
| 60-79 | L3 High | block |
| 80-100 | L4 Critical | block |

## Module Overview

| Module | Purpose |
| --- | --- |
| `extractors.py` | Extract URLs, domains, and GitHub repositories. |
| `verification.py` | DNS, HTTP, TLS, WHOIS/RDAP, and GitHub checks. |
| `checkers.py` | Convert verification and KB evidence into issues. |
| `scoring.py` | Calculate and recalibrate risk. |
| `decision.py` | Select enforcement action and modify answers. |
| `kb.py` | Manage trusted, blacklisted, seed, and external KB data. |
| `semantic.py` | Optional relevance and typosquatting checks. |
| `schemas.py` | Lightweight result schema helpers. |
| `config.py` | JSON and environment-based configuration. |
| `utils.py` | Convenience helpers for direct analysis. |

## Security Notes

- HTTP verification blocks private, loopback, link-local, and reserved IP literals before `urlopen`.
- External KB lookup validates domain names and confirms resolved file paths remain under the KB root.
- GitHub API checks only run at `verification_level="full"`.
- Blocking verification work is delegated to an executor from async actions to avoid event-loop stalls.

## Testing

```bash
python -m pytest tests/library/domain_hallucination -q
python -m pytest tests/library/domain_hallucination --cov=nemoguardrails.library.domain_hallucination --cov-report=term-missing -q
```

## License

SPDX-License-Identifier: Apache-2.0
