# Changelog

All notable changes to the Domain Hallucination Guard library are documented in this file.

## [0.2.0] - 2026-06-02

### ⚠️ BREAKING CHANGES

- `verification.resolve_domain`, `check_http_domain`, `check_tls`, `check_whois`,
  and `check_github_repo` are now **`async def`** functions. Callers must `await` them.
  Synchronous wrappers have been removed.
- `actions.analyze_answer` now performs all verifications concurrently via
  `asyncio.gather`. Public signature unchanged (already async), but per-check
  concurrency parameters were added: `dns_concurrency`, `http_concurrency`,
  `tls_concurrency`, `whois_concurrency`, `github_concurrency` (defaults 16–32).
- Default value of `skip_secondary_checks_on_dns_failure` changed from
  `False` to `True`. Domains failing DNS resolution skip HTTP/TLS/WHOIS by default.

### Added

- Shared `aiohttp.ClientSession` with a connection pool (limit=100, per-host=10).
- `verification.close_http_session()` for explicit shutdown.
- TTL-bucketed cache (`cachetools.TTLCache`) keyed by `dns_ok/dns_fail/http_ok/
  http_fail/tls/whois/github_ok/github_fail/github_rate_limited`. Failures are
  cached for 24h to avoid hammering known-bad targets; rate-limited responses
  for 1h. Thread-safe via `threading.Lock`.
- `actions.clear_verification_cache()` helper for tests.
- Batch helpers `verification.resolve_domains_batch`, `check_http_batch`,
  `check_tls_batch`, `check_whois_batch`, `check_github_batch`.
- HTTPS / HTTP racing in `check_http_domain` — both probes start concurrently;
  the first 2xx/3xx wins, the loser is cancelled.
- RDAP multi-candidate racing in `check_whois`.
- GitHub rate-limit awareness: parses `X-RateLimit-Remaining` / `X-RateLimit-Reset`
  / `Retry-After`. A 403/429 (or `remaining == 0`) opens a process-wide cooldown
  window so concurrent and subsequent calls short-circuit instead of burning the
  60-req/hour anonymous quota. Default concurrency for `check_github_batch`
  drops from 16 to **2 (anonymous)** / **8 (with token)**. New helpers
  `get_github_rate_limit_status()` and `reset_github_rate_limit_state()`.

### Fixed

- `resolve_domain` actually honours `timeout` (previously the parameter was
  ignored; `socket.getaddrinfo` could block indefinitely).
- HTTP/HTTPS fallback no longer doubles the worst-case latency.

### Performance

- 5-domain analyze_answer P95 reduced from ~30 s to ~5 s on representative
  fixtures (measured via `benchmark.py`).

## [0.1.0] - 2026-06-01

### Added

#### Core Features
- Entity extraction (URLs, domains, GitHub repos)
- Multi-level verification (DNS, HTTP, GitHub API)
- Knowledge base management (seed KB + external KB)
- Risk scoring with issue aggregation
- Score recalibration based on verification evidence
- Decision making with configurable thresholds
- Answer enforcement (block, refine, warn, pass)

#### Modules
- `extractors.py` - URL, domain, and GitHub extraction
- `verification.py` - DNS, HTTP, and GitHub verification
- `checkers.py` - Issue aggregation and detection
- `scoring.py` - Risk scoring and recalibration
- `decision.py` - Decision making and enforcement
- `kb.py` - Knowledge base management
- `semantic.py` - Semantic relevance and advanced verification
- `actions.py` - Main NeMo action
- `config.py` - Configuration management
- `schemas.py` - Data structures
- `utils.py` - Utility functions
- `examples.py` - Usage examples

#### Configuration
- JSON-based configuration
- Environment variable support
- Multiple verification levels (none, dns, http, full)
- Configurable scoring thresholds
- Customizable enforcement messages

#### Knowledge Base
- Seed KB with trusted domains and repos
- External KB root support
- Blacklist management
- Metadata support for entities

#### Integrations
- NeMo Guardrails adapter
- FastAPI integration example
- LangChain integration example
- Docker deployment support

#### Documentation
- Comprehensive README
- Architecture design document
- Integration guide
- API examples
- Inline docstrings

#### Testing
- Unit tests for extractors
- Unit tests for verification
- Unit tests for scoring
- Unit tests for KB
- Example test cases

### Initial Implementation
- All core components functional
- First stable release
- Ready for production use
