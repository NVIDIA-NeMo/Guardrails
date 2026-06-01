# Architecture Design

## Overview

The Domain Hallucination Guard is designed as a modular, composable pipeline for detecting and preventing domain hallucinations in LLM outputs.

## Design Principles

1. **Modularity**: Each component (extraction, verification, scoring, decision) is independent
2. **Composability**: Components can be combined in different ways for different use cases
3. **Extensibility**: Easy to add new verification methods, issue types, scoring rules
4. **Performance**: Async operations, fast-pass optimization, efficient data structures
5. **Configurability**: All thresholds, weights, and behaviors are configurable
6. **Transparency**: All decisions include detailed evidence and reasoning

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  INPUT: LLM Output                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │   EXTRACTION Layer       │
        │  (extractors.py)         │
        │                          │
        │ • URL extraction         │
        │ • Domain extraction      │
        │ • GitHub repo extraction │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │  FAST PASS CHECK         │
        │  (no_links_detected?)    │
        └────────────┬─────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
      [PASS]          ┌─────────────────────┐
                      │ VERIFICATION Layer  │
                      │ (verification.py)   │
                      │                     │
                      │ • DNS resolution    │
                      │ • HTTP check        │
                      │ • GitHub API check  │
                      └────────────┬────────┘
                                   │
                      ┌────────────▼────────────┐
                      │   KB/RAG Layer         │
                      │   (kb.py)              │
                      │                        │
                      │ • Query seed KB        │
                      │ • Query external KB    │
                      │ • Blacklist check      │
                      └────────────┬───────────┘
                                   │
                      ┌────────────▼────────────┐
                      │  DETECTION Layer       │
                      │  (checkers.py)         │
                      │                        │
                      │ • Aggregate issues     │
                      │ • Normalize evidence   │
                      │ • Deduplicate         │
                      │ • Summarize           │
                      └────────────┬───────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
         ┌────────────┐      ┌────────────┐    ┌────────────┐
         │ SCORING    │      │ SEMANTIC   │    │ ADVANCED   │
         │ (scoring)  │      │ (semantic) │    │ (semantic) │
         │            │      │            │    │            │
         │ • Risk     │      │ • Relevance│   │ • Typosquat│
         │   score    │      │   check    │    │ • SSL/HTTPS│
         └────────────┘      └────────────┘    └────────────┘
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │
                      ┌────────────▼──────────┐
                      │  RECALIBRATION Layer  │
                      │  (scoring)            │
                      │                       │
                      │ • Adjust based on     │
                      │   verification       │
                      │ • Apply evidence     │
                      │ • Compute bonus      │
                      └────────────┬──────────┘
                                   │
                      ┌────────────▼──────────┐
                      │  DECISION Layer       │
                      │  (decision.py)        │
                      │                       │
                      │ • Apply policy        │
                      │ • Determine action    │
                      │ • Generate reason     │
                      └────────────┬──────────┘
                                   │
                ┌──────────┬────────┴────────┬──────────┐
                ▼          ▼                 ▼          ▼
             [BLOCK]   [REFINE]          [WARN]    [PASS]
                │          │                │          │
                └──────────┴────────────────┴──────────┘
                           │
                           ▼
            ┌──────────────────────────────────┐
            │  ENFORCEMENT Layer               │
            │  (decision)                      │
            │                                  │
            │ • Modify answer                  │
            │ • Add warnings/notices           │
            │ • Log decision                   │
            └──────────────┬───────────────────┘
                           │
                           ▼
                ┌─────────────────────────┐
                │  OUTPUT: Modified Answer │
                │         + Decision       │
                │         + Evidence       │
                └─────────────────────────┘
```

## Data Flow

### Input Types

```python
# Answer to be analyzed
answer: str = "Check https://github.com/pytorch/pytorch for details"

# Optional user query for semantic checks
user_query: str = "How do I use PyTorch?"
```

### Extraction Output

```python
{
    "urls": [
        {
            "raw": "https://github.com/pytorch/pytorch",
            "normalized": "https://github.com/pytorch/pytorch",
            "host": "github.com",
            "scheme": "https",
            ...
        }
    ],
    "domains": [
        {
            "host": "github.com",
            "domain": "github",
            "suffix": "com",
            "registered_domain": "github.com",
            ...
        }
    ],
    "github_repos": [
        {
            "owner": "pytorch",
            "repo": "pytorch",
            "full_name": "pytorch/pytorch",
            "link_type": "repo",
            ...
        }
    ],
    "no_links": False
}
```

### Verification Output

```python
{
    "dns": [
        {
            "domain": "github.com",
            "resolves": True,
            "addresses": ["192.30.255.112"],
            "status": "resolved",
            "confidence": "high",
            ...
        }
    ],
    "http": [
        {
            "target": "https://github.com/pytorch/pytorch",
            "reachable": True,
            "status_code": 200,
            "status": "http_ok",
            ...
        }
    ],
    "github": [
        {
            "owner": "pytorch",
            "repo": "pytorch",
            "exists": True,
            "stars": 68000,
            "status": "repo_exists",
            ...
        }
    ]
}
```

### Detection Output

```python
{
    "has_issues": False,
    "issues": [],
    "issue_summary": {
        "total": 0,
        "by_type": {},
        "by_severity": {},
        "highest_severity": "none"
    }
}
```

### Scoring Output

```python
{
    "score": 0.0,          # Final risk score (0-100)
    "raw_score": 0.0,      # Before recalibration
    "level": "L0",         # L0-L4
    "label": "Normal",
    "bonus": 0.0,
    "score_details": [     # Per-issue breakdown
        {
            "type": "issue_type",
            "base_score": 50.0,
            "severity_weight": 1.3,
            "confidence_boost": 1.0,
            "weighted_score": 65.0,
            "target": "domain.com"
        }
    ]
}
```

### Decision Output

```python
{
    "action": "pass",      # block, refine, warn, pass
    "reason": "Risk score 15.0 below warn threshold",
    "level": "L0",
    "score": 15.0,
    "threshold_fail": 60.0,
    "threshold_refine": 40.0,
    "threshold_warn": 20.0,
    "verification_level": "dns"
}
```

## Issue Type Hierarchy

```
Domain Hallucination Issues
├── Verification Failures
│   ├── non_existent_domain (DNS fails)
│   │   └── Type: Critical
│   │   └── Base Score: 80
│   ├── delegated_no_address_record (DNS record missing)
│   │   └── Type: Medium
│   │   └── Base Score: 50
│   └── fake_github_repo (GitHub 404)
│       └── Type: Critical
│       └── Base Score: 85
├── Blacklist/Reputation
│   └── blacklisted_domain (Known malicious)
│       └── Type: Critical
│       └── Base Score: 95
├── Domain Characteristics
│   └── recent_domain (Registered recently)
│       └── Type: Low
│       └── Base Score: 20
├── Evidence Gaps
│   └── no_local_kb_evidence (Not in KB)
│       └── Type: Low
│       └── Base Score: 15
├── Semantic Checks
│   └── semantic_mismatch (Not relevant to query)
│       └── Type: Low
│       └── Base Score: 30
└── Advanced Verification
    ├── possible_typosquatting (Similar to known repo)
    │   └── Type: Low
    │   └── Base Score: 25
    └── insecure_protocol (HTTP not HTTPS)
        └── Type: Low
        └── Base Score: 10
```

## Verification Strategy

### DNS Level (Fast)
- Check domain resolution
- Identify non-existent domains
- Detect DNS failures
- Latency: ~100-500ms per domain

### HTTP Level (Medium)
- All DNS checks
- HTTP accessibility check
- Follow redirects
- Validate SSL/TLS
- Latency: ~1-3s per URL

### Full Level (Slow)
- All HTTP checks
- Advanced verification
- Semantic checking
- Typosquatting detection
- External KB queries
- Latency: ~3-10s

### None Level (Fastest)
- Skip all verification
- Use KB only
- Use extraction + aggregation
- Latency: ~10-50ms

## Scoring Algorithm

```
score = min(100, sum([
    issue_i.base_score 
    * severity_weight[issue_i.severity]
    * confidence_boost[issue_i.confidence]
    for each issue_i
]) + bonus)

bonus = sum([
    critical_count * 10,
    (high_count >= 3) * 5,
    (multiple_issue_types_coexist) * 5
])

recalibrated_score = score - sum([
    dns_success_count * 10,
    http_success_count * 15,
    github_success_count * 20,
    kb_evidence_count * 2
])

final_score = max(0, min(100, recalibrated_score))
```

## Decision Policy

```
if final_score >= fail_threshold (60):
    action = "block"
elif final_score >= refine_threshold (40):
    action = "refine"
elif final_score >= warn_threshold (20):
    action = "warn"
else:
    action = "pass"

if verification_level == "none":
    downgrade action by 1 level
```

## Configuration Hierarchy

```
Default Config
    ↓
Config File (JSON)
    ↓
Environment Variables
    ↓
Runtime Override
```

## Extension Points

### 1. Custom Issue Types

Add to `ISSUE_TYPE_SCORES` in `scoring.py`:
```python
ISSUE_TYPE_SCORES["custom_issue"] = 50.0
```

### 2. Custom Verification Methods

Create new function in `verification.py`:
```python
def check_custom_verification(item) -> Dict[str, Any]:
    ...
```

### 3. Custom KB Sources

Extend `KnowledgeBase` class:
```python
class CustomKB(KnowledgeBase):
    def query_custom_source(self, domain):
        ...
```

### 4. Custom Scoring Logic

Create wrapper around `calculate_risk_score`:
```python
def custom_scoring(detection, custom_weights):
    ...
```

## Performance Optimization

### Caching Strategies
- Cache DNS lookups (TTL: 1 hour)
- Cache HTTP checks (TTL: 24 hours)
- Cache GitHub API calls (TTL: 24 hours)
- Cache KB queries (TTL: session)

### Concurrency
- Parallel DNS/HTTP/GitHub checks (async/await)
- Batch verification for multiple entities
- Request pooling for HTTP

### Early Exit
- Fast pass for no-links answers (skip all)
- Skip HTTP if DNS fails
- Skip advanced checks if score already over threshold

## Security Considerations

1. **Injection Protection**
   - URL parsing uses robust regex
   - No shell execution
   - Input validation on all paths

2. **Rate Limiting**
   - GitHub API token support
   - Timeout protection
   - DNS query limits

3. **Privacy**
   - No logging of user content
   - Optional debug mode
   - Configurable log levels

4. **Credential Handling**
   - GitHub token via environment/config
   - No hardcoded secrets
   - Secure token storage

## Testing Strategy

1. **Unit Tests**: Individual components
2. **Integration Tests**: Component combinations
3. **Performance Tests**: Latency benchmarks
4. **Real-world Tests**: Common LLM outputs
5. **Regression Tests**: Known hallucinations

## Monitoring & Telemetry

- Action distribution (block/refine/warn/pass)
- Risk score distribution
- Verification latency
- False positive/negative rates
- KB coverage stats
