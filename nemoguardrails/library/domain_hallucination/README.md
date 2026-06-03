# Domain Hallucination Guard Library

A comprehensive library for detecting and preventing domain hallucinations in LLM outputs within the NeMo Guardrails framework.

## Features

- **Entity Extraction**: Automatically extracts URLs, domains, and GitHub repositories from text
- **Multi-level Verification**: DNS, HTTP, TLS, WHOIS/RDAP, and GitHub API verification
- **Knowledge Base Integration**: Local seed KB + external KB support
- **Risk Scoring**: Sophisticated risk scoring with issue aggregation and recalibration
- **Semantic Analysis**: Optional semantic relevance checking
- **Advanced Verification**: Typosquatting detection, HTTPS validation
- **Flexible Enforcement**: Block, refine, warn, or pass decisions
- **Configurable Thresholds**: Adjustable scoring and decision thresholds

## Architecture

```
┌─────────────────────────────────────────────────┐
│         NeMo Guardrails Flow (flows.co)         │
└─────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  analyze_answer() Action      │
        │  (actions.py)                 │
        └───────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    ┌─────────┐  ┌─────────┐  ┌──────────────┐
    │Extract  │  │Verify   │  │Query KB      │
    │Entities │  │Evidence │  │(RAG)         │
    └─────────┘  └─────────┘  └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ Aggregate Issues (checkers)   │
        └───────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    ┌─────────┐  ┌──────────┐  ┌──────────────┐
    │Score    │  │Recalibr. │  │Semantic/Adv. │
    │Risk     │  │           │  │Verification  │
    └─────────┘  └──────────┘  └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ Make Decision (decision.py)   │
        └───────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
    ENFORCE DECISION            MODIFIED ANSWER
    (block/refine/warn/pass)
```

## Installation

```bash
# Assuming the library is in nemoguardrails/library/domain_hallucination/
python -m pip install -e .
```

## Quick Start

### 1. Basic Usage with NeMo Guardrails

```python
from nemoguardrails import RailsConfig
from domain_hallucination_guard_system.nemo_adapter import get_adapter

# Initialize adapter
adapter = get_adapter(
    seed_kb_path="nemoguardrails/library/domain_hallucination/seed_kb.json"
)

# In your NeMo rails config
config = RailsConfig.from_path("config_folder")

# Register the action in rails
config.actions = [adapter.analyze_answer]
```

### 2. Direct Usage

```python
import asyncio
from domain_hallucination_guard_system.nemo_adapter import DomainHallucinationAdapter

async def main():
    adapter = DomainHallucinationAdapter(
        verification_level="dns",
        enable_semantic_check=False,
        enable_advanced_verification=False
    )
    
    answer = "You can find more info at https://github.com/pytorch/pytorch"
    query = "How do I use PyTorch?"
    
    result = await adapter.analyze_answer(answer, user_query=query)
    
    print(f"Status: {result['status']}")
    print(f"Decision: {result['decision']['action']}")
    if result['decision']['action'] != 'pass':
        print(f"Modified Answer: {result['enforced_answer']['modified_answer']}")

asyncio.run(main())
```

### 3. Configuration

Create a config file `config.json`:

```json
{
  "verification": {
    "level": "dns",
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

Use the config:

```python
from domain_hallucination_guard_system.nemo_adapter import DomainHallucinationAdapter

adapter = DomainHallucinationAdapter(config_path="config.json")
```

## Modules Overview

### extractors.py
Extracts URLs, domains, and GitHub repositories from text with robust parsing and normalization.

**Key Functions:**
- `extract_urls(text)` - Extract and normalize URLs
- `extract_domains(text)` - Extract domain names
- `extract_github_repos(urls)` - Parse GitHub repository URLs
- `extract_all(text)` - Extract all entity types

### verification.py
Performs DNS, HTTP, TLS certificate, WHOIS/RDAP, and GitHub API verification.

**Key Functions:**
- `resolve_domain(domain)` - DNS resolution
- `check_http_domain(url)` - HTTP accessibility check
- `check_tls(domain)` - TLS certificate verification
- `check_whois(domain)` - WHOIS/RDAP registration metadata lookup
- `check_github_repo(repo_item)` - GitHub API verification

### checkers.py
Aggregates verification results into normalized issue types.

**Key Functions:**
- `check_domain_hallucination(extracted, verification, rag)` - Main checking logic
- `_check_dns_failures()` - Detect non-existent domains
- `_check_tls_failures()` - Detect TLS certificate problems
- `_check_github_repos()` - Detect fake GitHub repos
- `_check_phishing_domains()` - Detect suspicious domains

### scoring.py
Calculates risk scores and recalibrates based on verification evidence.

**Key Functions:**
- `calculate_risk_score(detection_result)` - Initial scoring
- `recalibrate_score(risk_score, verification_results)` - Evidence-based adjustment

### decision.py
Makes enforcement decisions based on risk scores and policies.

**Key Functions:**
- `make_decision(risk_score, policy, verification_level)` - Determine action
- `apply_decision(decision, answer)` - Modify answer based on action

### kb.py
Manages local seed knowledge base and external KB integration.

**Key Classes:**
- `KnowledgeBase` - In-memory KB for trusted/blacklisted domains and repos
- `initialize_kb()` - Load seed KB and set external root

### semantic.py
Optional semantic relevance and advanced verification checks.

**Key Functions:**
- `check_semantic_relevance()` - Check if mentioned domains relate to query
- `check_advanced_verification()` - Typosquatting, HTTPS, etc.

### schemas.py
Data structures for issues, detection results, and scores.

**Key Classes:**
- `Issue` - Represents a domain hallucination issue
- `DetectionResult` - Aggregated detection results
- `RiskScore` - Scoring result
- `Decision` - Enforcement decision

### config.py
Configuration management with JSON serialization.

**Key Classes:**
- `DomainHallucinationGuardConfig` - Main config object
- Environment variable support via `from_env()`

## Verification Levels

- **none**: No verification (fast, but no checking)
- **dns**: DNS resolution only (default, good balance)
- **http**: DNS + HTTP accessibility check
- **full**: DNS + HTTP + TLS + WHOIS/RDAP + GitHub checks

## Risk Scoring

Scores are calculated by:
1. **Base Score**: Issue type-specific base scores (0-100)
2. **Severity Weight**: 1.5× (critical), 1.3× (high), 1.0× (medium), 0.7× (low)
3. **Confidence Boost**: 1.0× (high), 0.8× (medium), 0.6× (low)
4. **Bonus**: Additional points for multiple critical issues
5. **Recalibration**: Adjusted down based on successful verification

### Risk Levels

- **L0 (Normal)**: Score 0-19 → Pass
- **L1 (Low)**: Score 20-39 → Warn
- **L2 (Medium)**: Score 40-59 → Refine
- **L3 (High)**: Score 60-79 → Block
- **L4 (Critical)**: Score 80+ → Block

## Knowledge Base Format

### Seed KB (JSON)

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
    {"domain": "phishing.com", "reason": "Known phishing site"}
  ]
}
```

### External KB Structure

```
kb_root/
├── domains/
│   ├── github.com.json
│   ├── pytorch.org.json
│   └── *.example.com.json
├── repos/
│   └── pytorch_pytorch.json
└── blacklist.json
```

## Environment Variables

- `DOMAIN_HALLUCINATION_VERIFICATION_LEVEL`: dns, http, full
- `DOMAIN_HALLUCINATION_FAIL_THRESHOLD`: 60.0
- `DOMAIN_HALLUCINATION_SEMANTIC_CHECK`: true/false
- `DOMAIN_HALLUCINATION_GITHUB_TOKEN`: GitHub API token
- `DOMAIN_HALLUCINATION_SEED_KB_PATH`: Path to seed KB
- `DOMAIN_HALLUCINATION_EXTERNAL_KB_ROOT`: External KB root dir
- `DOMAIN_HALLUCINATION_DEBUG`: true/false

## Performance Considerations

1. **Fast Pass**: Enable `no_link_fast_pass` to skip checking when no links are detected
2. **Verification Level**: 
   - Use "dns" for most cases (good balance)
   - Use "none" for maximum speed
   - Use "http" or "full" only when strict verification needed
3. **Caching**: Consider caching verification results to avoid repeated checks
4. **Async**: All verification is async to allow concurrent checking

## Extending the Library

### Add Custom Issue Types

Edit `scoring.py`:

```python
ISSUE_TYPE_SCORES = {
    "my_custom_issue": 50.0,
    # ...
}
```

Then in `checkers.py`, add your custom checking function.

### Add Custom Verification Methods

Create a new function in `verification.py`:

```python
def check_custom_verification(item: Dict[str, Any]) -> Dict[str, Any]:
    """Custom verification logic."""
    return {
        "source": "custom",
        "status": "verified",
        "confidence": "high",
    }
```

Call it in `actions.py`:

```python
custom_results = [check_custom_verification(item) for item in items]
verification_results["custom"] = custom_results
```

## Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=nemoguardrails/library/domain_hallucination tests/
```

## API Reference

See docstrings in individual modules for detailed API documentation.

## License

SPDX-License-Identifier: Apache-2.0

## Contributing

Contributions welcome! Please follow the existing code style and add tests for new features.
