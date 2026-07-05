# Quick Start Guide

Get the Domain Hallucination Guard up and running in 5 minutes.

## Installation

```bash
# Install from local source
pip install -e nemoguardrails/library/domain_hallucination/

# Or use as a module
python -c "from nemoguardrails.library.domain_hallucination import actions"
```

## 1. Basic Usage (Sync Wrapper)

```python
import asyncio
from nemoguardrails.library.domain_hallucination import actions

async def main():
    answer = "Visit https://github.com/pytorch/pytorch for PyTorch"

    result = await actions.analyze_answer(
        answer=answer,
        user_query="How do I use PyTorch?"
    )

    if result["decision"]["action"] != "pass":
        print(f"Action: {result['decision']['action']}")
        print(f"Modified: {result['enforced_answer']['modified_answer']}")

asyncio.run(main())
```

## 2. Using the NeMo Adapter

```python
from domain_hallucination_guard_system.nemo_adapter import DomainHallucinationAdapter
import asyncio

# Initialize
adapter = DomainHallucinationAdapter()

async def check_answer(answer, query):
    result = await adapter.analyze_answer(answer, user_query=query)
    return result["decision"]["action"]

# Use
action = asyncio.run(check_answer(
    "See https://pytorch.org",
    "Learn PyTorch"
))
print(f"Action: {action}")
```

## 3. With Configuration

```python
from nemoguardrails.library.domain_hallucination import config, actions
import asyncio

# Create config
cfg = config.DomainHallucinationGuardConfig(
    verification=config.VerificationConfig(level="dns"),
    scoring=config.ScoringConfig(
        fail_threshold=50.0,  # More strict
        warn_threshold=20.0
    )
)

# Save it
cfg.save("my_config.json")

# Use it
config.set_config(cfg)

result = asyncio.run(actions.analyze_answer("Visit https://fake-domain.xyz"))
print(result["decision"])
```

## 4. With Knowledge Base

```python
from nemoguardrails.library.domain_hallucination import kb, actions
import asyncio

# Initialize KB
kb_instance = kb.KnowledgeBase()

# Add trusted domains
kb_instance.add_trusted_domain("my-company.com")
kb_instance.add_trusted_github_repo("myorg", "myrepo")

# Add blacklist
kb_instance.add_blacklisted_domain("phishing.com", reason="Known phishing")

# Use it
result = asyncio.run(actions.analyze_answer(
    "Check https://my-company.com and https://github.com/myorg/myrepo",
    kb_instance=kb_instance
))

print(f"Has issues: {result['detection']['has_issues']}")
```

## 5. With NeMo Guardrails (Full Integration)

```python
# config/config.yml
version: "0.1"
models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  output:
    flows:
      - guardrails

# config/colang/guardrails.co
flow output_rail
  execute analyze_answer(
    answer=$assistant_output,
    user_query=$user_message,
    verification_level="dns"
  )

  $result = output

  if $result.decision.action == "block"
    reject "Response contains unverified information"
  elif $result.decision.action == "refine"
    override $result.enforced_answer.modified_answer

# app.py
from nemoguardrails import RailsConfig
from nemoguardrails.library.domain_hallucination import actions

config = RailsConfig.from_path("config/")
config.register_action(actions.analyze_answer)

# Use normally
```

## 6. Run Examples

```bash
# Run all examples
python -m nemoguardrails.library.domain_hallucination.examples

# Run specific example
python -c "
from nemoguardrails.library.domain_hallucination.examples import example_basic_detection
import asyncio
asyncio.run(example_basic_detection())
"
```

## 7. Run Tests

```bash
# Run all tests
pytest nemoguardrails/library/domain_hallucination/test_*.py -v

# Run specific test file
pytest nemoguardrails/library/domain_hallucination/test_extractors.py -v

# Run with coverage
pytest --cov=nemoguardrails.library.domain_hallucination nemoguardrails/library/domain_hallucination/test_*.py
```

## Common Use Cases

### Use Case 1: Strict Verification

```python
# High fail threshold = strict enforcement
config = DomainHallucinationGuardConfig(
    verification=VerificationConfig(level="http"),  # HTTP level
    scoring=ScoringConfig(
        fail_threshold=40.0,    # Lower = stricter
        warn_threshold=15.0
    )
)
```

### Use Case 2: Lenient Checking

```python
# Low fail threshold = lenient enforcement
config = DomainHallucinationGuardConfig(
    verification=VerificationConfig(level="dns"),   # DNS level
    scoring=ScoringConfig(
        fail_threshold=80.0,    # Higher = more lenient
        warn_threshold=50.0
    )
)
```

### Use Case 3: Speed Optimized

```python
# Fast-pass everything without links
config = DomainHallucinationGuardConfig(
    detection=DetectionConfig(no_link_fast_pass=True),
    verification=VerificationConfig(level="none")  # No verification
)
```

### Use Case 4: Comprehensive Checking

```python
# All checks enabled
config = DomainHallucinationGuardConfig(
    verification=VerificationConfig(level="full"),
    detection=DetectionConfig(
        enable_semantic_check=True,
        enable_advanced_verification=True
    )
)
```

## Environment Variables

Set these for quick configuration without a file:

```bash
# Verification
export DOMAIN_HALLUCINATION_VERIFICATION_LEVEL=dns
export DOMAIN_HALLUCINATION_GITHUB_TOKEN=ghp_xxxx

# Thresholds
export DOMAIN_HALLUCINATION_FAIL_THRESHOLD=60
export DOMAIN_HALLUCINATION_REFINE_THRESHOLD=40

# Features
export DOMAIN_HALLUCINATION_SEMANTIC_CHECK=false
export DOMAIN_HALLUCINATION_ADVANCED_VERIFICATION=false

# KB
export DOMAIN_HALLUCINATION_SEED_KB_PATH=seed_kb.json
export DOMAIN_HALLUCINATION_EXTERNAL_KB_ROOT=/opt/kb

# Debug
export DOMAIN_HALLUCINATION_DEBUG=true
```

## Troubleshooting

### Import Error

```python
# Make sure library is installed
pip install -e nemoguardrails/library/domain_hallucination/

# Or add to Python path
import sys
sys.path.insert(0, "nemoguardrails/library/domain_hallucination")
```

### Async/Await Issues

```python
# All main functions are async
import asyncio

result = asyncio.run(adapter.analyze_answer("text"))
```

### DNS Timeout

```python
# Increase timeout
config = DomainHallucinationGuardConfig(
    verification=VerificationConfig(dns_timeout=10.0)
)
```

### Too Many Rate Limits

```python
# Add GitHub token
adapter = DomainHallucinationAdapter(github_token="ghp_xxxx")

# Or use "none" verification level
config.verification.level = "none"
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Review [Architecture](ARCHITECTURE.md) to understand the design
3. Check [Integration Guide](INTEGRATION_GUIDE.md) for advanced setups
4. Explore examples in [examples.py](examples.py)
5. Review test cases in `test_*.py` files

## Support

- File issues on GitHub
- Check existing documentation
- Run examples to debug
- Enable debug logging: `config.debug = True`
