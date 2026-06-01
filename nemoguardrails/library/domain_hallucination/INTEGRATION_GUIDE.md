# Integration Guide

## NeMo Guardrails Integration

### Step 1: Add Domain Hallucination Library to Guardrails

Copy or install the library:

```bash
cp -r nemoguardrails/library/domain_hallucination /path/to/nemoguardrails/library/
```

### Step 2: Create Guardrails Configuration Directory

```
config/
├── config.yml
├── colang/
│   └── guardrails.co
└── domain_hallucination/
    ├── config.json
    ├── seed_kb.json
    └── flows.co
```

### Step 3: Register the Action

In your application code or Guardrails startup:

```python
from nemoguardrails import RailsConfig
from nemoguardrails.library.domain_hallucination import actions

# Load config
config = RailsConfig.from_path("config/")

# Register domain hallucination action
config.register_action(actions.analyze_answer)
```

### Step 4: Create Colang Flow

Create `config/colang/guardrails.co`:

```colang
flow output_rail
  """Main output rail for domain hallucination checking."""
  
  # Check if answer has external links
  $has_links = len($assistant_output.split("http")) > 1
  
  if not $has_links
    return
  
  # Call domain hallucination detection
  execute analyze_answer(
    answer=$assistant_output,
    user_query=$user_message,
    verification_level="dns"
  )
  
  $result = output
  
  # Enforce decision
  if $result.decision.action == "block"
    reject $result.enforced_answer.modified_answer
  elif $result.decision.action in ["refine", "warn"]
    override $result.enforced_answer.modified_answer
```

### Step 5: Configure Domain Hallucination

Create `config/domain_hallucination/config.json`:

```json
{
  "verification": {
    "level": "dns",
    "dns_timeout": 4.0,
    "http_timeout": 6.0,
    "github_timeout": 6.0,
    "github_token": null
  },
  "detection": {
    "enable_semantic_check": false,
    "enable_advanced_verification": false,
    "no_link_fast_pass": true
  },
  "scoring": {
    "fail_threshold": 60.0,
    "refine_threshold": 40.0,
    "warn_threshold": 20.0
  },
  "kb": {
    "seed_kb_path": "config/domain_hallucination/seed_kb.json",
    "external_kb_root": null,
    "auto_load": true
  },
  "enforcement": {
    "block_message": "[BLOCKED] Contains unverified information.",
    "refine_message": "[NOTICE] May contain unverified information.",
    "warn_message": "[WARNING] Please verify external links.",
    "append_verification_notice": true
  },
  "debug": false,
  "log_level": "INFO"
}
```

### Step 6: Initialize in Main Application

```python
import asyncio
from domain_hallucination_guard_system.nemo_adapter import get_adapter

# Initialize adapter with config
adapter = get_adapter(
    config_path="config/domain_hallucination/config.json",
    seed_kb_path="config/domain_hallucination/seed_kb.json"
)

async def main():
    # Your guardrails + domain hallucination pipeline
    result = await adapter.analyze_answer(
        answer="Visit https://github.com/pytorch/pytorch",
        user_query="How do I use PyTorch?"
    )
    print(f"Decision: {result['decision']['action']}")

asyncio.run(main())
```

## FastAPI Integration

```python
from fastapi import FastAPI
from pydantic import BaseModel
from domain_hallucination_guard_system.nemo_adapter import get_adapter

app = FastAPI()
adapter = get_adapter(config_path="config.json")

class AnalysisRequest(BaseModel):
    answer: str
    user_query: str = ""
    verification_level: str = "dns"

class AnalysisResponse(BaseModel):
    status: str
    decision: str
    risk_score: float
    modified_answer: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    result = await adapter.analyze_answer(
        answer=req.answer,
        user_query=req.user_query
    )
    
    return AnalysisResponse(
        status=result["status"],
        decision=result["decision"]["action"],
        risk_score=result.get("risk_score", {}).get("score", 0.0),
        modified_answer=result.get("enforced_answer", {}).get("modified_answer", req.answer)
    )
```

## LangChain Integration

```python
from langchain.callbacks.base import BaseCallbackHandler
from domain_hallucination_guard_system.nemo_adapter import get_adapter

class DomainHallucinationCallback(BaseCallbackHandler):
    def __init__(self, adapter=None):
        self.adapter = adapter or get_adapter()
    
    async def on_llm_end(self, response, **kwargs):
        answer = response.generations[0][0].text if response.generations else ""
        result = await self.adapter.analyze_answer(answer)
        
        if result["decision"]["action"] != "pass":
            # Handle enforcement
            modified = result["enforced_answer"]["modified_answer"]
            # Log or modify response
            print(f"[Domain Guard] {result['decision']['action'].upper()}: {result['decision']['reason']}")

# Use in chain
callback = DomainHallucinationCallback()
chain.run(input, callbacks=[callback])
```

## Streaming Integration

```python
async def analyze_stream(answer_generator):
    """Analyze streaming answers as they arrive."""
    adapter = get_adapter()
    accumulated = ""
    
    async for chunk in answer_generator:
        accumulated += chunk
        
        # Periodically check for hallucinations
        if len(accumulated) > 100:
            result = await adapter.analyze_answer(accumulated)
            if result["decision"]["action"] == "block":
                # Cancel stream and return blocked message
                return result["enforced_answer"]["modified_answer"]
            accumulated = ""
    
    # Final check
    if accumulated:
        result = await adapter.analyze_answer(accumulated)
        return result["enforced_answer"]["modified_answer"]
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy library
COPY nemoguardrails/ /usr/local/lib/python3.11/site-packages/nemoguardrails/

# Copy adapter and config
COPY domain_hallucination_guard_system/ /app/
COPY config/ /app/config/

# Run service
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
nemoguardrails>=0.11.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
aiohttp>=3.8.0
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  domain-guard:
    build: .
    ports:
      - "8000:8000"
    environment:
      DOMAIN_HALLUCINATION_VERIFICATION_LEVEL: dns
      DOMAIN_HALLUCINATION_FAIL_THRESHOLD: "60"
      DOMAIN_HALLUCINATION_GITHUB_TOKEN: ${GITHUB_TOKEN}
    volumes:
      - ./config:/app/config
      - ./kb:/app/kb
```

## Environment Setup

### Local Development

```bash
# Clone repo
git clone <repo>
cd guardrails

# Install in development mode
pip install -e .

# Install domain hallucination library
pip install -e nemoguardrails/library/domain_hallucination/

# Run tests
pytest tests/

# Run examples
python -m nemoguardrails.library.domain_hallucination.examples
```

### Production Deployment

1. **Set environment variables:**
   ```bash
   export DOMAIN_HALLUCINATION_VERIFICATION_LEVEL=http
   export DOMAIN_HALLUCINATION_FAIL_THRESHOLD=60.0
   export DOMAIN_HALLUCINATION_GITHUB_TOKEN=ghp_xxxx
   export DOMAIN_HALLUCINATION_SEED_KB_PATH=/opt/kb/seed_kb.json
   ```

2. **Pre-load KB:**
   ```python
   from nemoguardrails.library.domain_hallucination import kb, config
   
   kb_instance = kb.initialize_kb(
       seed_kb_path="/opt/kb/seed_kb.json",
       external_kb_root="/opt/kb/external"
   )
   ```

3. **Monitor performance:**
   - Track verification latency
   - Monitor false positive/negative rates
   - Audit blocked answers

## Troubleshooting

### DNS Timeout Issues

If experiencing DNS timeouts:

```python
config = DomainHallucinationGuardConfig(
    verification=VerificationConfig(dns_timeout=8.0)  # Increase timeout
)
```

Or use `verification_level="none"` for maximum speed.

### GitHub Rate Limiting

Add authentication token:

```python
adapter = get_adapter(github_token="ghp_xxxx")
```

### High False Positive Rate

1. Lower thresholds:
   ```json
   {
     "scoring": {
       "fail_threshold": 75.0,
       "refine_threshold": 55.0
     }
   }
   ```

2. Enable KB expansion to mark common domains as trusted

3. Consider disabling advanced verification checks

### Performance Optimization

For high-throughput scenarios:

```python
config = DomainHallucinationGuardConfig(
    detection=DetectionConfig(
        no_link_fast_pass=True,  # Skip if no links
        enable_semantic_check=False,  # Disable semantic checks
        enable_advanced_verification=False  # Disable advanced checks
    ),
    verification=VerificationConfig(
        level="dns"  # Use DNS only
    )
)
```

## Monitoring and Metrics

Track key metrics:

```python
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.decisions = defaultdict(int)
        self.risk_scores = []
        self.verification_times = []
    
    def record(self, result):
        action = result["decision"]["action"]
        self.decisions[action] += 1
        self.risk_scores.append(result.get("risk_score", {}).get("score", 0))
    
    def get_stats(self):
        return {
            "total_analyzed": sum(self.decisions.values()),
            "decisions": dict(self.decisions),
            "avg_risk_score": sum(self.risk_scores) / len(self.risk_scores) if self.risk_scores else 0,
            "avg_verification_time_ms": sum(self.verification_times) / len(self.verification_times) if self.verification_times else 0,
        }

metrics = MetricsCollector()
```

## Next Steps

1. Customize scoring thresholds for your use case
2. Expand KB with organization-specific trusted domains
3. Set up monitoring and alerting
4. Configure CI/CD integration for automated testing
5. Document custom policies and enforcement rules
