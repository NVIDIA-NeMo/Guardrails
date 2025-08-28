# GuardrailsAI Integration Example

This example demonstrates how to use GuardrailsAI validators with NeMo Guardrails for comprehensive input and output validation.

## Overview

The configuration showcases multiple GuardrailsAI validators working together to provide:

- **PII Detection**: Prevents personally identifiable information in inputs
- **Competitor Checking**: Blocks mentions of competitor companies
- **Topic Restriction**: Ensures outputs stay within allowed topics
- **Toxic Language Detection**: Filters harmful or inappropriate content

## Setup

1. **Install GuardrailsAI**:

   ```bash
   pip install guardrails-ai
   ```

2. **Install required validators**:

   ```bash
   guardrails hub install hub://guardrails/guardrails_pii
   guardrails hub install hub://guardrails/competitor_check
   guardrails hub install hub://tryolabs/restricttotopic
   ```

## Configuration Explanation

### Validator Definitions

The `config.yml` defines four validators under `rails.config.guardrails_ai.validators`:

```yaml

- name: guardrails_pii
  parameters:
    entities: ["phone_number", "email", "ssn"]  # PII types to detect
  metadata: {}

- name: competitor_check
  parameters:
    competitors: ["Apple", "Google", "Microsoft"]  # Competitor names
  metadata: {}

- name: restricttotopic
  parameters:
    valid_topics: ["technology", "science", "education"]  # Allowed topics
  metadata: {}
```

### Rail Configuration

**Input Rails** (check user messages):

```yaml
input:
  flows:
    - guardrailsai check input $validator="guardrails_pii"     # Block PII
    - guardrailsai check input $validator="competitor_check"   # Block competitors
```

**Output Rails** (check bot responses):

```yaml
output:
  flows:
    - guardrailsai check output $validator="restricttotopic"   # Ensure on-topic
```

## Running the Example

### Using Python API

```python
from nemoguardrails import RailsConfig, LLMRails

# Load the configuration
config = RailsConfig.from_path(".")
rails = LLMRails(config)

# Test input validation (should be blocked - contains email)
response = rails.generate(messages=[{
    "role": "user",
    "content": "My email is john.doe@example.com, can you help me?"
}])
print(response)  # Should refuse to respond

# Test competitor mention (should be blocked)
response = rails.generate(messages=[{
    "role": "user",
    "content": "What do you think about Apple's latest iPhone?"
}])
print(response)  # Should refuse to respond

# Test valid input
response = rails.generate(messages=[{
    "role": "user",
    "content": "Can you explain how machine learning works?"
}])
print(response)  # Should provide a response about ML
```
