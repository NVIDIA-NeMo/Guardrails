# Self Check Domain Hallucination

Detects hallucinated (fabricated) URLs, domains, and GitHub repository
references in bot responses.

## How it differs from existing rails

| Rail | What it checks | Method |
| --- | --- | --- |
| `self_check_hallucination` | General hallucination | Self-consistency (multiple LLM samples) |
| `self_check_facts` | Factual accuracy | Evidence grounding (RAG chunks) |
| **`self_check_domain_hallucination`** | **Fabricated URLs / domains / repos** | **Entity extraction + LLM domain-trust evaluation** |

General hallucination checks miss domain-specific fabrications because the
LLM can consistently hallucinate the same fake URL across multiple samples.
Fact-checking rails require retrieval context and do not focus on whether
external references are real.  This rail fills that gap by extracting
URLs, domains, and GitHub repository references from the bot response and
prompting the LLM to evaluate each one for signs of fabrication.

## Checks performed by the LLM

The prompt guides the LLM through five verification dimensions:

1. **Domain existence** - Is the domain well-known and real?
2. **GitHub repository verification** - Does the owner/repo combination match a known project?
3. **URL path plausibility** - Does the path structure match the real site?
4. **Typosquatting detection** - Does the name closely resemble a real domain with subtle misspellings?
5. **Suspicious patterns** - Are there invented subdomains, unofficial mirrors, or unusual name combinations?

## Usage

### Colang 2.0

```yaml
# config.yml
rails:
  output:
    flows:
      - self check domain hallucination
```

### Colang 1.0

```yaml
# config.yml
rails:
  output:
    flows:
      - self check domain hallucination
```

### Warning mode (non-blocking)

```yaml
rails:
  output:
    flows:
      - self check domain hallucination warning
```

## Prompt customization

Override the default prompt in your `prompts.yml`:

```yaml
prompts:
  - task: self_check_domain_hallucination
    content: |
      Your custom prompt here ...
```
