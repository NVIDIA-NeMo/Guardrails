---
title:
  page: "Third-Party APIs"
  nav: "Third-Party APIs"
description: "Reference for third-party API integrations that connect with managed services for guardrail use cases."
topics: ["Configuration", "AI Safety"]
tags: ["Rails", "Third-Party", "Integration", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Third-Party APIs

Third-party APIs connect with managed services for a wide variety of guardrail use cases.
Combine techniques across the guardrail ecosystem for a best-of-breed approach.

## ActiveFence

The NeMo Guardrails library supports using the [ActiveFence ActiveScore API](https://docs.activefence.com/index.html) as an input and output rail out-of-the-box (you need to have the `ACTIVEFENCE_API_KEY` environment variable set).

### Example usage

```yaml
rails:
  input:
    flows:
      - activefence moderation on input
  output:
    flows:
      - activefence moderation on output
```

For more details, check out the [ActiveFence Integration](community/active-fence.md) page.

## PolicyAI

The NeMo Guardrails library supports using [PolicyAI](https://musubilabs.ai) by Musubi Labs as an input and output rail out-of-the-box (you need to have the `POLICYAI_API_KEY` environment variable set).

PolicyAI provides policy-based content moderation, allowing you to define custom policies and organize them with tags for environment-based management.

### Example usage

```yaml
rails:
  input:
    flows:
      - policyai moderation on input
  output:
    flows:
      - policyai moderation on output
```

For more details, check out the [PolicyAI Integration](community/policyai.md) page.

## AutoAlign

The NeMo Guardrails library supports using the AutoAlign's guardrails API (you need to have the `AUTOALIGN_API_KEY` environment variable set).

### Example usage

```yaml
rails:
  input:
    flows:
      - autoalign check input
  output:
    flows:
      - autoalign check output
```

For more details, check out the [AutoAlign Integration](community/auto-align.md) page.

## Clavata

The NeMo Guardrails library supports using [Clavata AI](https://www.clavata.ai/blogs/partner-nvidia) as an input and output rail out-of-the-box (you need to have the CLAVATA_API_KEY environment variable set).

### Example usage

```yaml
rails:
  config:
    clavata:
      policies:
        Fraud: 00000000-0000-0000-0000-000000000000
        Bot_Behavior: 00000000-0000-0000-0000-000000000000
      label_match_logic: ANY

```

For more details, check out the [Clavata Integration](community/clavata.md) page.

## Cleanlab

The NeMo Guardrails library supports using the [Cleanlab Trustworthiness Score API](https://cleanlab.ai/blog/trustworthy-language-model/) as an output rail (you need to have the `CLEANLAB_API_KEY` environment variable set).

### Example usage

```yaml
rails:
  output:
    flows:
      - cleanlab trustworthiness
```

For more details, check out the [Cleanlab Integration](community/cleanlab.md) page.

## GCP Text Moderation

The NeMo Guardrails library supports using the GCP Text Moderation. You need to be authenticated with GCP, refer [here](https://cloud.google.com/docs/authentication/application-default-credentials) for auth details.

### Example usage

```yaml
rails:
  input:
    flows:
      - gcpnlp moderation
```

For more details, check out the [GCP Text Moderation](community/gcp-text-moderations.md) page.

## GuardrailsAI Integration

The NeMo Guardrails library supports using [GuardrailsAI validators](https://github.com/guardrails-ai/guardrails) for comprehensive input and output validation. GuardrailsAI provides a wide range of validators for content safety, PII detection, toxic language filtering, jailbreak detection, and more.

### Example usage

```yaml
rails:
  config:
    guardrails_ai:
      validators:
        - name: toxic_language
          parameters:
            threshold: 0.5
        - name: guardrails_pii
          parameters:
            entities: ["phone_number", "email", "ssn"]
  input:
    flows:
      - guardrailsai check input $validator="guardrails_pii"
  output:
    flows:
      - guardrailsai check output $validator="toxic_language"
```

For more details, check out the [GuardrailsAI Integration](community/guardrails-ai.md) page.

## Fiddler Guardrails for Safety and Hallucination Detection

The NeMo Guardrails library supports using [Fiddler Guardrails](https://docs.fiddler.ai/product-guide/llm-monitoring/guardrails) for safety and hallucination detection in input and output flows.

In order to access Fiddler guardrails, you need access to a valid Fiddler environment, and a [Fiddler environment key](https://docs.fiddler.ai/ui-guide/administration-ui/settings#credentials). You'll need to set the `FIDDLER_API_KEY` environment variable to authenticate into the Fiddler service.

```yaml
rails:
  config:
    fiddler:
      server_endpoint: https://testfiddler.ai # Replace this with your fiddler environment

```

### Example usage

```yaml
rails:
    config:
        fiddler:
            fiddler_endpoint: https://testfiddler.ai # Replace this with your fiddler environment
    input:
        flows:
            - fiddler user safety
    output:
        flows:
            - fiddler bot safety
            - fiddler bot faithfulness

```

For more details, check out the [Fiddler Integration](community/fiddler.md) page.

## Prompt Security Protection

The NeMo Guardrails library supports using [Prompt Security API](https://prompt.security/?utm_medium=github&utm_campaign=nemo-guardrails) for protecting input and output retrieval flows.

To activate the protection, you need to set the `PS_PROTECT_URL` and `PS_APP_ID` environment variables.

### Example usage

```yaml
rails:
  input:
    flows:
      - protect prompt
  output:
    flows:
      - protect response
```

For more details, check out the [Prompt Security Integration](community/prompt-security.md) page.

## CrowdStrike AIDR

The NeMo Guardrails library supports using CrowdStrike AIDR for protecting data and interactions with LLMs within
AI-powered applications.

### Example usage

```yaml
rails:
  input:
    flows:
      - crowdstrike aidr guard input

  output:
    flows:
      - crowdstrike aidr guard output
```

For more details, check out the [CrowdStrike AIDR Integration](community/crowdstrike-aidr.md) page.

## Pangea AI Guard

> **Warning:** The Pangea AI Guard integration is deprecated and will be removed in a future release.
> Users should migrate to the CrowdStrike AIDR integration.

The NeMo Guardrails library supports using [Pangea AI Guard](https://pangea.cloud/services/ai-guard/) for protecting data and
interactions with LLMs within AI-powered applications.

### Example usage

```yaml
rails:
  input:
    flows:
      - pangea ai guard input

  output:
    flows:
      - pangea ai guard output
```

For more details, check out the [Pangea AI Guard Integration](community/pangea.md) page.

## Trend Micro Vision One AI Application Security

The NeMo Guardrails library supports using
[Trend Micro Vision One AI Guard](https://docs.trendmicro.com/en-us/documentation/article/trend-vision-one-ai-scanner-ai-guard) for protecting input and output flows within AI-powered applications.

### Example usage

```yaml
rails:
  input:
    flows:
      - trend ai guard input
  output:
    flows:
      - trend ai guard output
```

For more details, check out the [Trend Micro Vision One AI Application Security](community/trend-micro.md) page.

## Cisco AI Defense

The NeMo Guardrails library supports using [Cisco AI Defense Inspection](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html?utm_medium=github&utm_campaign=nemo-guardrails) for protecting input and output flows.

To activate the protection, you need to set the `AI_DEFENSE_API_KEY` and `AI_DEFENSE_API_ENDPOINT` environment variables.

### Example usage

```yaml
rails:
  input:
    flows:
      - ai defense inspect prompt

  output:
    flows:
      - ai defense inspect response
```

For more details, check out the [Cisco AI Defense Integration](community/ai-defense.md) page.

```{toctree}
:caption: Community Integrations
:hidden:

ActiveFence <community/active-fence>
AlignScore <community/alignscore>
AutoAlign <community/auto-align>
Cisco AI Defense <community/ai-defense>
Clavata <community/clavata>
Cleanlab <community/cleanlab>
CrowdStrike AIDR <community/crowdstrike-aidr>
Fiddler <community/fiddler>
GCP Text Moderation <community/gcp-text-moderations>
GLiNER PII <community/gliner>
GuardrailsAI <community/guardrails-ai>
Llama Guard <community/llama-guard>
Pangea AI Guard <community/pangea>
Patronus Evaluate API <community/patronus-evaluate-api>
Patronus Lynx <community/patronus-lynx>
PolicyAI <community/policyai>
Presidio <community/presidio>
Private AI <community/privateai>
Prompt Security <community/prompt-security>
Regex <community/regex>
Trend Micro <community/trend-micro>
```
