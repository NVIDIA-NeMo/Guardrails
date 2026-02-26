---
title:
  page: NeMo Guardrails Library Python API Reference
  nav: Python API
description: Auto-generated Python SDK reference from source code docstrings.
topics:
- AI Safety
- LLM Guardrails
tags:
- Python
- SDK
- API
- Reference
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - AI Engineer
---

# Python API Reference

This reference is auto-generated from source code docstrings.

## Core Classes

The primary entry points for the NeMo Guardrails library.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   nemoguardrails.rails.llm.llmrails.LLMRails
   nemoguardrails.rails.llm.config.RailsConfig
```

## Generation Options and Responses

Control generation behavior and inspect results.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   nemoguardrails.rails.llm.options.GenerationOptions
   nemoguardrails.rails.llm.options.GenerationResponse
   nemoguardrails.rails.llm.options.GenerationRailsOptions
   nemoguardrails.rails.llm.options.GenerationLogOptions
   nemoguardrails.rails.llm.options.GenerationLog
   nemoguardrails.rails.llm.options.GenerationStats
   nemoguardrails.rails.llm.options.ActivatedRail
   nemoguardrails.rails.llm.options.ExecutedAction
```

## Actions

Register custom Python functions as actions for use in guardrails flows.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   nemoguardrails.actions.action
```

## Enumerations

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   nemoguardrails.rails.llm.options.RailType
   nemoguardrails.rails.llm.options.RailStatus
   nemoguardrails.rails.llm.options.RailsResult
```
