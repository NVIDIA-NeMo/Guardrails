---
title:
  page: "Guardrails Sequence Diagrams"
  nav: "Sequence Diagrams"
description: "View sequence diagrams showing input, retrieval, dialog, execution, and output stages."
keywords: ["guardrails flow", "sequence diagram", "rails process"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "security_for_ai", "nlp"]
content:
  type: concept
  difficulty: technical_intermediate
  audience: [engineer, data_scientist]
---

# Guardrails Sequence Diagrams

This guide provides an overview of the process of invoking guardrails.

The following diagram depicts the guardrails process in detail:

```{image} ../_static/puml/master_rails_flow.png
:alt: "Sequence diagram showing the complete guardrails process in NeMo Guardrails: 1) Input Validation stage where user messages are processed by input rails that can use actions and LLM to validate or alter input, 2) Dialog stage where messages are processed by dialog rails that can interact with a knowledge base, use retrieval rails to filter retrieved information, and use execution rails to perform custom actions, 3) Output Validation stage where bot responses are processed by output rails that can use actions and LLM to validate or alter output. The diagram shows all optional components and their interactions, including knowledge base queries, custom actions, and LLM calls at various stages."
:width: 800px
:align: center
:class: sd-mb-3
```

The guardrails process has multiple stages that a user message goes through:

1. **Input stage**: The user input is first processed by the input rails. The input rails decide if the input is allowed, or whether it should be altered or rejected.
2. **Retrieval stage**: If configured to use a knowledge base, relevant context is retrieved and then processed by retrieval rails to validate, filter, and transform the retrieved information before it is injected into prompts.
3. **Dialog stage**: If the input is allowed and the configuration contains dialog rails (i.e., at least one user message is defined), then the user message is processed by the dialog flows. This will ultimately result in a bot message.
4. **Execution stage**: If custom actions/tools are defined, the system executes them under the control of execution rails, which govern which actions can run and how their inputs/outputs are validated; results may be incorporated into the response.
5. **Output stage**: The Output rails decide if the output is allowed, or whether it should be altered or rejected.
