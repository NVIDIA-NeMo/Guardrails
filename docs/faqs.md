# Frequently Asked Questions (FAQ)

This is an FAQ document. If your question isn't answered here, feel free to open a GitHub issue or ask a question using GitHub Discussions.

## Table of Contents
1. [Can I deploy NeMo Guardrails in a production?](#can-i-deploy-nemo-guardrails-in-production)
2. [How robust are the examples provided in the repo?](#how-robust-are-the-examples-provided-in-the-repo)
3. [What type of information can I add to the knowledge base?](#what-type-of-information-can-i-add-to-the-knowledge-base)
4. [What LLMs are supported by NeMo Guardrails?](#what-llms-are-supported-by-nemo-guardrails)
5. [How well does this work?](#how-well-does-this-work)

---

### Can I deploy NeMo Guardrails in production?

The current alpha release is undergoing active development and may be subject to changes and improvements, which could potentially cause instability and unexpected behavior. We currently do not recommend deploying this alpha version in a production setting. We appreciate your understanding and contribution during this stage.

[Back to top](#table-of-contents)

### How robust are the examples provided in the repo?

The example configurations are meant to be educational. Their purpose is to showcase the core behavior of the toolkit. To achieve a high degree of robustness, the guardrails configurations should be extended through careful application design along with iterative testing and refinement.

[Back to top](#table-of-contents)

### What type of information can I add to the knowledge base?

The knowledge base is designed for question answering on non-sensitive information (e.g., not including PII, PHI). The knowledge base's content is chunked, and any part of it can end up in the prompt(s) sent to the LLM. In any responsible security architecture, sensitive information should not be included in any source that would be exposed to the LLM.

[Back to top](#table-of-contents)

### What LLMs are supported by NeMo Guardrails?

Technically, you can connect a guardrails configuration to any LLM provider that is supported by LangChain (e.g., `ai21`, `aleph_alpha`, `anthropic`, `anyscale`, `azure`, `cohere`, `huggingface_endpoint`, `huggingface_hub`, `openai`, `self_hosted`, `self_hosted_hugging_face` - check out the LangChain official documentation for the full list) or to any [custom LLM](user-guides/configuration-guide.md#custom-llm-models). Depending on the capabilities of the LLM, some will work better than others. We are performing evaluations, and we will share more details soon.

Changes to some configuration elements can help improve compatibility with a given LLM provider or custom LLM, including the general instructions or prompt templates. This is essentially prompt engineering, and it is an imperfect process. As the capabilities of various LLMs evolve in the future, we expect this process to get easier.

[Back to top](#table-of-contents)

### How well does this work?

We'll be putting out a more fulsome evaluation soon, breaking down the components like canonical form generation, flow generation, safety rail accuracy, and so forth.

[Back to top](#table-of-contents)
