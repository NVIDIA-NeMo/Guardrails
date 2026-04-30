# HuggingFace Pipeline with Dolly models

> **Framework requirement.** This example registers a custom LangChain LLM provider through `nemoguardrails.integrations.langchain` and uses LangChain-only constructs such as `HuggingFacePipelineCompatible(..., model_kwargs=...)` and `AsyncTextIteratorStreamer`.
> Set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install `pip install langchain langchain-community transformers torch` before running.

This configuration uses the HuggingFace Pipeline LLM with the [dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b) model.
It also shows how to support streaming in NeMo Guardrails for LLMs deployed using HuggingFacePipeline.

The `dolly-v2-3b` LLM model has been tested on the topical rails evaluation sets, results are available [here](../../../../nemoguardrails/evaluate/README.md).

In this folder, the guardrails application is very basic, but anyone can change it with any other more complex configuration.

**Disclaimer**: The current results of using `dolly-v2-3b` LLM are promising, but still incipient.
On more complex guardrails apps, this model may not work correctly. Thorough testing and optimizations, including for the prompts, are needed before considering a production deployment.
