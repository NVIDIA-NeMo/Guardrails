# HuggingFace Pipeline with Mosaic ML MPT models

> **Framework requirement.** This example registers a custom LangChain LLM provider through `nemoguardrails.integrations.langchain` and uses LangChain-only constructs such as `HuggingFacePipelineCompatible(..., model_kwargs=...)`.
> Set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install `pip install langchain langchain-community transformers torch` before running.

This configuration uses the HuggingFace Pipeline LLM with the [mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) model.

The `mpt-7b-instruct` LLM model has been tested on the topical rails evaluation sets, results are available [here](../../../../nemoguardrails/evaluate/README.md).

In this folder, the guardrails application is very basic, but anyone can change it with any other more complex configuration.

**Disclaimer**: The current results of using `mpt-7b-instruct` LLM are promising, but still incipient.
On more complex guardrails apps, this model may not work correctly. Thorough testing and optimizations, including for the prompts, are needed before considering a production deployment.
