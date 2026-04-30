# HuggingFace Pipeline with Vicuna models

> **Framework requirement.** This example registers a custom LangChain LLM provider through `nemoguardrails.integrations.langchain` and uses LangChain-only constructs such as `HuggingFacePipelineCompatible(..., model_kwargs=...)`.
> Set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install `pip install langchain langchain-community transformers torch` before running.

This configuration uses the HuggingFace Pipeline LLM with various Vicuna models, including LMSYS and Bloke variants, 7B and 13B, e.g. [vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3).

The `vicuna-7b-v1.3` LLM model has been tested on the topical rails evaluation sets, results are available [here](../../../../nemoguardrails/evaluate/README.md).
There are examples on how to use the models with a HF repo id or from a local path.

In this folder, the guardrails application is very basic, but anyone can change it with any other more complex configuration.

**Disclaimer**: The current results of using `vicuna-7b-v1.3` LLM are promising, but still incipient.
On more complex guardrails apps, this model may not work correctly. Thorough testing and optimizations, including for the prompts, are needed before considering a production deployment.
