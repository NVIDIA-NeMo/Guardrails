# LLM Support

We aim to provide support in NeMo Guardrails for a wide range of LLMs from different providers,
with a focus on open models.
However, due to the complexity of the tasks required for employing dialog rails and most of the predefined
input and output rails (e.g. moderation or  fact-checking), not all LLMs are capable enough to be used.

## Evaluation experiments

This document aims to provide a summary of the evaluation experiments we have employed to assess
the performance of various LLMs for the different type of rails.

For more details about the evaluation of guardrails, including datasets and quantitative results,
please read [this document](../evaluation/README.md).
The tools used for evaluation are described in the same file, for a summary of topics [read this section](../README.md#evaluation-tools) from the user guide.
Any new LLM available in Guardrails should be evaluated using at least this set of tools.

## LLM Support and Guidance

The following tables summarize the LLM support for the main features of NeMo Guardrails, focusing on the different rails available out of the box.
If you want to use an LLM and you cannot see a prompt in the [prompts folder](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/llm/prompts), please also check the configuration defined in the [LLM examples' configurations](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/llm/README.md).

| Feature                                            | gpt-3.5-turbo-instruct    | text-davinci-003          | llama-2-13b-chat          | falcon-7b-instruct        | gpt-3.5-turbo             | gpt-4              | gpt4all-13b-snoozy   | vicuna-7b-v1.3       | mpt-7b-instruct      | dolly-v2-3b          | HF Pipeline model                  |
|----------------------------------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|--------------------|----------------------|----------------------|----------------------|----------------------|------------------------------------|
| Dialog Rails                                       | ✔ (0.74)                  | ✔ (0.83)                  | ✔ (0.77)                  | ✔ (0.76)                  | ❗ (0.45)                  | ❗                  | ❗ (0.54)             | ❗ (0.54)             | ❗ (0.50)             | ❗ (0.40)             | ❗ _(DEPENDS ON MODEL)_             |
| • Single LLM call                                  | ✔ (0.83)                  | ✔ (0.81)                  | ✖                         | ✖                         | ✖                         | ✖                  | ✖                    | ✖                    | ✖                    | ✖                    | ✖                                 |
| • Multi-step flow generation                       | _EXPERIMENTAL_            | _EXPERIMENTAL_            | ✖                         | ✖                         | ✖                         | ✖                  | ✖                    | ✖                    | ✖                    | ✖                    | ✖                                 |
| Streaming                                          | ✔                         | ✔                         | -                         | -                         | ✔                         | ✔                  | -                    | -                    | -                    | -                    | ✔                                 |
| Hallucination detection (SelfCheckGPT with AskLLM) | ✔                         | ✔                         | ✖                         | ✖                         | ✖                         | ✖                  | ✖                    | ✖                    | ✖                    | ✖                    | ✖                                 |
| AskLLM rails                                       |                           |                           |                           |                           |                           |                    |                      |                      |                      |                      |                                    |
| • Jailbreak detection                              | ✔ (0.88)                  | ✔ (0.88)                  | ✖                         | ✖                         | ✔ (0.85)                  | ✖                  | ✖                    | ✖                    | ✖                    | ✖                    | ✖                                 |
| • Output moderation                                | ✔                         | ✔                         | ✖                         | ✖                         | ✔ (0.85)                  | ✖                  | ✖                    | ✖                    | ✖                    | ✖                    | ✖                                 |
| • Fact-checking                                    | ✔ (0.81)                  | ✔ (0.82)                  | ✔ (0.80)                  | ✖                         | ✔ (0.83)                  | ✖                  | ✖                    | ✖                    | ✖                    | ✖                    | ❗ _(DEPENDS ON MODEL)_             |
| AlignScore fact-checking _(LLM independent)_       | ✔ (0.89)                  | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔                                 |
| ActiveFence moderation _(LLM independent)_         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔                                 |
| Llama Guard moderation _(LLM independent)_         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔                                 |
| Got It AI RAG TruthChecker _(LLM independent)_     | ✔                         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔                                 |
| Patronus Lynx RAG Hallucination detection _(LLM independent)_ | ✔                         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔                                 |
| GCP Text Moderation _(LLM independent)_            | ✔                         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔                                 |
| Patronus Evaluate API _(LLM independent)_          | ✔                         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔                                 |
| Fiddler Fast Faitfhulness Hallucination Detection _(LLM independent)_          | ✔                         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔
| Fiddler Fast Safety & Jailbreak Detection _(LLM independent)_          | ✔                         | ✔                         | ✔                         | ✔                         | ✔                         | ✔                  | ✔                    | ✔                    | ✔                    | ✔                    | ✔                     |

Table legend:

- ✔ - Supported (_The feature is fully supported by the LLM based on our experiments and tests_)
- ❗ - Limited Support (_Experiments and tests show that the LLM is under-performing for that feature_)
- ✖ - Not Supported (_Experiments show very poor performance or no experiments have been done for the LLM-feature pair_)
- \- - Not Applicable (_e.g. models support streaming, it depends how they are deployed_)

The performance numbers reported in the table above for each LLM-feature pair are as follows:

- the banking dataset evaluation for dialog (topical) rails
- fact-checking using MSMARCO dataset and moderation rails experiments
More details in the [evaluation docs](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/evaluate/README.md).
