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

| Feature                                            | gpt-3.5-turbo-instruct    | text-davinci-003          | nemollm-43b               | llama-2-13b-chat          | falcon-7b-instruct        | gpt-3.5-turbo             | gpt-4              | gpt4all-13b-snoozy   | vicuna-7b-v1.3       | mpt-7b-instruct      | dolly-v2-3b          | HF Pipeline model                  |
|----------------------------------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|--------------------|----------------------|----------------------|----------------------|----------------------|------------------------------------|
| Dialog Rails                                       | :heavy_check_mark: (0.74) | :heavy_check_mark: (0.83) | :heavy_check_mark: (0.82) | :heavy_check_mark: (0.77) | :heavy_check_mark: (0.76) | :exclamation: (0.45)      | :exclamation:      | :exclamation: (0.54) | :exclamation: (0.54) | :exclamation: (0.50) | :exclamation: (0.40) | :exclamation: _(DEPENDS ON MODEL)_ |
| &bull; Single LLM call                             | :heavy_check_mark: (0.83) | :heavy_check_mark: (0.81) | :heavy_check_mark:        | :x:                       | :x:                       | :x:                       | :x:                | :x:                  | :x:                  | :x:                  | :x:                  | :x:                                |
| &bull; Multi-step flow generation                  | _EXPERIMENTAL_            | _EXPERIMENTAL_            | :x:                       | :x:                       | :x:                       | :x:                       | :x:                | :x:                  | :x:                  | :x:                  | :x:                  | :x:                                |
| Streaming  	                                       | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | -                         | -                         | :heavy_check_mark:        | :heavy_check_mark: | -                    | -                    | -                    | -                    | :heavy_check_mark:                 |
| Hallucination detection (SelfCheckGPT with AskLLM) | :heavy_check_mark:        | :heavy_check_mark:        | :x:                       | :x:                       | :x:                       | :x:                       | :x:                | :x:                  | :x:                  | :x:                  | :x:                  | :x:                                |
| AskLLM rails                                       |                           |                           |                           |                           |                           |                           |                    |                      |                      |                      |                      |                                    |
| &bull; Jailbreak detection                         | :heavy_check_mark: (0.88) | :heavy_check_mark: (0.88) | :heavy_check_mark: (0.86) | :x:                       | :x:                       | :heavy_check_mark: (0.85) | :x:                | :x:                  | :x:                  | :x:                  | :x:                  | :x:                                |
| &bull; Output moderation                           | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :x:                       | :x:                       | :heavy_check_mark: (0.85) | :x:                | :x:                  | :x:                  | :x:                  | :x:                  | :x:                                |
| &bull; Fact-checking                               | :heavy_check_mark: (0.81) | :heavy_check_mark: (0.82) | :heavy_check_mark: (0.81) | :heavy_check_mark: (0.80) | :x:                       | :heavy_check_mark: (0.83) | :x:                | :x:                  | :x:                  | :x:                  | :x:                  | :exclamation: _(DEPENDS ON MODEL)_ |
 | AlignScore fact-checking _(LLM independent)_       | :heavy_check_mark: (0.89) | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:                 |
| ActiveFence moderation _(LLM independent)_         | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:                 |
| Llama Guard moderation _(LLM independent)_         | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:                 |
| Got It AI RAG TruthChecker _(LLM independent)_         | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:                 |
| Patronus Lynx RAG Hallucination detection _(LLM independent)_         | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:                 |
| GCP Text Moderation _(LLM independent)_         | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:                 |
| Patronus Evaluate API _(LLM independent)_         | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark:        | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:                 |


Table legend:
- :heavy_check_mark: - Supported (_The feature is fully supported by the LLM based on our experiments and tests_)
- :exclamation: - Limited Support (_Experiments and tests show that the LLM is under-performing for that feature_)
- :x: - Not Supported (_Experiments show very poor performance or no experiments have been done for the LLM-feature pair_)
- \- - Not Applicable (_e.g. models support streaming, it depends how they are deployed_)

The performance numbers reported in the table above for each LLM-feature pair are as follows:
- the banking dataset evaluation for dialog (topical) rails
- fact-checking using MSMARCO dataset and moderation rails experiments
More details in the [evaluation docs](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/eval/README.md).
