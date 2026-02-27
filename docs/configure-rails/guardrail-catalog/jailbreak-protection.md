---
title:
  page: "Jailbreak Protection"
  nav: "Jailbreak Protection"
description: "Reference for jailbreak protection guardrails that help prevent adversarial attempts from bypassing safety measures."
topics: ["Configuration", "Jailbreak"]
tags: ["Rails", "Jailbreak Protection", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Jailbreak Protection

Jailbreak protection helps prevent adversarial attempts from bypassing safety measures and manipulating the LLM into generating harmful or unwanted content.

## Jailbreak Detection

The NeMo Guardrails library supports jailbreak detection using a set of heuristics. Currently, two heuristics are supported:

1. [Length per Perplexity](#length-per-perplexity)
2. [Prefix and Suffix Perplexity](#prefix-and-suffix-perplexity)

Perplexity is a metric that measures how well a language model predicts text. Lower is better, meaning less randomness or surprise. Typically jailbreak attempts result in higher perplexity.

To activate the jailbreak detection heuristics, you first need include the `jailbreak detection heuristics` flow as an input rail:

```yaml
rails:
  input:
    flows:
      - jailbreak detection heuristics
```

Also, you need to configure the desired thresholds in your `config.yml`:

```yaml
rails:
  config:
    jailbreak_detection:
      server_endpoint: "http://0.0.0.0:1337/heuristics"
      length_per_perplexity_threshold: 89.79
      prefix_suffix_perplexity_threshold: 1845.65
```

```{note}
If the `server_endpoint` parameter is not set, the checks will run in-process. This is useful for TESTING PURPOSES ONLY and **IS NOT RECOMMENDED FOR PRODUCTION DEPLOYMENTS**.
```

### Heuristics

#### Length per Perplexity

The *length per perplexity* heuristic computes the length of the input divided by the perplexity of the input. If the value is above the specified threshold (default `89.79`) then the input is considered a jailbreak attempt.

The default value represents the mean length/perplexity for a set of jailbreaks derived from a combination of datasets including [AdvBench](https://github.com/llm-attacks/llm-attacks), [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat/blob/main/README.md), and [JailbreakChat](https://github.com/verazuo/jailbreak_llms), with non-jailbreaks taken from the same datasets and incorporating 1000 examples from [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

The statistics for this metric across jailbreak and non jailbreak datasets are as follows:

|      | Jailbreaks | Non-Jailbreaks |
|------|------------|----------------|
| mean | 89.79      | 27.11          |
| min  | 0.03       | 0.00           |
| 25%  | 12.90      | 0.46           |
| 50%  | 47.32      | 2.40           |
| 75%  | 116.94     | 18.78          |
| max  | 1380.55    | 3418.62        |

Using the mean value of `89.79` yields 31.19% of jailbreaks being detected with a false positive rate of 7.44% on the dataset.
Increasing this threshold will decrease the number of jailbreaks detected but will yield fewer false positives.

**USAGE NOTES**:

- Manual inspection of false positives uncovered a number of mislabeled examples in the dataset and a substantial number of system-like prompts. If your application is intended for simple question answering or retrieval-aided generation, this should be a generally safe heuristic.
- This heuristic in its current form is intended only for English language evaluation and will yield significantly more false positives on non-English text, including code.

#### Prefix and Suffix Perplexity

The *prefix and suffix perplexity* heuristic takes the input and computes the perplexity for the prefix and suffix. If any of the is above the specified threshold (default `1845.65`), then the input is considered a jailbreak attempt.

This heuristic examines strings of more than 20 "words" (strings separated by whitespace) to detect potential prefix/suffix attacks.

The default threshold value of `1845.65` is the second-lowest perplexity value across 50 different prompts generated using [GCG](https://github.com/llm-attacks/llm-attacks) prefix/suffix attacks.
Using the default value allows for detection of 49/50 GCG-style attacks with a 0.04% false positive rate on the "non-jailbreak" dataset derived above.

**USAGE NOTES**:

- This heuristic in its current form is intended only for English language evaluation and will yield significantly more false positives on non-English text, including code.

### Perplexity Computation

To compute the perplexity of a string, the current implementation uses the `gpt2-large` model.

### Model-based Jailbreak Detections

There is currently one available model-based detection, using a random forest-based detector trained on [`Snowflake/snowflake-arctic-embed-m-long`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long) embeddings.

### Setup

The recommended way for using the jailbreak detection heuristics and models is to [deploy the jailbreak detection server](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/docs/user-guides/jailbreak-detection-heuristics/README.md) separately.

For quick testing, you can use the jailbreak detection heuristics rail locally by first installing `transformers` and `torch`.

```bash
pip install transformers torch
```

### Latency

Latency was tested in-process and via local Docker for both CPU and GPU configurations.
For each configuration, we tested the response time for 10 prompts ranging in length from 5 to 2048 tokens.
Inference times for sequences longer than the model's maximum input length (1024 tokens for GPT-2) necessarily take longer.
Times reported below in are **averages** and are reported in milliseconds.

|            | CPU   | GPU |
|------------|-------|-----|
| Docker     | 2057  | 115 |
| In-Process | 3227  | 157 |
