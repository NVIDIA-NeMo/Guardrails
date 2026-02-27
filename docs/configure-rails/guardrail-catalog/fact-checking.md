---
title:
  page: "Hallucinations & Fact-Checking"
  nav: "Hallucinations & Fact-Checking"
description: "Reference for fact-checking and hallucination detection guardrails that ensure LLM output is grounded in evidence."
topics: ["Configuration", "AI Safety"]
tags: ["Rails", "Fact-Checking", "Hallucination", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Hallucinations & Fact-Checking

Fact-checking guardrails help ensure that LLM output is well grounded in evidence and reduce so-called hallucinations or false claims.

## Self-Check Fact-Checking

The goal of the self-check fact-checking output rail is to ensure that the answer to a RAG (Retrieval Augmented Generation) query is grounded in the provided evidence extracted from the knowledge base (KB).

The NeMo Guardrails library uses the concept of **relevant chunks** (which are stored in the `$relevant_chunks` context variable) as the evidence against which fact-checking should be performed. The relevant chunks can be extracted automatically, if the built-in knowledge base support is used, or provided directly alongside the query.

```{important}
The performance of this rail is strongly dependent on the capability of the LLM to follow the instructions in the `self_check_facts` prompt.
```

### Usage

To use the self-check fact-checking rail, you should:

1. Include the `self check facts` flow name in the output rails section of the `config.yml` file:

    ```yaml
    rails:
      output:
        flows:
          - self check facts
    ```

2. Define the `self_check_facts` prompt in the `prompts.yml` file:

    ```yaml
    prompts:
      - task: self_check_facts
        content: |-
          You are given a task to identify if the hypothesis is grounded and entailed to the evidence.
          You will only use the contents of the evidence and not rely on external knowledge.
          Answer with yes/no. "evidence": {{ evidence }} "hypothesis": {{ response }} "entails":
    ```

    ```{note}
    If a prompt is not defined, an exception will be raised when the configuration is loaded.
    ```

The above is an example prompt that you can use with the *self check facts rail*. The `self_check_facts` prompt has two input variables: `{{ evidence }}`, which includes the relevant chunks, and `{{ response }}`, which includes the bot response that should be fact-checked. The completion must be "yes" if the response is factually correct and "no" otherwise.

The self-check fact-checking rail executes the [`self_check_facts` action](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/nemoguardrails/library/self_check/output_check/actions.py), which returns a score between `0.0` (response is not accurate) and `1.0` (response is accurate). The reason a number is returned, instead of a boolean, is to keep a consistent API with other methods that return a score, e.g., the AlignScore method below.

```text
define subflow self check facts
  if $check_facts == True
    $check_facts = False

    $accuracy = execute self_check_facts
    if $accuracy < 0.5
      bot refuse to respond
      stop
```

To trigger the self-check fact-checking rail for a bot message, you must set the `$check_facts` context variable to `True` before a bot message requiring fact-checking. This enables you to explicitly enable fact-checking only when needed (e.g. when answering an important question vs. chitchat).

The example below will trigger the fact-checking output rail every time the bot responds to a question about the report.

```text
define flow
  user ask about report
  $check_facts = True
  bot provide report answer
```

### Usage in combination with a custom RAG

Fact-checking also works in a custom RAG implementation based on a custom action:

```text
define flow answer report question
  user ...
  $answer = execute rag()
  $check_facts = True
  bot $answer
```

Please refer to the [Custom RAG Output Rails example](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/rag/custom_rag_output_rails/README.md).

## Hallucination Detection

The goal of the hallucination detection output rail is to protect against false claims (also called "hallucinations") in the generated bot message. While similar to the fact-checking rail, hallucination detection can be used when there are no supporting documents (i.e., `$relevant_chunks`).

### Usage

To use the hallucination rail, you should:

1. Include the `self check hallucination` flow name in the output rails section of the `config.yml` file:

    ```yaml
    rails:
      output:
        flows:
          - self check hallucination
    ```

2. Define a `self_check_hallucination` prompt in the `prompts.yml` file:

    ```yaml
    prompts:
      - task: self_check_hallucination
        content: |-
          You are given a task to identify if the hypothesis is in agreement with the context below.
          You will only use the contents of the context and not rely on external knowledge.
          Answer with yes/no. "context": {{ paragraph }} "hypothesis": {{ statement }} "agreement":
    ```

    ```{note}
    If a prompt is not defined, an exception will be raised when the configuration is loaded.
    ```

The above is an example prompt you can use with the *self check hallucination rail*. The `self_check_hallucination` prompt has two input variables: `{{ paragraph }}`, which represents alternative generations for the same user query, and `{{ statement }}`, which represents the current bot response. The completion must be "yes" if the statement is not a hallucination (i.e., agrees with alternative generations) and "no" otherwise.

You can use the self-check hallucination detection in two modes:

1. **Blocking**: block the message if a hallucination is detected.
2. **Warning**: warn the user if the response is prone to hallucinations.

### Blocking Mode

Similar to self-check fact-checking, to trigger the self-check hallucination rail in blocking mode, you have to set the `$check_hallucination` context variable to `True` to verify that a bot message is not prone to hallucination:

```text
define flow
  user ask about people
  $check_hallucination = True
  bot respond about people
```

The above example will trigger the hallucination rail for every people-related question (matching the canonical form `user ask about people`), which is usually more prone to contain incorrect statements. If the bot message contains hallucinations, the default `bot inform answer unknown` message is used. To override it, include the following in one of your Colang files:

```text
define bot inform answer unknown
  "I don't know the answer to that."
```

### Warning Mode

Similar to above, if you want to allow sending the response back to the user, but with a warning, you have to set the `$hallucination_warning` context variable to `True`.

```text
define flow
  user ask about people
  $hallucination_warning = True
  bot respond about people
```

To override the default message, include the following in one of your Colang files:

```text
define bot inform answer prone to hallucination
  "The previous answer is prone to hallucination and may not be accurate."
```

### Usage in combination with a custom RAG

Hallucination-checking also works in a custom RAG implementation based on a custom action:

```text
define flow answer report question
  user ...
  $answer = execute rag()
  $check_hallucination = True
  bot $answer
```

Please refer to the [Custom RAG Output Rails example](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/rag/custom_rag_output_rails/README.md).

### Implementation Details

The implementation for the self-check hallucination rail uses a slight variation of the [SelfCheckGPT paper](https://arxiv.org/abs/2303.08896):

1. First, sample several extra responses from the LLM (by default, two extra responses).
2. Use the LLM to check if the original and extra responses are consistent.

Similar to the self-check fact-checking, we formulate the consistency checking similar to an NLI task with the original bot response as the *hypothesis* (`{{ statement }}`) and the extra generated responses as the context or *evidence* (`{{ paragraph }}`).

## AlignScore-based Fact-Checking

The NeMo Guardrails library provides out-of-the-box support for the [AlignScore metric (Zha et al.)](https://aclanthology.org/2023.acl-long.634.pdf), which uses a RoBERTa-based model for scoring factual consistency in model responses with respect to the knowledge base.

### Example usage

```yaml
rails:
  config:
    fact_checking:
      parameters:
        # Point to a running instance of the AlignScore server
        endpoint: "http://localhost:5000/alignscore_large"

  output:
    flows:
      - alignscore check facts
```

For more details, check out the [AlignScore Integration](community/alignscore.md) page.

## Patronus Lynx-based RAG Hallucination Detection

The NeMo Guardrails library supports hallucination detection in RAG systems using [Patronus AI](https://www.patronus.ai)'s Lynx model. The model is hosted on Hugging Face and comes in both a 70B parameters (see [here](https://huggingface.co/PatronusAI/Patronus-Lynx-70B-Instruct)) and 8B parameters (see [here](https://huggingface.co/PatronusAI/Patronus-Lynx-8B-Instruct)) variant.

### Example usage

```yaml
rails:
  output:
    flows:
      - patronus lynx check output hallucination
```

For more details, check out the [Patronus Lynx Integration](community/patronus-lynx.md) page.
