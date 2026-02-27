---
title:
  page: "Content Safety"
  nav: "Content Safety"
description: "Reference for pre-built content safety guardrails for protecting against violence, criminal activity, hate speech, sexually explicit content, and similar areas."
topics: ["Configuration", "Content Safety"]
tags: ["Rails", "Content Safety", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Content Safety

The content safety checks inside Guardrails act as a robust set of guardrails designed to ensure the integrity and safety of both input and output text. This feature allows users to utilize a variety of advanced content safety models such as Nvidia's [Nemotron Content Safety](https://docs.nvidia.com/nim/#nemoguard) model, Meta's [Llama Guard 3](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/), and Google's [ShieldGemma](https://ai.google.dev/gemma/docs/shieldgemma).

To use the content safety check, you should:

1. Include the desired content safety models in the models section of the `config.yml` file:

    ```yaml
    models:
      - type: main
        engine: openai
        model: gpt-3.5-turbo-instruct

      - type: "content_safety"
        engine: nim
        parameters:
          base_url: "http://localhost:8123/v1"
          model_name: "llama-3.1-nemoguard-8b-content-safety"

      - type: llama_guard_2
        engine: vllm_openai
        parameters:
          openai_api_base: "http://localhost:5005/v1"
          model_name: "meta-llama/Meta-Llama-Guard-2-8B"
    ```

    ```{note}
    The `type` is a unique identifier for the model that will be passed to the input and output rails as a parameter.
    ```

2. Include the content safety check in the input and output rails section of the `config.yml` file:

    ```yaml
    rails:
      input:
        flows:
          - content safety check input $model=content_safety
      output:
        flows:
          - content safety check output $model=content_safety
    ```

    You must define the models in the `models` section of the `config.yml` file before using them in the input and output flows. The `content safety check input` and `content safety check output` flows are used to check the input and output text, respectively. The `$model` parameter specifies the model to be used for content safety checking. The model must be defined in the `models` section of the `config.yml` file. The `content safety check input` and `content safety check output` flows return a boolean value indicating whether the input or output text is safe. Depending on the model, it also returns a set of policy violations. Please refer to the [content safety example](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/content_safety/README.md) for more details.

3. Specify the prompts for each content safety check flow in the `prompts.yml` file, here is the example prompt for the `shieldgemma` model:

    ```yaml
    prompts:
      - task: content_safety_check_input $model=shieldgemma
        content: |
          You are a policy expert trying to help determine whether a user prompt is in
          violation of the defined safety policies.

          <start_of_turn>
          Human Question: {{ user_input }}
          <end_of_turn>

          Our safety principle is defined in the below:

          * "No Hate Speech": The prompt shall not contain or seek generation of content
            that expresses, incites, or promotes hate based on race, gender, ethnicity,
            religion, nationality, sexual orientation, disability status, or caste.
          * "No Dangerous Content": The prompt shall not contain or seek generation of content that harming oneself
            and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).
          * "No Sexually Explicit Information": The prompt shall not contain or seek generation of content
            that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions,
            content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted.
          * "No Harassment": The prompt shall not contain or seek generation of content that is malicious,
            intimidating, bullying, or abusive content targeting another individual (e.g., physical threats,
            denial of tragic events, disparaging victims of violence).
          ===

          Does the human prompt/question violate the above principle? Your answer must start with
          'Yes' or 'No'. And then walk through step by step to be sure we answer
          correctly.
        output_parser: is_content_safe
    ```

    ```{warning}
    If a prompt is not defined, an exception will be raised when the configuration is loaded.
    ```

4. You must specify the output parser. You can use your own parser and register it or use the off-the-shelf `is_content_safe` output parser as shown above.

    This parser works by checking for specific keywords in the response:
    - If the response includes "safe", the content is considered safe.
    - If the response includes "unsafe" or "yes", the content is considered unsafe.
    - If the response includes "no", the content is considered safe.

    ```{note}
    If you're using this function for a different task with a custom prompt, you'll need to update the logic to fit the new context. In this case, "yes" means the content should be blocked, is unsafe, or breaks a policy, while "no" means the content is safe and doesn't break any policies.
    ```

The above is an example prompt that you can use with the *content safety check input $model=shieldgemma*. The prompt has one input variable: `{{ user_input }}`, which includes user input that should be moderated. The completion must be "yes" if the response is not safe and "no" otherwise. Optionally, some models may return a set of policy violations.

The `content safety check input` and `content safety check output` rails executes the [`content_safety_check_input`](../../../nemoguardrails/library/content_safety/actions.py) and [`content_safety_check_output`](../../../nemoguardrails/library/content_safety/actions.py) actions respectively.

## Multilingual Refusal Messages

<!-- TODO: should we mention nvidia/llama-3.1-nemotron-safety-guard-8b-v3  -->
When content safety rails block unsafe content, you can configure the NeMo Guardrails library to automatically detect the user's input language and return refusal messages in that same language. This provides a better user experience for multilingual applications.

### Supported Languages

The multilingual feature supports 9 languages:

| Language | Code | Default Refusal Message |
|----------|------|-------------------------|
| English | `en` | I'm sorry, I can't respond to that. |
| Spanish | `es` | Lo siento, no puedo responder a eso. |
| Chinese | `zh` | 抱歉，我无法回应。 |
| German | `de` | Es tut mir leid, darauf kann ich nicht antworten. |
| French | `fr` | Je suis désolé, je ne peux pas répondre à cela. |
| Hindi | `hi` | मुझे खेद है, मैं इसका जवाब नहीं दे सकता। |
| Japanese | `ja` | 申し訳ありませんが、それには回答できません。 |
| Arabic | `ar` | عذراً، لا أستطيع الرد على ذلك. |
| Thai | `th` | ขออภัย ฉันไม่สามารถตอบได้ |

If the detected language is not in this list, English is used as the fallback.

### Installation

To use multilingual refusal messages, install the NeMo Guardrails library with the `multilingual` extra:

```bash
pip install nemoguardrails[multilingual]
```

### Usage

To enable multilingual refusal messages, add the `multilingual` configuration to your `config.yml`:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.3-70b-instruct

  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemotron-safety-guard-8b-v3

rails:
  config:
    content_safety:
      multilingual:
        enabled: true

  input:
    flows:
      - content safety check input $model=content_safety

  output:
    flows:
      - content safety check output $model=content_safety
```

### Custom Refusal Messages

You can customize the refusal messages for each language:

```yaml
rails:
  config:
    content_safety:
      multilingual:
        enabled: true
        refusal_messages:
          en: "Sorry, I cannot help with that request."
          es: "Lo siento, no puedo ayudar con esa solicitud."
          zh: "抱歉，我无法处理该请求。"
          # Add other languages as needed
```

If a custom message is not provided for a detected language, the built-in default message for that language is used.

<!--  TODO: shall we include it here -->
### How It Works

When `multilingual.enabled` is set to `true`:

1. The `detect_language` action uses the [fast-langdetect](https://github.com/LlmKira/fast-langdetect) library to detect the language of the user's input
2. If the content safety check blocks the input, the refusal message is returned in the detected language
3. Language detection adds minimal latency (~12μs per request)

### Cold Start Behavior

The fast-langdetect library downloads a language detection model on first use:

| Model | Download Size | [Memory Usage](https://github.com/LlmKira/fast-langdetect?tab=readme-ov-file#memory-note) | First Call Behavior |
|-------|---------------|--------------|---------------------|
| `auto` (default) | 125 MB | ~170-210 MB | Downloads model on first call if not cached |
| `lite` | ~0.9 MB (bundled) | ~45-60 MB | No download, works offline immediately |

**Default cache location:**

fast-langdetect stores its downloaded FastText model in a temporary, OS-specific cache directory at `{system_temp_dir}/fasttext-langdetect/`, where `system_temp_dir` is whatever directory your operating system uses for temporary files:

- **macOS**: A sandboxed temp path such as `/var/folders/<random>/T/fasttext-langdetect/`
- **Linux**: The global temp directory `/tmp/fasttext-langdetect/`
- **Windows**: The user's temporary directory, e.g., `C:\Users\<User>\AppData\Local\Temp\fasttext-langdetect\`

You can override this location via the `FTLANG_CACHE` environment variable.

**Production considerations:**

- First API call may take ~10-20 seconds to download and load the full model (network-dependent)
- Subsequent calls use the cached model with ~9-12μs latency
- For container/serverless environments, consider pre-warming during startup or persisting the model cache in your container image

### Accuracy

Language detection accuracy was benchmarked on two datasets:

| Dataset | Samples | Accuracy |
|---------|---------|----------|
| [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification) | 40,500 | 99.71% |
| [nvidia/Nemotron-Safety-Guard-Dataset-v3](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3) | 336,283 | 99.35% |

## Llama Guard-based Content Moderation

TODO: is this covered by the general content safety abstraction?

The NeMo Guardrails library provides out-of-the-box support for content moderation using Meta's [Llama Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) model.

### Example usage

```yaml
rails:
  input:
    flows:
      - llama guard check input
  output:
    flows:
      - llama guard check output
```

For more details, check out the [Llama-Guard Integration](community/llama-guard.md) page.

# Third-party Content Safety APIs

NeMo Guardrails integrates with a collection of third-party managed services which offer content safety guardrails. These include:

- [ActiveFence](community/active-fence.md)
- [AutoAlign](community/auto-align.md)
- [Clavata](community/clavata.md)
- [GCP Text Moderation](community/gcp-text-moderations.md)
- [Guardrails AI](community/guardrails-ai.md)
- [Fiddler Guardrails](community/fiddler.md)
- [Prompt Security](community/prompt-security.md)
- [Pangea (Crowdstrike) AI Guard](community/pangea.md)

See the above reference pages or [Third-Party APIs](./third-party.md) for more information.
