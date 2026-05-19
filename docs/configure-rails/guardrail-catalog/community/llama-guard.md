# Llama-Guard Integration

NeMo Guardrails provides out-of-the-box support for content moderation using Meta's [Llama Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) model.

In our testing, we observe significantly improved input and output content moderation performance compared to the [self-check method](../self-check.md). Please see the [performance evaluation](llamaguard-based-moderation-rails-performance) for benchmark numbers.

## Usage

To configure your bot to use Llama Guard for input/output checking, follow the below steps:

1. Add a model of type `llama_guard` to the models section of the `config.yml` file. The example below serves Llama Guard with vLLM. Because vLLM exposes an OpenAI-compatible API, `engine: openai` plus `parameters.base_url` reaches it through NeMo Guardrails' built-in client with no LangChain dependency. For background, see [Migrating to 0.22](../../../migration/0.22.md).

    ```yaml
    models:
      ...

      - type: llama_guard
        engine: openai
        model: meta-llama/LlamaGuard-7b
        parameters:
          base_url: "http://localhost:5123/v1"
          api_key: EMPTY
    ```

    :::{note}
    Set `api_key: EMPTY` (or any non-empty placeholder) when self-hosted vLLM does not enforce auth. If your deployment requires a real token, replace `api_key: EMPTY` with the literal token value, or omit `api_key` and set `api_key_env_var` at the **top level** of the model entry (not inside `parameters:`):

    ```yaml
    - type: llama_guard
      engine: openai
      model: meta-llama/LlamaGuard-7b
      api_key_env_var: MY_LLAMA_GUARD_API_KEY
      parameters:
        base_url: "http://localhost:5123/v1"
    ```
    :::

2. Include the `llama guard check input` and `llama guard check output` flow names in the rails section of the `config.yml` file:

    ```yaml
    rails:
      input:
        flows:
          - llama guard check input
      output:
        flows:
          - llama guard check output
    ```

3. Define the `llama_guard_check_input` and the `llama_guard_check_output` prompts in the [`prompts.yml`](https://raw.githubusercontent.com/NVIDIA-NeMo/Guardrails/refs/heads/develop/examples/configs/llama_guard/prompts.yml) file.

    ```yaml
    prompts:
      - task: llama_guard_check_input
        content: |
          <s>[INST] Task: ...
          <BEGIN UNSAFE CONTENT CATEGORIES>
          O1: ...
          O2: ...
      - task: llama_guard_check_output
        content: |
          <s>[INST] Task: ...
          <BEGIN UNSAFE CONTENT CATEGORIES>
          O1: ...
          O2: ...
    ```

The rails execute the [`llama_guard_check_*` actions](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/nemoguardrails/library/llama_guard/actions.py), which return `True` if the user input or the bot message should be allowed, and `False` otherwise, along with a list of the unsafe content categories as defined in the Llama Guard prompt.

```text
define flow llama guard check input
  $llama_guard_response = execute llama_guard_check_input
  $allowed = $llama_guard_response["allowed"]
  $llama_guard_policy_violations = $llama_guard_response["policy_violations"]

  if not $allowed
    bot refuse to respond
    stop

# (similar flow for checking output)
```

A complete example configuration that uses Llama Guard for input and output moderation is provided in this [example folder](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/llama_guard/README.md).
