# LlamaGuard Usage Example

This example showcases the use of Meta's [Llama Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) model for content moderation.

The structure of the config folder is the following:

- `config.yml` - The config file holding all the configuration options.
- `prompts.yml` - The config file holding the adjustable content categories to use with Llama Guard.

Please see the docs for more details about the [recommended Llama Guard deployment](./../../../docs/user-guides/advanced/llama-guard-deployment.md#self-hosting-llama-guard-using-vllm) method, the [performance evaluation numbers](./../../../docs/evaluation/README.md#llamaguard-based-moderation-rails-performance), and a [step-by-step explanation](./../../../docs/user-guides/guardrails-library.md#llama-guard-based-content-moderation) of this configuration.
