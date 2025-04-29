# NemoGuard ContentSafety Usage Example

This example showcases the use of NVIDIA's [NemoGuard ContentSafety model](./../../../docs/user-guides/advanced/nemoguard-contentsafety-deployment.md) for topical and dialogue moderation.

The structure of the config folder is the following:

- `config.yml` - The config file holding all the configuration options for the model.
- `prompts.yml` - The config file holding the topical rules used for topical and dialogue moderation by the current guardrail configuration.

Please see the docs for more details about the [recommended ContentSafety deployment](./../../../docs/user-guides/advanced/nemoguard-contentsafety-deployment.md) methods, either using locally downloaded NIMs or NVIDIA AI Enterprise (NVAIE).

Before running this example, please set environment variables `NG_OPENAI_API_KEY` and `NG_NVIDIA_API_KEY` to your OpenAI API Key and Nvidia build.nvidia.com Key as below:

```shell
export NG_OPENAI_API_KEY="<OpenAI API Key>"
export NG_NVIDIA_API_KEY="<NVIDIA API Key>"
```
