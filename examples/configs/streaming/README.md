# Streaming

This is an example configuration that uses streaming. To activate streaming support, you have to set the following key in your `config.yml`:

```yaml
streaming: True
```

**NOTE**: This configuration uses the OpenAI `gpt-6-luna` model. You can replace it with any other chat model you have access to.

## Testing

To test this config, in streaming mode, using the chat CLI, run the following command from the root of the project:

```bash
nemoguardrails chat --config=examples/configs/streaming --streaming
```
