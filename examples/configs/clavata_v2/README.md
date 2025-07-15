# Clavata Example

This example demonstrates how to integrate with the [Clavata](https://clavata.ai) API for content moderation.

To test this configuration you can use the CLI Chat by running the following command from the `examples/configs/clavata` directory:

```bash
nemoguardrails chat
```

The structure of the config folder is the following:

- `config.yml` - The config file holding all the configuration options for Clavata integration.
- `rails.co` - This file shows how to use Clavata's rails if you're using `colang v2.x`. In version `2.x` configuration of rails via the `config.yml` file is deprecated and the new approach shown in `rails.co` is something you would include in your flows.

Please see the docs for more details about:

- [Full Clavata integration guide](../../../docs/user-guides/community/clavata.md)
- [Configuration options and setup instructions](../../../docs/user-guides/community/clavata.md#setup)
- [Error handling and best practices](../../../docs/user-guides/community/clavata.md#error-handling)

## Policy Matching vs Label Matching

If you take a peek at the example [config.yml](config.yml), you'll notice that the integration no longer specifies `input` or `output` keys. Instead, the input and output rails are specified in a `rails.co` file as shown [here](./rails.co).

Additionally, because the `$input_text` or `$output_text` is passed directly to the rail, both input and output rails use the same Clavata action flow `clavata check for`. Just pass the text to be evaluated, the policy to use, and, optionally, the specific labels to look for, and the flow will take care of the rest.

Whether specific label matches are necessary will depend on both the policy being used, and what you're trying to accomplish. For most use cases, matching on the policy will likely be sufficient.

**As an example of when you might want to specify labels:**

Suppose you are using the same policy for input and output rails. In this case, the policy might include many labels, with some relevant to input, some relevant to output, and some relevant to both. You can specify the labels for input and output rails to ensure each rail only triggers when appropriate.

> Note: When setting specific labels to match, you can also control whether the logic is `ALL` (meaning all specified labels must be found), or `ANY`, meaning any of the specific labels being found will trigger the rail. The default is `ANY`, but you can control this by setting the `label_match_logic` key in your [rails config](./config.yml).
