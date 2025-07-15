# Hello World Bot

This guardrails configuration showcases a basic configuration that controls the greeting behavior and refuses to talk about politics and the stock market.

## Test

To test this configuration you can use the CLI Chat by running the following command from the `examples/bots/hello_world` folder:

```bash
$ nemoguardrails chat
```

```
Starting the chat (Press Ctrl+C to quit) ...

> Hello there!
Hello World!
How are you doing?

> What is the capital of France?
The capital of france is Paris.

> And how many people live there?
According to the latest estimates, the population of Paris is around 2.2 million people.
```

To understand in more detail how this was built, check out the [Hello World Guide](../../../docs/getting-started/1-hello-world).
