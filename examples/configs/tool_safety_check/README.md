# Tool Safety Check Usage Example

This example layers [Tool Safety Check](./../../../docs/configure-rails/guardrail-catalog/tool-safety-check.mdx)
on top of the structural [tool call / tool result validation](./../../../docs/configure-rails/guardrail-catalog/tool-calling.mdx)
rails: `tool call validation` checks every tool call against the declared tools and their JSON
Schema, and `tool result validation` checks every tool result's `tool_call_id` linkage. The
per-tool safety check additionally judges `run_sql`'s call arguments and result content with a
judge model.

The structure of the config folder is the following:

- `config.yml` - The config file declaring the main and judge models; `rails.tool_output.flows`
  / `rails.tool_input.flows` run the structural validation for every tool, and
  `rails.tool_output.per_tool` / `rails.tool_input.per_tool` additionally scope the judged
  safety check to `run_sql`.
- `prompts.yml` - The per-tool judge prompts, one for the tool's call arguments and one for
  its result content.

The `per_tool` dispatch and the tool safety check rail itself require the IORails engine.
This config does not declare the `run_sql` tool: declare it either per request in
`options.llm_params.tools`, or statically on the model in `config.yml`. See
[Declare Tools](./../../../docs/configure-rails/guardrail-catalog/tool-calling.mdx) for both
options.

```python
from nemoguardrails import Guardrails, RailsConfig

config = RailsConfig.from_path("./examples/configs/tool_safety_check")
rails = Guardrails(config, require_iorails=True)

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a read-only SQL query against the application database.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

response = await rails.generate_async(
    messages=[{"role": "user", "content": "List all users"}],
    options={"llm_params": {"tools": tools}},
)
```

If the model returns a tool call, send the result back the same way, with a `role: "tool"`
message answering the call's `tool_call_id`, and call `generate_async` again. See
[Tool Calling](./../../../docs/configure-rails/guardrail-catalog/tool-calling.mdx) for the
full request/response shapes.
