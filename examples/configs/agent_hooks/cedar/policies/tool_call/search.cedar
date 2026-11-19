// The Cedar form of examples/configs/agent_hooks/stop/config.yml, for NeMo Relay's application tool `search`.

// allowed_tools with allow_unlisted_tools: false. Only `search` may be called, and only with `query`: a call that
// passes another argument matches no permit, so it is denied.
@id("search/permit")
permit(principal, action == Hooks::Action::"call_tool", resource == Hooks::Tool::"search")
when { ["query"].containsAll(context.argument_names) };

// The regex rail's job in the stop example: an empty query is denied.
@id("search/empty-query")
forbid(principal, action == Hooks::Action::"call_tool", resource == Hooks::Tool::"search")
unless { context.arguments has query && context.arguments.query != "" };
